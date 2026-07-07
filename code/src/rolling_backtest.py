import argparse
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from config import config
from positioning import choose_positioning, compute_market_snapshot, compute_score_profile
from tabular_ranker import (
    DEFAULT_COMPONENT_WEIGHTS,
    DEFAULT_HEURISTIC_WEIGHTS,
    _add_blended_score,
    add_strategy_selector_score,
    copy_selector_auxiliary_columns,
    _prepare_training_table,
    build_tabular_features,
    fit_component_models,
)


def normalize_stock_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """统一股票代码、日期和数值列格式，便于多个窗口复用。"""
    df = raw_df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])
    for col in ["开盘", "收盘", "最高", "最低"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["股票代码", "日期"]).reset_index(drop=True)


def pick_cutoff_indices(dates: List[pd.Timestamp], windows: int, step: int) -> List[int]:
    """选择最近的若干个模拟提交日；每个提交日后必须保留5个交易日用于评分。"""
    max_cutoff_idx = len(dates) - 6
    if max_cutoff_idx <= 70:
        raise ValueError("可回测交易日太少，无法构造滚动窗口")

    indices = []
    idx = max_cutoff_idx
    while idx > 70 and len(indices) < windows:
        indices.append(idx)
        idx -= step
    return sorted(indices)


def fit_tabular_models(
    train_table: pd.DataFrame,
    feature_columns: List[str],
    seed: int,
) -> Dict[str, object]:
    return fit_component_models(train_table, feature_columns, seed)


def predict_one_cutoff(
    featured: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    feature_columns: List[str],
    medians: pd.Series,
    models: Dict[str, object],
    component_weights: Dict[str, float],
    heuristic_weights: Dict[str, float],
) -> pd.DataFrame:
    latest = featured[featured["日期"].eq(cutoff_date)].copy()
    if latest.empty:
        raise ValueError(f"{cutoff_date.date()} 没有可预测股票")

    for col in feature_columns:
        if col not in latest.columns:
            latest[col] = 0.0
    latest[feature_columns] = latest[feature_columns].replace([np.inf, -np.inf], np.nan)
    latest[feature_columns] = latest[feature_columns].fillna(medians).fillna(0.0)

    pred = latest[["股票代码", "日期"]].copy()
    for name, model in models.items():
        pred[name] = model.predict(latest[feature_columns])
    for col in heuristic_weights:
        if col in latest.columns:
            pred[col] = latest[col].values
    pred = copy_selector_auxiliary_columns(pred, latest)

    pred = _add_blended_score(pred, component_weights, heuristic_weights)
    pred = add_strategy_selector_score(pred)
    return pred.sort_values("tabular_score", ascending=False).reset_index(drop=True)


def future_return_table(
    raw_df: pd.DataFrame,
    dates: List[pd.Timestamp],
    cutoff_idx: int,
) -> pd.DataFrame:
    """按比赛本地评分口径计算后5个交易日的开盘到开盘收益。"""
    test_dates = dates[cutoff_idx + 1 : cutoff_idx + 6]
    first_date = test_dates[0]
    last_date = test_dates[-1]

    first = raw_df[raw_df["日期"].eq(first_date)][["股票代码", "开盘"]].rename(
        columns={"开盘": "first_open"}
    )
    last = raw_df[raw_df["日期"].eq(last_date)][["股票代码", "开盘"]].rename(
        columns={"开盘": "last_open"}
    )
    ret = first.merge(last, on="股票代码", how="inner")
    ret = ret[ret["first_open"] > 1e-12].copy()
    ret["future_return"] = (ret["last_open"] - ret["first_open"]) / ret["first_open"]
    ret["test_start"] = first_date
    ret["test_end"] = last_date
    return ret


def score_strategy(pred: pd.DataFrame, returns: pd.DataFrame, weights: List[float]) -> float:
    selected = pred.head(len(weights))[["股票代码"]].copy()
    selected["weight"] = weights
    scored = selected.merge(returns[["股票代码", "future_return"]], on="股票代码", how="left")
    scored["future_return"] = scored["future_return"].fillna(0.0)
    return float((scored["future_return"] * scored["weight"]).sum())


def strategy_weights() -> Dict[str, List[float]]:
    return {
        "top3_fixed_full": [0.40, 0.35, 0.25],
        "top2_concentrated": [0.60, 0.40],
        "top1_allin": [1.0],
        "top5_equal": [0.20, 0.20, 0.20, 0.20, 0.20],
    }


def score_strategy_by_column(
    pred: pd.DataFrame,
    returns: pd.DataFrame,
    weights: List[float],
    score_col: str,
) -> float:
    if score_col not in pred.columns:
        return float("nan")
    ranked = pred.sort_values(score_col, ascending=False).reset_index(drop=True)
    return score_strategy(ranked, returns, weights)


def _trimmed_mean(values: pd.Series, trim_ratio: float = 0.10) -> float:
    clean = values.dropna().sort_values().reset_index(drop=True)
    if clean.empty:
        return float("nan")
    cut = int(len(clean) * trim_ratio)
    if cut > 0 and len(clean) > 2 * cut:
        clean = clean.iloc[cut:-cut]
    return float(clean.mean())


def summarize_backtest(
    result: pd.DataFrame,
    strategy_cols: List[str],
    baseline_col: str,
) -> pd.DataFrame:
    rows = []
    for col in strategy_cols:
        values = result[col].dropna()
        if values.empty:
            continue
        without_max = values.drop(values.idxmax()) if len(values) > 1 else values
        baseline = result[baseline_col] if baseline_col in result.columns else pd.Series(index=result.index, dtype=float)
        rows.append(
            {
                "strategy": col,
                "mean_return": float(values.mean()),
                "median_return": float(values.median()),
                "win_rate": float((values > 0).mean()),
                "max_loss": float(values.min()),
                "mean_without_best": float(without_max.mean()),
                "trimmed_mean_10_90": _trimmed_mean(values, 0.10),
                "beat_baseline_ratio": float((result[col] > baseline).mean()) if baseline_col in result.columns else float("nan"),
                "beat_market_ratio": float((result[col] > result["market_mean"]).mean()) if "market_mean" in result.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_return", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling backtest for the tabular ranking branch.")
    parser.add_argument("--windows", type=int, default=8, help="回测窗口数量")
    parser.add_argument("--step", type=int, default=10, help="窗口间隔交易日数")
    parser.add_argument("--output", default="./output/rolling_backtest.csv", help="结果输出CSV")
    args = parser.parse_args()

    data_file = os.path.join(config["data_path"], "stock_data.csv")
    raw_df = normalize_stock_data(pd.read_csv(data_file, dtype={"股票代码": str}))
    dates = sorted(raw_df["日期"].unique())
    cutoff_indices = pick_cutoff_indices(dates, args.windows, args.step)

    print("=== 构造全量历史特征 ===")
    print(f"数据范围: {pd.Timestamp(dates[0]).date()} 至 {pd.Timestamp(dates[-1]).date()}")
    print(f"股票数量: {raw_df['股票代码'].nunique()}, 交易日数量: {len(dates)}")
    feature_start = time.time()
    featured = build_tabular_features(raw_df)
    print(f"特征构造完成，用时 {time.time() - feature_start:.1f}s")

    seed = int(config.get("seed", 20260416))
    min_history_days = int(config.get("tabular_min_history_days", 60))
    component_weights = config.get("tabular_component_weights", DEFAULT_COMPONENT_WEIGHTS)
    heuristic_weights = config.get("tabular_heuristic_weights", DEFAULT_HEURISTIC_WEIGHTS)
    strategies = strategy_weights()

    rows = []
    for turn, cutoff_idx in enumerate(cutoff_indices, start=1):
        cutoff_date = pd.Timestamp(dates[cutoff_idx])
        label_cutoff_date = pd.Timestamp(dates[cutoff_idx - 5])
        test_dates = [pd.Timestamp(d).date().isoformat() for d in dates[cutoff_idx + 1 : cutoff_idx + 6]]

        print(
            f"\n=== Window {turn}/{len(cutoff_indices)} | "
            f"train_asof={cutoff_date.date()} | test={test_dates[0]}~{test_dates[-1]} ==="
        )

        train_featured = featured[featured["日期"].le(label_cutoff_date)].copy()
        train_table, feature_columns, medians = _prepare_training_table(train_featured, min_history_days)
        if train_table.empty:
            print("训练样本为空，跳过")
            continue

        fit_start = time.time()
        models = fit_tabular_models(train_table, feature_columns, seed + turn)
        pred = predict_one_cutoff(
            featured,
            cutoff_date,
            feature_columns,
            medians,
            models,
            component_weights,
            heuristic_weights,
        )
        returns = future_return_table(raw_df, dates, cutoff_idx)
        oracle_top5 = returns.nlargest(5, "future_return")["future_return"].mean()
        market_mean = returns["future_return"].mean()
        final_score_col = config.get("submission_score_col", "tabular_score")
        if final_score_col not in pred.columns:
            final_score_col = "tabular_score"
        snapshot_seed_col = "controlled_hotspot" if "controlled_hotspot" in pred.columns else final_score_col
        snapshot_seed = pred.sort_values(snapshot_seed_col, ascending=False)["股票代码"].head(5).tolist()
        snapshot = compute_market_snapshot(raw_df, cutoff_date, snapshot_seed)
        pred = add_strategy_selector_score(pred, snapshot)
        final_score_col = config.get("submission_score_col", "tabular_score")
        if final_score_col not in pred.columns:
            final_score_col = "tabular_score"
        pred_final = pred.sort_values(final_score_col, ascending=False).reset_index(drop=True)
        snapshot["selector_regime"] = str(pred_final.iloc[0].get("selector_regime", ""))
        score_profile = compute_score_profile(
            pred_final.rename(columns={final_score_col: "final_score"}),
            "final_score",
        )
        position_name, dynamic_weights, snapshot = choose_positioning(snapshot, len(pred), score_profile)

        row = {
            "window": turn,
            "train_asof": cutoff_date.date().isoformat(),
            "label_until": label_cutoff_date.date().isoformat(),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "final_score_col": final_score_col,
            "selector_regime": pred_final.iloc[0].get("selector_regime", ""),
            "top1": pred_final.iloc[0]["股票代码"],
            "top2": pred_final.iloc[1]["股票代码"],
            "top3": pred_final.iloc[2]["股票代码"],
            "fit_seconds": round(time.time() - fit_start, 2),
            "dynamic_position_name": position_name,
            "dynamic_weight_sum": round(float(sum(dynamic_weights)), 6),
            "market_ret_5": snapshot.get("market_ret_5", np.nan),
            "market_ret_20": snapshot.get("market_ret_20", np.nan),
            "up_ratio_5": snapshot.get("up_ratio_5", np.nan),
            "up_ratio_20": snapshot.get("up_ratio_20", np.nan),
            "selected_ret_20_mean": snapshot.get("selected_ret_20_mean", np.nan),
            "oracle_top5_equal": oracle_top5,
            "market_mean": market_mean,
            "dynamic_topk": len(dynamic_weights),
            "dynamic_topk_attack": score_strategy(pred_final, returns, dynamic_weights),
        }
        row["top3_dynamic_attack"] = row["dynamic_topk_attack"]
        for name, weights in strategies.items():
            row[name] = score_strategy(pred_final, returns, weights)
        row["selector_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "selector_score")
        row["selector_consensus_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "selector_consensus")
        row["selector_hot_rotation_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "selector_hot_rotation")
        row["selector_defensive_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "selector_defensive")
        row["controlled_hotspot_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "controlled_hotspot")
        row["lgb_ranker_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "lgb_rank")
        row["lgb_return_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "lgb_return")
        row["hgb_rank_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "hgb_rank")
        row["hgb_return_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "hgb_return")
        row["extra_rank_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "extra_rank")
        row["heuristic_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "heuristic")
        row["hotspot_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "hotspot_strength_rank")
        row["relative_strength_top3"] = score_strategy_by_column(pred, returns, [0.40, 0.35, 0.25], "relative_strength_20_rank")
        rows.append(row)

        print(
            f"Top{row['dynamic_topk']}: "
            f"{row['top1']}, {row['top2']}, {row['top3']} | "
            f"最终列={final_score_col}/{row['selector_regime']}, "
            f"动态={row['dynamic_topk_attack']:.6f}({position_name}), "
            f"三股满仓={row['top3_fixed_full']:.6f}, "
            f"五股均分={row['top5_equal']:.6f}, "
            f"Top1梭哈={row['top1_allin']:.6f}, "
            f"LGBMRanker={row['lgb_ranker_top3']:.6f}, "
            f"市场均值={row['market_mean']:.6f}"
        )

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)

    print("\n=== Summary ===")
    strategy_cols = [
        "dynamic_topk_attack",
        "top3_fixed_full",
        "top2_concentrated",
        "top1_allin",
        "top5_equal",
        "selector_top3",
        "selector_consensus_top3",
        "selector_hot_rotation_top3",
        "selector_defensive_top3",
        "controlled_hotspot_top3",
        "lgb_ranker_top3",
        "lgb_return_top3",
        "hgb_rank_top3",
        "hgb_return_top3",
        "extra_rank_top3",
        "heuristic_top3",
        "hotspot_top3",
        "relative_strength_top3",
    ]
    summary = summarize_backtest(result, strategy_cols, baseline_col="top5_equal")
    summary_path = args.output.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    best_counts = result[strategy_cols].idxmax(axis=1).value_counts()
    print("\nbest_strategy_counts")
    print(best_counts.to_string())
    print(f"\n结果已写入: {args.output}")
    print(f"汇总已写入: {summary_path}")


if __name__ == "__main__":
    main()
