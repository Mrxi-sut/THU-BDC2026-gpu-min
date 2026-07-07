import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

warnings.simplefilter("ignore", PerformanceWarning)

try:
    from lightgbm import LGBMRanker, LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    LGBMRanker = None
    LGBMRegressor = None
    HAS_LIGHTGBM = False


MODEL_FILE_NAME = "tabular_ranker.pkl"
RANDOM_SEED = 20260416
WINDOWS = (2, 3, 5, 10, 20, 40, 60)

NUMERIC_COLUMNS = [
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]

OFFLINE_FACTOR_FILES = (
    "daily_basic_factors.csv",
    "daily_basic_core_factors.csv",
    "moneyflow_factors.csv",
    "moneyflow_residual_factors.csv",
    "index_weight_factors.csv",
    "index_weight_enhanced_factors.csv",
    "industry_factors.csv",
    "industry_core_factors.csv",
    "event_factors.csv",
    "cyq_perf_factors.csv",
    "financial_factors.csv",
)

OFFLINE_FACTOR_TIERS = {
    "first_tier": ("index_weight_factors.csv",),
    "second_tier": ("daily_basic_core_factors.csv",),
    "experimental": (
        "moneyflow_residual_factors.csv",
        "index_weight_enhanced_factors.csv",
        "industry_factors.csv",
        "industry_core_factors.csv",
        "event_factors.csv",
        "cyq_perf_factors.csv",
        "financial_factors.csv",
    ),
    "raw_legacy": ("daily_basic_factors.csv", "moneyflow_factors.csv"),
}

# 这些权重不是收益率预测本身，而是把不同模型/规则的横截面排名融合起来。
DEFAULT_COMPONENT_WEIGHTS = {
    "lgb_rank": 0.30,
    "lgb_return": 0.08,
    "hgb_return": 0.16,
    "hgb_rank": 0.14,
    "extra_rank": 0.10,
    "heuristic": 0.22,
}

DEFAULT_HEURISTIC_WEIGHTS = {
    "hotspot_strength_rank": 0.24,
    "relative_strength_20_rank": 0.16,
    "ret_20_rank": 0.14,
    "ret_10_rank": 0.10,
    "ret_5_rank": 0.08,
    "amount_ratio_20_rank": 0.08,
    "volume_ratio_20_rank": 0.06,
    "turnover_accel_5_20_rank": 0.06,
    "breakout_20_rank": 0.05,
    "board_hotspot_rank": 0.03,
    # 0权重列用于复制给二层 selector，不改变原 heuristic 分支。
    "amount_moderate_20_rank": 0.00,
    "no_overheat_20_rank": 0.00,
}

SELECTOR_COMPONENT_COLUMNS = [
    "hgb_rank",
    "lgb_rank",
    "extra_rank",
    "hgb_return",
    "lgb_return",
    "heuristic",
    "hotspot_strength_rank",
    "controlled_hotspot",
    "relative_strength_20_rank",
    "board_hotspot_rank",
]

SELECTOR_DEBUG_COLUMNS = [
    "selector_score",
    "selector_consensus",
    "selector_hot_rotation",
    "selector_defensive",
    "selector_regime",
]

SELECTOR_GATE_BRANCHES = (
    "selector_consensus",
    "selector_hot_rotation",
    "selector_defensive",
)

SELECTOR_GATE_FEATURE_COLUMNS = (
    "market_ret_5",
    "market_ret_20",
    "up_ratio_5",
    "up_ratio_20",
    "market_drawdown_20",
    "ret5_p95_minus_median",
    "ret20_p95_minus_median",
    "amount_ratio20_p90_p50",
    "top2_vs_top10_gap",
    "selector_hgb_overlap",
)

SELECTOR_AUXILIARY_PREFIXES = (
    "ts_ind_",
    "ts_mfr_",
    "ts_event_",
    "ts_cyq_",
)

SELECTOR_LOCAL_AUXILIARY_COLUMNS = (
    "ret_5_rank",
    "ret_10_rank",
    "ret_20_rank",
    "relative_strength_5_rank",
    "relative_strength_10_rank",
    "relative_strength_20_rank",
    "amount_ratio_5_rank",
    "amount_ratio_10_rank",
    "amount_ratio_20_rank",
    "amount_moderate_20_rank",
    "no_overheat_20_rank",
    "hotspot_strength_rank",
    "controlled_hotspot",
    "near_high_20",
    "near_high_20_rank",
    "overheat_20",
    "overheat_20_rank",
    "price_pos_20",
    "price_pos_20_rank",
    "drawdown_20",
    "drawdown_20_rank",
    "stable_mom_20",
    "stable_mom_20_rank",
    "up_day_ratio_5",
    "up_day_ratio_5_rank",
    "up_day_ratio_20",
    "up_day_ratio_20_rank",
    "ret_accel_5_20",
    "ret_accel_5_20_rank",
    "turnover_accel_5_20",
    "turnover_accel_5_20_rank",
    "volume_price_confirm",
    "volume_price_confirm_rank",
    "amount_price_confirm",
    "amount_price_confirm_rank",
    "breakout_20",
    "breakout_20_rank",
    "breakout_count_20",
    "breakout_count_20_rank",
)


def _board_id_from_code(code: str) -> int:
    """没有行业字段时，用股票代码段做离线可复现的板块代理。"""
    code = str(code).zfill(6)
    if code.startswith("688"):
        return 4  # 科创板
    if code.startswith("300"):
        return 3  # 创业板
    if code.startswith(("002", "003")):
        return 2  # 深市中小盘代码段
    if code.startswith(("000", "001")):
        return 1  # 深市主板代码段
    return 0      # 沪市主板及其他


def _pct_rank(values: pd.Series) -> pd.Series:
    """把任意预测分数压成 0~1 的当日横截面排名，降低不同模型量纲差异。"""
    return values.rank(pct=True, method="average").fillna(0.5)


def copy_selector_auxiliary_columns(pred: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Expose compact offline-factor diagnostics to the final selector/gate layer."""
    out = pred.copy()
    for col in source.columns:
        if col not in SELECTOR_LOCAL_AUXILIARY_COLUMNS and not col.startswith(SELECTOR_AUXILIARY_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(source[col]):
            continue
        out[col] = source[col].values
    return out


def _normalize_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["board_id"] = df["股票代码"].map(_board_id_from_code).astype(float)
    return df.sort_values(["股票代码", "日期"]).reset_index(drop=True)


def _offline_factor_dir() -> Path:
    path = Path(os.environ.get("OFFLINE_FACTOR_DIR", "data/offline_factors"))
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _selected_offline_factor_files() -> Tuple[str, ...]:
    """
    Offline factor admission policy.

    Default production mode only loads first-tier factors that survived
    walk-forward ablation. Research scripts can opt into all/subset modes with
    OFFLINE_FACTOR_POLICY without changing model code.
    """
    policy = os.environ.get("OFFLINE_FACTOR_POLICY", "first_tier").strip().lower()
    if policy in {"", "first", "first_tier", "tier1", "production", "prod"}:
        return OFFLINE_FACTOR_TIERS["first_tier"]
    if policy in {"none", "off", "disable", "disabled", "no_factors"}:
        return tuple()
    if policy in {"all", "full", "p0", "research"}:
        return OFFLINE_FACTOR_FILES
    if policy in {"second", "second_tier", "tier2"}:
        return OFFLINE_FACTOR_TIERS["second_tier"]
    if policy in {"experimental", "third", "third_tier", "tier3"}:
        return OFFLINE_FACTOR_TIERS["experimental"]

    aliases = {
        "daily_basic": "daily_basic_factors.csv",
        "daily_basic_raw": "daily_basic_factors.csv",
        "daily_basic_core": "daily_basic_core_factors.csv",
        "moneyflow": "moneyflow_factors.csv",
        "moneyflow_raw": "moneyflow_factors.csv",
        "moneyflow_residual": "moneyflow_residual_factors.csv",
        "index_weight": "index_weight_factors.csv",
        "index_weight_enhanced": "index_weight_enhanced_factors.csv",
        "industry": "industry_factors.csv",
        "industry_core": "industry_core_factors.csv",
        "event": "event_factors.csv",
        "cyq": "cyq_perf_factors.csv",
        "cyq_perf": "cyq_perf_factors.csv",
        "financial": "financial_factors.csv",
    }
    requested = []
    for item in policy.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        filename = aliases.get(item, item)
        if not filename.endswith(".csv"):
            filename = f"{filename}_factors.csv"
        if filename in OFFLINE_FACTOR_FILES and filename not in requested:
            requested.append(filename)
    if requested:
        return tuple(requested)

    raise ValueError(
        "未知 OFFLINE_FACTOR_POLICY: "
        f"{policy}. 可用: first_tier, none, all, second_tier, experimental, "
        "或逗号分隔的因子族/文件名。"
    )


def _merge_optional_offline_factors(df: pd.DataFrame) -> pd.DataFrame:
    factor_dir = _offline_factor_dir()
    if not factor_dir.exists():
        return df

    merged = df.copy()
    loaded_files = []
    blocked_columns = {
        "股票代码",
        "日期",
        "available_date",
        "publish_datetime",
        "weight_trade_date",
        "weight_available_date",
        "weight_month_end_date",
        "moneyflow_trade_date",
        "ts_idx_weight_age_days",
        "ts_idx_visible_lag_days",
        "ts_idx_asof_valid",
        "ts_code",
        "con_code",
    }
    selected_files = _selected_offline_factor_files()
    for filename in selected_files:
        path = factor_dir / filename
        if not path.exists():
            continue

        factor_df = pd.read_csv(path, dtype={"股票代码": str})
        required = {"股票代码", "日期"}
        if not required.issubset(factor_df.columns):
            raise ValueError(f"离线因子文件缺少身份列: {path}")

        factor_df["股票代码"] = factor_df["股票代码"].astype(str).str.zfill(6)
        factor_df["日期"] = pd.to_datetime(factor_df["日期"], errors="coerce")
        factor_df = factor_df.dropna(subset=["日期"]).copy()

        for visible_col in ("available_date", "publish_datetime"):
            if visible_col in factor_df.columns:
                visible_dates = pd.to_datetime(factor_df[visible_col], errors="coerce")
                factor_df = factor_df[visible_dates.isna() | visible_dates.le(factor_df["日期"])].copy()

        numeric_columns = []
        for col in factor_df.columns:
            if col in blocked_columns:
                continue
            factor_df[col] = pd.to_numeric(factor_df[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(factor_df[col]) and factor_df[col].notna().any():
                numeric_columns.append(col)
        if not numeric_columns:
            continue

        rename_map = {
            col: f"{path.stem}_{col}"
            for col in numeric_columns
            if col in merged.columns
        }
        factor_df = factor_df[["股票代码", "日期"] + numeric_columns].rename(columns=rename_map)
        factor_df = factor_df.drop_duplicates(["股票代码", "日期"], keep="last")
        merged = merged.merge(factor_df, on=["股票代码", "日期"], how="left")
        loaded_files.append(path.name)

    if loaded_files:
        policy = os.environ.get("OFFLINE_FACTOR_POLICY", "first_tier")
        print(f"已合并离线因子[{policy}]: {', '.join(loaded_files)}")
    return merged


@contextmanager
def _temporary_factor_policy(policy: str | None = None, factor_dir: str | None = None):
    old_policy = os.environ.get("OFFLINE_FACTOR_POLICY")
    old_dir = os.environ.get("OFFLINE_FACTOR_DIR")
    try:
        if policy is not None:
            os.environ["OFFLINE_FACTOR_POLICY"] = policy
        if factor_dir is not None:
            os.environ["OFFLINE_FACTOR_DIR"] = factor_dir
        yield
    finally:
        if old_policy is None:
            os.environ.pop("OFFLINE_FACTOR_POLICY", None)
        else:
            os.environ["OFFLINE_FACTOR_POLICY"] = old_policy
        if old_dir is None:
            os.environ.pop("OFFLINE_FACTOR_DIR", None)
        else:
            os.environ["OFFLINE_FACTOR_DIR"] = old_dir


def _add_single_stock_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    eps = 1e-12

    open_ = group["开盘"]
    close = group["收盘"]
    high = group["最高"]
    low = group["最低"]
    volume = group["成交量"]
    amount = group["成交额"]
    turnover = pd.to_numeric(group["换手率"], errors="coerce")

    group["history_days"] = np.arange(1, len(group) + 1)
    group["intraday_strength"] = (close - open_) / (open_ + eps)
    group["range_pct"] = (high - low) / (open_ + eps)
    group["upper_shadow"] = (high - np.maximum(open_, close)) / (open_ + eps)
    group["lower_shadow"] = (np.minimum(open_, close) - low) / (open_ + eps)
    group["vwap_gap"] = (amount / (volume + eps) - close) / (close + eps)
    group["ret_1"] = close.pct_change(1)
    group["open_gap_1"] = open_.pct_change(1)
    group["turnover_chg_1"] = turnover.diff()
    pct_chg = pd.to_numeric(group["涨跌幅"], errors="coerce").fillna(0.0)
    group["limit_up_like"] = (pct_chg >= 9.5).astype(float)
    group["limit_down_like"] = (pct_chg <= -9.5).astype(float)
    group["big_up_day"] = (pct_chg >= 5.0).astype(float)
    group["big_down_day"] = (pct_chg <= -5.0).astype(float)
    group["up_day"] = (group["ret_1"] > 0).astype(float)

    for window in WINDOWS:
        group[f"ret_{window}"] = close.pct_change(window)
        group[f"open_ret_{window}"] = open_.pct_change(window)
        group[f"volatility_{window}"] = group["ret_1"].rolling(window).std()
        group[f"ma_gap_{window}"] = close / (close.rolling(window).mean() + eps) - 1
        group[f"volume_ratio_{window}"] = volume / (volume.rolling(window).mean() + eps)
        group[f"amount_ratio_{window}"] = amount / (amount.rolling(window).mean() + eps)
        group[f"turnover_mean_{window}"] = group["换手率"].rolling(window).mean()
        group[f"turnover_ratio_{window}"] = turnover / (turnover.rolling(window).mean() + eps)
        group[f"limit_up_count_{window}"] = group["limit_up_like"].rolling(window).sum()
        group[f"big_up_count_{window}"] = group["big_up_day"].rolling(window).sum()
        group[f"big_down_count_{window}"] = group["big_down_day"].rolling(window).sum()
        group[f"up_day_ratio_{window}"] = group["up_day"].rolling(window).mean()

        prior_high = high.shift(1).rolling(window).max()
        high_window = high.rolling(window).max()
        low_window = low.rolling(window).min()
        group[f"price_pos_{window}"] = (close - low_window) / (high_window - low_window + eps)
        group[f"drawdown_{window}"] = close / (high_window + eps) - 1
        group[f"breakout_{window}"] = (close > prior_high).astype(float)
        group[f"near_high_{window}"] = close / (prior_high + eps) - 1

    # 动量除以波动，避免只追涨而不看走势质量。
    group["stable_mom_20"] = group["ret_20"] / (group["volatility_20"] + eps)
    group["ret_accel_5_20"] = group["ret_5"] - group["ret_20"]
    group["turnover_accel_5_20"] = group["turnover_mean_5"] - group["turnover_mean_20"]
    group["volume_price_confirm"] = group["ret_5"] * np.log1p(group["volume_ratio_5"].clip(lower=0))
    group["amount_price_confirm"] = group["ret_5"] * np.log1p(group["amount_ratio_5"].clip(lower=0))
    group["breakout_count_20"] = group["breakout_20"].rolling(5).sum()
    group["overheat_20"] = (
        (group["price_pos_20"] > 0.9) &
        (group["ret_20"] > 0.2) &
        (group["volatility_20"] > group["volatility_20"].rolling(60).median())
    ).astype(float)

    # 比赛评分口径：预测日后的第1个交易日开盘买入，第5个交易日开盘卖出。
    group["future_return"] = (open_.shift(-5) - open_.shift(-1)) / (open_.shift(-1) + eps)
    return group


def _add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    day_group = df.groupby("日期")
    for window in (3, 5, 10, 20):
        ret_col = f"ret_{window}"
        if ret_col in df.columns:
            df[f"market_ret{window}_mean"] = day_group[ret_col].transform("mean")
            df[f"market_ret{window}_std"] = day_group[ret_col].transform("std")
            df[f"market_breadth{window}"] = day_group[ret_col].transform(lambda s: (s > 0).mean())
            df[f"relative_strength_{window}"] = df[ret_col] - df[f"market_ret{window}_mean"]

    board_group = df.groupby(["日期", "board_id"])
    if "ret_20" in df.columns:
        df["board_ret20_mean"] = board_group["ret_20"].transform("mean")
        df["board_relative_strength_20"] = df["ret_20"] - df["board_ret20_mean"]
    if "amount_ratio_20" in df.columns:
        df["board_amount_ratio20_mean"] = board_group["amount_ratio_20"].transform("mean")
    if "up_day_ratio_20" in df.columns:
        df["board_up_ratio20_mean"] = board_group["up_day_ratio_20"].transform("mean")
    df["board_hotspot"] = (
        df.get("board_ret20_mean", 0.0).fillna(0.0)
        + 0.20 * df.get("board_amount_ratio20_mean", 0.0).fillna(0.0)
        + 0.10 * df.get("board_up_ratio20_mean", 0.0).fillna(0.0)
    )

    rank_columns = [
        "涨跌幅",
        "换手率",
        "振幅",
        "intraday_strength",
        "range_pct",
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_20",
        "ret_40",
        "ret_60",
        "volatility_10",
        "volatility_20",
        "volatility_40",
        "volume_ratio_5",
        "volume_ratio_10",
        "volume_ratio_20",
        "amount_ratio_5",
        "amount_ratio_10",
        "amount_ratio_20",
        "turnover_ratio_5",
        "turnover_ratio_20",
        "turnover_mean_20",
        "price_pos_20",
        "drawdown_20",
        "stable_mom_20",
        "relative_strength_5",
        "relative_strength_10",
        "relative_strength_20",
        "board_relative_strength_20",
        "board_hotspot",
        "limit_up_count_3",
        "limit_up_count_5",
        "big_up_count_5",
        "big_down_count_5",
        "up_day_ratio_5",
        "up_day_ratio_20",
        "ret_accel_5_20",
        "turnover_accel_5_20",
        "volume_price_confirm",
        "amount_price_confirm",
        "breakout_20",
        "breakout_count_20",
        "near_high_20",
        "overheat_20",
    ]

    for col in rank_columns:
        if col not in df.columns:
            continue
        grouped = df.groupby("日期")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        df[f"{col}_rank"] = grouped.rank(pct=True)
        df[f"{col}_z"] = (df[col] - mean) / (std + 1e-12)
        df[f"{col}_excess"] = df[col] - mean

    hotspot_parts = [
        ("ret_20_rank", 0.22),
        ("ret_10_rank", 0.16),
        ("relative_strength_20_rank", 0.18),
        ("amount_ratio_20_rank", 0.10),
        ("volume_ratio_20_rank", 0.08),
        ("turnover_accel_5_20_rank", 0.08),
        ("big_up_count_5_rank", 0.07),
        ("breakout_20_rank", 0.06),
        ("board_hotspot_rank", 0.05),
    ]
    df["hotspot_strength"] = 0.0
    for col, weight in hotspot_parts:
        if col in df.columns:
            df["hotspot_strength"] += df[col].fillna(0.5) * weight
    df["hotspot_strength_rank"] = day_group["hotspot_strength"].rank(pct=True)
    df["hotspot_strength_z"] = (
        df["hotspot_strength"] - day_group["hotspot_strength"].transform("mean")
    ) / (day_group["hotspot_strength"].transform("std") + 1e-12)

    if "amount_ratio_20" in df.columns:
        amount20 = df["amount_ratio_20"].fillna(1.0)
        amount_moderate = (
            1.0
            - ((amount20 - 1.65).clip(lower=0.0) / 1.65).clip(0.0, 1.0)
        ) * (
            1.0
            - ((0.80 - amount20).clip(lower=0.0) / 0.80).clip(0.0, 1.0)
        )
        df["amount_moderate_20"] = amount_moderate
        df["amount_moderate_20_rank"] = day_group["amount_moderate_20"].rank(pct=True)
    if "overheat_20" in df.columns:
        df["no_overheat_20"] = 1.0 - df["overheat_20"].fillna(0.0)
        df["no_overheat_20_rank"] = day_group["no_overheat_20"].rank(pct=True)

    df["market_ret1_mean"] = day_group["ret_1"].transform("mean")
    df["market_ret1_std"] = day_group["ret_1"].transform("std")
    df["market_up_ratio"] = day_group["ret_1"].transform(lambda s: (s > 0).mean())
    df["market_vol20_mean"] = day_group["volatility_20"].transform("mean")

    return df.replace([np.inf, -np.inf], np.nan)


def build_tabular_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """构造表格模型特征：个股时序特征 + 当日横截面排名 + 市场状态。"""
    df = _normalize_raw_data(raw_df)
    df = _merge_optional_offline_factors(df)
    pieces = [_add_single_stock_features(group) for _, group in df.groupby("股票代码", sort=False)]
    featured = pd.concat(pieces, ignore_index=True)
    return _add_cross_sectional_features(featured)


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "股票代码",
        "日期",
        "future_return",
        "target_rank",
        "target_lgb_rank",
        "target_top5",
        "target_gain",
        "sample_weight",
        # Gate-only diagnostics. They are useful for second-leg/micro-leader
        # audits, but the unvalidated 20-day breadth ranks drifted A1 trunk
        # ordering when admitted into the first-layer learner.
        "up_day_ratio_20_excess",
        "up_day_ratio_20_rank",
        "up_day_ratio_20_z",
    }
    return [
        col
        for col in df.columns
        if col not in blocked and pd.api.types.is_numeric_dtype(df[col])
    ]


def _prepare_training_table(featured: pd.DataFrame, min_history_days: int) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    labeled = featured.dropna(subset=["future_return"]).copy()
    labeled = labeled[labeled["history_days"] >= min_history_days].copy()

    feature_columns = _select_feature_columns(labeled)
    medians = labeled[feature_columns].median(numeric_only=True)
    labeled[feature_columns] = labeled[feature_columns].fillna(medians).fillna(0.0)

    labeled["target_rank"] = labeled.groupby("日期")["future_return"].rank(pct=True)
    descending_rank = labeled.groupby("日期")["future_return"].rank(
        method="first", ascending=False
    )
    # LGBMRanker 使用“每天横截面排名等级”作为相关性标签，而不是单纯收益回归。
    # 0~15 的离散等级兼顾 lambdarank 稳定性和 Top3/Top5 的排序压力。
    labeled["target_lgb_rank"] = np.floor(labeled["target_rank"] * 15).clip(0, 15).astype(int)
    labeled.loc[descending_rank.le(3), "target_lgb_rank"] = 15
    labeled.loc[(descending_rank.gt(3)) & (descending_rank.le(5)), "target_lgb_rank"] = 14
    labeled["target_top5"] = descending_rank.le(5)
    labeled["target_gain"] = np.select(
        [
            descending_rank.le(1),
            descending_rank.le(5),
            labeled["target_rank"] >= 0.90,
            labeled["target_rank"] >= 0.75,
            labeled["future_return"] > 0,
        ],
        [5, 4, 3, 2, 1],
        default=0,
    ).astype(int)
    labeled["sample_weight"] = (
        1.0
        + 3.0 * labeled["target_top5"].astype(float)
        + 1.2 * (descending_rank <= 3).astype(float)
        + 0.8 * (labeled["target_rank"] > 0.75).astype(float)
        + 0.2 * (labeled["future_return"] > 0).astype(float)
    )
    return labeled, feature_columns, medians


def _model_specs(seed: int):
    return [
        (
            "hgb_return",
            "future_return",
            HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.045,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                l2_regularization=0.05,
                random_state=seed,
                early_stopping=False,
            ),
        ),
        (
            "hgb_rank",
            "target_rank",
            HistGradientBoostingRegressor(
                max_iter=260,
                learning_rate=0.04,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                l2_regularization=0.05,
                random_state=seed + 1,
                early_stopping=False,
            ),
        ),
        (
            "extra_rank",
            "target_rank",
            ExtraTreesRegressor(
                n_estimators=180,
                min_samples_leaf=12,
                max_features=0.45,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ]


def _lightgbm_model_specs(seed: int):
    if not HAS_LIGHTGBM:
        return []
    return [
        (
            "lgb_rank",
            "target_lgb_rank",
            LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=320,
                learning_rate=0.035,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.05,
                reg_lambda=0.10,
                random_state=seed + 7,
                n_jobs=-1,
                verbose=-1,
            ),
            "ranker",
        ),
        (
            "lgb_return",
            "future_return",
            LGBMRegressor(
                objective="regression",
                n_estimators=260,
                learning_rate=0.035,
                num_leaves=31,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.05,
                reg_lambda=0.10,
                random_state=seed + 11,
                n_jobs=-1,
                verbose=-1,
            ),
            "regressor",
        ),
    ]


def _date_groups(table: pd.DataFrame) -> List[int]:
    """LightGBM Ranker 需要每个交易日对应一个 group size。"""
    return table.groupby("日期", sort=False).size().astype(int).tolist()


def fit_component_models(train_table: pd.DataFrame, feature_columns: List[str], seed: int) -> Dict[str, object]:
    """训练所有表格组件模型；LightGBM 不可用时自动回退到 sklearn 组件。"""
    models = {}
    for name, target, model in _model_specs(seed):
        model.fit(
            train_table[feature_columns],
            train_table[target],
            sample_weight=train_table["sample_weight"],
        )
        models[name] = model

    for name, target, model, model_type in _lightgbm_model_specs(seed):
        if model_type == "ranker":
            sorted_table = train_table.sort_values(["日期", "股票代码"]).reset_index(drop=True)
            model.fit(
                sorted_table[feature_columns],
                sorted_table[target],
                group=_date_groups(sorted_table),
                sample_weight=sorted_table["sample_weight"],
                eval_at=[1, 3, 5],
            )
        else:
            model.fit(
                train_table[feature_columns],
                train_table[target],
                sample_weight=train_table["sample_weight"],
            )
        models[name] = model
    return models


def _add_blended_score(df: pd.DataFrame, component_weights: Dict[str, float], heuristic_weights: Dict[str, float]) -> pd.DataFrame:
    scored = df.copy()
    scored["heuristic"] = 0.0
    for col, weight in heuristic_weights.items():
        if col in scored.columns:
            scored["heuristic"] += scored[col].fillna(0.5) * weight

    scored["tabular_score"] = 0.0
    for col, weight in component_weights.items():
        if col in scored.columns:
            scored["tabular_score"] += _pct_rank(scored[col]) * weight

    controlled_parts = [
        ("relative_strength_20_rank", 0.22),
        ("ret_20_rank", 0.23),
        ("ret_10_rank", 0.22),
        ("ret_5_rank", 0.12),
        ("hotspot_strength_rank", 0.10),
        ("amount_moderate_20_rank", 0.07),
        ("no_overheat_20_rank", 0.04),
    ]
    scored["controlled_hotspot"] = 0.0
    for col, weight in controlled_parts:
        if col in scored.columns:
            scored["controlled_hotspot"] += scored[col].fillna(0.5) * weight

    return scored


def _selector_gate_feature_row(
    snapshot: Dict[str, float] | None,
    score_profile: Dict[str, float] | None,
    hgb_overlap: float,
) -> Dict[str, float]:
    snapshot = snapshot or {}
    score_profile = score_profile or {}
    row = {
        "market_ret_5": float(snapshot.get("market_ret_5", 0.0) or 0.0),
        "market_ret_20": float(snapshot.get("market_ret_20", 0.0) or 0.0),
        "up_ratio_5": float(snapshot.get("up_ratio_5", 0.5) or 0.5),
        "up_ratio_20": float(snapshot.get("up_ratio_20", 0.5) or 0.5),
        "market_drawdown_20": float(snapshot.get("market_drawdown_20", 0.0) or 0.0),
        "ret5_p95_minus_median": float(snapshot.get("ret5_p95_minus_median", 0.0) or 0.0),
        "ret20_p95_minus_median": float(snapshot.get("ret20_p95_minus_median", 0.0) or 0.0),
        "amount_ratio20_p90_p50": float(snapshot.get("amount_ratio20_p90_p50", 1.0) or 1.0),
        "top2_vs_top10_gap": float(score_profile.get("top2_vs_top10_gap", 0.0) or 0.0),
        "selector_hgb_overlap": float(hgb_overlap or 0.0),
    }
    for key, value in row.items():
        if not np.isfinite(value):
            row[key] = 0.0
    return row


def _neutral_selector_gate_weights() -> Dict[str, float]:
    return {
        "selector_consensus": 0.50,
        "selector_hot_rotation": 0.25,
        "selector_defensive": 0.25,
    }


def _predict_selector_gate_weights(
    selector_gate: Dict[str, object] | None,
    snapshot: Dict[str, float] | None,
    score_profile: Dict[str, float] | None,
    hgb_overlap: float,
) -> Dict[str, float] | None:
    if not selector_gate or selector_gate.get("model") is None:
        return None

    feature_columns = selector_gate.get("feature_columns", SELECTOR_GATE_FEATURE_COLUMNS)
    medians = selector_gate.get("feature_medians", {})
    feature_row = _selector_gate_feature_row(snapshot, score_profile, hgb_overlap)
    x = pd.DataFrame([{col: feature_row.get(col, medians.get(col, 0.0)) for col in feature_columns}])
    x = x.replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).fillna(0.0)

    model = selector_gate["model"]
    if not hasattr(model, "predict_proba"):
        return None

    probs = model.predict_proba(x)[0]
    weights = {branch: 0.0 for branch in SELECTOR_GATE_BRANCHES}
    for cls, prob in zip(model.classes_, probs):
        branch = str(cls)
        if branch in weights:
            weights[branch] = float(prob)

    total = sum(weights.values())
    if total <= 1e-12:
        return _neutral_selector_gate_weights()
    return {branch: weight / total for branch, weight in weights.items()}


def _fit_selector_gate_model(
    val_pred: pd.DataFrame,
    val_table: pd.DataFrame,
    raw_df: pd.DataFrame,
    seed: int,
) -> Dict[str, object] | None:
    """
    Learn which selector branch works best under each validation-day regime.

    The target is the branch with the highest daily Top3 future return. Features
    only use information visible at the prediction date plus current score shape.
    """
    try:
        from positioning import compute_market_snapshot, compute_score_profile
    except Exception as exc:
        print(f"学习型 selector gate 跳过：无法加载市场状态工具: {exc}")
        return None

    rows = []
    targets = []
    val_future = val_table[["股票代码", "日期", "future_return"]].copy()
    val_future["股票代码"] = val_future["股票代码"].astype(str).str.zfill(6)

    for date, day_pred in val_pred.groupby("日期", sort=True):
        day = day_pred.merge(val_future, on=["股票代码", "日期"], how="left")
        day = day.dropna(subset=["future_return"]).copy()
        if len(day) < 10:
            continue

        snapshot_seed_col = "controlled_hotspot" if "controlled_hotspot" in day.columns else "tabular_score"
        snapshot_seed = day.sort_values(snapshot_seed_col, ascending=False)["股票代码"].head(5).tolist()
        snapshot = compute_market_snapshot(raw_df, date, snapshot_seed)
        branch_scored = add_strategy_selector_score(day, snapshot)
        profile_df = branch_scored.sort_values("selector_consensus", ascending=False).rename(
            columns={"selector_consensus": "final_score"}
        )
        score_profile = compute_score_profile(profile_df, "final_score")
        hgb_overlap = float(branch_scored["selector_hgb_overlap"].iloc[0]) if "selector_hgb_overlap" in branch_scored else 0.0

        branch_returns = {}
        for branch in SELECTOR_GATE_BRANCHES:
            if branch not in branch_scored.columns:
                continue
            selected = branch_scored.nlargest(3, branch)
            branch_returns[branch] = float(selected["future_return"].mean())
        if len(branch_returns) < 2:
            continue

        rows.append(_selector_gate_feature_row(snapshot, score_profile, hgb_overlap))
        targets.append(max(branch_returns, key=branch_returns.get))

    if len(rows) < 12 or len(set(targets)) < 2:
        print(
            "学习型 selector gate 样本不足，保留规则回退: "
            f"samples={len(rows)}, classes={sorted(set(targets))}"
        )
        return None

    train_x = pd.DataFrame(rows, columns=SELECTOR_GATE_FEATURE_COLUMNS)
    medians = train_x.median(numeric_only=True).to_dict()
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).fillna(0.0)
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_x, targets)
    print(
        "学习型 selector gate 已训练: "
        f"samples={len(train_x)}, classes={dict(pd.Series(targets).value_counts())}"
    )
    return {
        "model": model,
        "feature_columns": list(SELECTOR_GATE_FEATURE_COLUMNS),
        "feature_medians": medians,
        "training_samples": len(train_x),
        "target_counts": dict(pd.Series(targets).value_counts()),
    }


def add_strategy_selector_score(
    df: pd.DataFrame,
    snapshot: Dict[str, float] | None = None,
    selector_gate: Dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    第二层离线日线 selector。

    它只使用当期一层模型输出、离线量价热点代理和预测日前市场快照；
    目标不是追求单轮极限，而是降低 hgb_rank 在风格切换期单独翻车的概率。
    """
    scored = df.copy()
    snapshot = snapshot or {}

    for col in SELECTOR_COMPONENT_COLUMNS:
        if col in scored.columns:
            scored[f"{col}_selector_rank"] = _pct_rank(scored[col])

    def rank_col(col: str, default: float = 0.5) -> pd.Series:
        rank_name = f"{col}_selector_rank"
        if rank_name in scored.columns:
            return scored[rank_name].fillna(default)
        return pd.Series(default, index=scored.index, dtype=float)

    hgb = rank_col("hgb_rank")
    lgb = rank_col("lgb_rank")
    extra = rank_col("extra_rank")
    hgb_ret = rank_col("hgb_return")
    lgb_ret = rank_col("lgb_return")
    heuristic = rank_col("heuristic")
    hotspot = rank_col("hotspot_strength_rank")
    controlled_hotspot = rank_col("controlled_hotspot")
    relative = rank_col("relative_strength_20_rank")
    board_hot = rank_col("board_hotspot_rank")

    scored["selector_consensus"] = (
        0.28 * hgb
        + 0.20 * lgb
        + 0.20 * extra
        + 0.12 * hgb_ret
        + 0.08 * lgb_ret
        + 0.12 * heuristic
    )
    scored["selector_hot_rotation"] = (
        0.36 * controlled_hotspot
        + 0.18 * hotspot
        + 0.16 * relative
        + 0.12 * heuristic
        + 0.08 * lgb
        + 0.06 * extra
        + 0.04 * board_hot
    )
    scored["selector_defensive"] = (
        0.34 * hgb
        + 0.24 * extra
        + 0.18 * lgb
        + 0.14 * hgb_ret
        + 0.10 * heuristic
    )

    market_ret_20 = float(snapshot.get("market_ret_20", 0.0) or 0.0)
    market_ret_5 = float(snapshot.get("market_ret_5", 0.0) or 0.0)
    up_ratio_20 = float(snapshot.get("up_ratio_20", 0.5) or 0.5)
    up_ratio_5 = float(snapshot.get("up_ratio_5", 0.5) or 0.5)
    selected_ret_20 = float(snapshot.get("selected_ret_20_mean", 0.0) or 0.0)
    drawdown_20 = float(snapshot.get("market_drawdown_20", 0.0) or 0.0)

    weak_breadth = up_ratio_20 < 0.38 or up_ratio_5 < 0.40
    market_under_pressure = market_ret_20 < -0.025 or market_ret_5 < -0.012 or drawdown_20 < -0.055
    hot_candidates = selected_ret_20 > 0.18

    top_sets = {}
    for col in ("hgb_rank", "lgb_rank", "extra_rank", "hotspot_strength_rank", "relative_strength_20_rank"):
        if col in scored.columns:
            top_sets[col] = set(scored.nlargest(10, col)["股票代码"].astype(str))

    hgb_overlap = 0.0
    if "hgb_rank" in top_sets and len(top_sets) > 1:
        peers = set().union(*(stocks for col, stocks in top_sets.items() if col != "hgb_rank"))
        hgb_top = set(scored.nlargest(5, "hgb_rank")["股票代码"].astype(str))
        hgb_overlap = len(hgb_top & peers) / max(len(hgb_top), 1)

    controlled_theme_squeeze = (
        selected_ret_20 > 0.50
        and up_ratio_20 < 0.38
        and -0.020 < market_ret_20 < 0.015
    )
    rotation_regime = (
        selected_ret_20 > 0.30
        and up_ratio_5 < 0.40
        and market_ret_20 > -0.025
    )
    defensive_regime = market_under_pressure and weak_breadth and not hot_candidates

    profile_df = scored.sort_values("selector_consensus", ascending=False).rename(
        columns={"selector_consensus": "final_score"}
    )
    try:
        from positioning import compute_score_profile

        score_profile = compute_score_profile(profile_df, "final_score")
    except Exception:
        score_profile = {}

    learned_weights = _predict_selector_gate_weights(
        selector_gate,
        snapshot,
        score_profile,
        hgb_overlap,
    )

    if learned_weights is not None:
        scored["selector_score"] = (
            learned_weights["selector_consensus"] * scored["selector_consensus"]
            + learned_weights["selector_hot_rotation"] * scored["selector_hot_rotation"]
            + learned_weights["selector_defensive"] * scored["selector_defensive"]
        )
        regime = "learned_regime_gate"
        scored["selector_gate_consensus_weight"] = learned_weights["selector_consensus"]
        scored["selector_gate_hot_rotation_weight"] = learned_weights["selector_hot_rotation"]
        scored["selector_gate_defensive_weight"] = learned_weights["selector_defensive"]
    elif controlled_theme_squeeze:
        scored["selector_score"] = controlled_hotspot
        regime = "controlled_theme_squeeze"
    elif rotation_regime:
        scored["selector_score"] = (
            0.72 * scored["selector_hot_rotation"]
            + 0.18 * controlled_hotspot
            + 0.10 * hgb
        )
        regime = "rotation_hotspot"
    elif defensive_regime:
        scored["selector_score"] = (
            0.70 * hgb
            + 0.18 * scored["selector_defensive"]
            + 0.12 * scored["selector_consensus"]
        )
        regime = "weak_defensive_hgb_guard"
    else:
        scored["selector_score"] = hgb
        regime = "hgb_anchor"

    scored["selector_regime"] = regime
    scored["selector_hgb_overlap"] = hgb_overlap
    if "selector_gate_consensus_weight" not in scored.columns:
        scored["selector_gate_consensus_weight"] = np.nan
        scored["selector_gate_hot_rotation_weight"] = np.nan
        scored["selector_gate_defensive_weight"] = np.nan
    return scored


def _daily_topk_score(df: pd.DataFrame, score_col: str, k: int = 5) -> Dict[str, float]:
    scores = []
    hit_positive = []
    for _, group in df.groupby("日期"):
        if len(group) < k:
            continue
        selected = group.nlargest(k, score_col)
        scores.append(selected["future_return"].mean())
        hit_positive.append((selected["future_return"] > 0).mean())

    return {
        "daily_top5_mean_return": float(np.mean(scores)) if scores else 0.0,
        "daily_top5_positive_ratio": float(np.mean(hit_positive)) if hit_positive else 0.0,
        "validation_days": int(len(scores)),
    }


def train_tabular_ranker(
    raw_df: pd.DataFrame,
    output_dir: str,
    config: Dict,
    *,
    model_file_name: str = MODEL_FILE_NAME,
    factor_policy: str | None = None,
) -> Dict[str, float]:
    print("\n=== 训练表格排序增强模型 ===")
    if factor_policy:
        print(f"离线因子策略: {factor_policy}")
    seed = int(config.get("seed", RANDOM_SEED))
    min_history_days = int(config.get("tabular_min_history_days", 60))
    validation_days = int(config.get("tabular_val_days", 40))
    embargo_days = int(config.get("tabular_embargo_days", 5))
    component_weights = config.get("tabular_component_weights", DEFAULT_COMPONENT_WEIGHTS)
    heuristic_weights = config.get("tabular_heuristic_weights", DEFAULT_HEURISTIC_WEIGHTS)

    with _temporary_factor_policy(factor_policy):
        featured = build_tabular_features(raw_df)
    labeled, feature_columns, medians = _prepare_training_table(featured, min_history_days)

    dates = np.array(sorted(labeled["日期"].unique()))
    if len(dates) <= validation_days + embargo_days + 30:
        validation_days = max(10, min(20, len(dates) // 5))
        embargo_days = min(embargo_days, 3)

    val_dates = set(dates[-validation_days:])
    train_cut = max(1, len(dates) - validation_days - embargo_days)
    train_dates = set(dates[:train_cut])
    train_table = labeled[labeled["日期"].isin(train_dates)].copy()
    val_table = labeled[labeled["日期"].isin(val_dates)].copy()

    print(f"表格特征数: {len(feature_columns)}")
    print(f"表格训练样本: {len(train_table)}, 验证样本: {len(val_table)}")
    print(f"验证日期数: {validation_days}, embargo交易日: {embargo_days}")

    val_pred = val_table[["股票代码", "日期", "future_return"]].copy()
    val_feature_part = val_table[feature_columns]
    print(f"LightGBM排序分支: {'启用' if HAS_LIGHTGBM else '未安装，跳过'}")
    val_models = fit_component_models(train_table, feature_columns, seed)
    for name, model in val_models.items():
        print(f"验证预测组件: {name}")
        val_pred[name] = model.predict(val_feature_part)

    for col in heuristic_weights:
        if col in val_table.columns:
            val_pred[col] = val_table[col].values

    val_pred = _add_blended_score(val_pred, component_weights, heuristic_weights)
    selector_gate = None
    if config.get("selector_gate_enabled", True):
        selector_gate = _fit_selector_gate_model(val_pred, val_table, raw_df, seed)

    val_metrics = _daily_topk_score(val_pred, "tabular_score", k=5)
    print(
        "表格模型验证: "
        f"Top5均值={val_metrics['daily_top5_mean_return']:.6f}, "
        f"正收益占比={val_metrics['daily_top5_positive_ratio']:.4f}"
    )

    final_models = {}
    print("训练最终表格组件模型")
    final_models = fit_component_models(labeled, feature_columns, seed)

    bundle = {
        "models": final_models,
        "feature_columns": feature_columns,
        "feature_medians": medians,
        "component_weights": component_weights,
        "heuristic_weights": heuristic_weights,
        "min_history_days": min_history_days,
        "validation_metrics": val_metrics,
        "selector_gate": selector_gate,
    }
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_file_name)
    joblib.dump(bundle, model_path)
    print(f"表格排序模型已保存: {model_path}")
    return val_metrics


def predict_tabular_ranker(
    raw_df: pd.DataFrame,
    output_dir: str,
    *,
    model_file_name: str = MODEL_FILE_NAME,
    factor_policy: str | None = None,
) -> pd.DataFrame:
    model_path = os.path.join(output_dir, model_file_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到表格排序模型: {model_path}")

    bundle = joblib.load(model_path)
    with _temporary_factor_policy(factor_policy):
        featured = build_tabular_features(raw_df)
    latest_date = featured["日期"].max()
    latest = featured[featured["日期"] == latest_date].copy()

    feature_columns = bundle["feature_columns"]
    medians = bundle["feature_medians"]
    for col in feature_columns:
        if col not in latest.columns:
            latest[col] = 0.0
    latest[feature_columns] = latest[feature_columns].replace([np.inf, -np.inf], np.nan)
    latest[feature_columns] = latest[feature_columns].fillna(medians).fillna(0.0)

    pred = latest[["股票代码", "日期"]].copy()
    for name, model in bundle["models"].items():
        pred[name] = model.predict(latest[feature_columns])

    for col in bundle["heuristic_weights"]:
        if col in latest.columns:
            pred[col] = latest[col].values
    pred = copy_selector_auxiliary_columns(pred, latest)

    pred = _add_blended_score(pred, bundle["component_weights"], bundle["heuristic_weights"])
    pred = add_strategy_selector_score(pred, selector_gate=bundle.get("selector_gate"))
    pred.attrs["selector_gate"] = bundle.get("selector_gate")
    return pred.sort_values("tabular_score", ascending=False).reset_index(drop=True)
