import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_COL = "\u80a1\u7968\u4ee3\u7801"
DATE_COL = "\u65e5\u671f"
OPEN_COL = "\u5f00\u76d8"

COOLDOWN_COLS = [
    "ts_event_near_limit_cooldown_3_rank",
    "ts_event_near_limit_cooldown_5_rank",
    "ts_event_limit_up_cooldown_5_rank",
]


def parse_codes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip().zfill(6) for part in str(value).replace(";", ",").split(",") if part.strip()]


def parse_weights(value: object, count: int) -> list[float]:
    if pd.isna(value):
        return [1.0 / count] * count if count else []
    weights = []
    for part in str(value).replace(",", "/").split("/"):
        part = part.strip()
        if not part:
            continue
        try:
            weights.append(float(part))
        except ValueError:
            pass
    if len(weights) < count:
        weights.extend([0.0] * (count - len(weights)))
    return weights[:count]


def load_returns(raw_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, dtype={CODE_COL: str})
    raw[CODE_COL] = raw[CODE_COL].astype(str).str.zfill(6)
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
    raw[OPEN_COL] = pd.to_numeric(raw[OPEN_COL], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    first = raw[raw[DATE_COL].eq(start)][[CODE_COL, OPEN_COL]].rename(columns={OPEN_COL: "first_open"})
    last = raw[raw[DATE_COL].eq(end)][[CODE_COL, OPEN_COL]].rename(columns={OPEN_COL: "last_open"})
    out = first.merge(last, on=CODE_COL, how="inner")
    out["future_return"] = (out["last_open"] - out["first_open"]) / (out["first_open"] + 1e-12)
    out = out.sort_values("future_return", ascending=False).reset_index(drop=True)
    out["single_stock_rank"] = np.arange(1, len(out) + 1)
    return out[[CODE_COL, "future_return", "single_stock_rank"]]


def score_portfolio(codes: list[str], weights: list[float], returns: pd.DataFrame) -> tuple[float, str]:
    selected = pd.DataFrame({CODE_COL: codes, "weight": weights[: len(codes)]})
    scored = selected.merge(returns, on=CODE_COL, how="left")
    scored["future_return"] = scored["future_return"].fillna(0.0)
    score = float((scored["future_return"] * scored["weight"]).sum())
    ranks = ",".join(
        f"{row[CODE_COL]}:{int(row['single_stock_rank']) if pd.notna(row['single_stock_rank']) else 999}"
        for _, row in scored.iterrows()
    )
    return score, ranks


def load_event_table(path: Path) -> pd.DataFrame:
    event = pd.read_csv(path, dtype={CODE_COL: str})
    event[CODE_COL] = event[CODE_COL].astype(str).str.zfill(6)
    event[DATE_COL] = pd.to_datetime(event[DATE_COL], errors="coerce")
    for col in COOLDOWN_COLS:
        if col not in event.columns:
            event[col] = np.nan
        event[col] = pd.to_numeric(event[col], errors="coerce")
    event["event_cooldown_guard"] = (
        0.45 * event["ts_event_near_limit_cooldown_3_rank"].fillna(0.5)
        + 0.35 * event["ts_event_near_limit_cooldown_5_rank"].fillna(0.5)
        + 0.20 * event["ts_event_limit_up_cooldown_5_rank"].fillna(0.5)
    )
    return event[[CODE_COL, DATE_COL, "event_cooldown_guard"] + COOLDOWN_COLS]


def event_scores_for_cutoff(event: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(cutoff)
    frame = event[event[DATE_COL].eq(cutoff_ts)].copy()
    return frame.set_index(CODE_COL)


def choose_top5(row: pd.Series) -> list[str]:
    chosen_lane = str(row.get("chosen_lane", ""))
    if chosen_lane == "index_weight" and "index_top5" in row:
        return parse_codes(row["index_top5"])
    if chosen_lane == "no_factors" and "nofactor_top5" in row:
        return parse_codes(row["nofactor_top5"])
    return parse_codes(row.get("selected", ""))


def guard_score(event_by_code: pd.DataFrame, code: str) -> float:
    if code not in event_by_code.index:
        return 0.5
    value = event_by_code.loc[code, "event_cooldown_guard"]
    return float(value) if pd.notna(value) else 0.5


def apply_conservative_guard(row: pd.Series, event_by_code: pd.DataFrame) -> tuple[list[str], str]:
    selected = parse_codes(row.get("selected", ""))
    top5 = choose_top5(row)
    if len(selected) != 2 or len(top5) < 3:
        return selected, "not_top2_or_no_candidate"

    second_quality = float(row.get("second_quality", 0.5) or 0.5)
    third_quality = float(row.get("third_quality", 0.5) or 0.5)
    second = selected[1]
    third = top5[2]
    second_guard = guard_score(event_by_code, second)
    third_guard = guard_score(event_by_code, third)

    rank3_model_quality_better = third_quality > second_quality + 0.06
    second_event_overheated = second_guard < 0.12
    third_event_cleaner = third_guard > second_guard + 0.32
    if rank3_model_quality_better and second_event_overheated and third_event_cleaner:
        return [selected[0], third], "swap_rank2_to_rank3_event_guard"
    return selected, "keep_conservative"


def apply_cooldown_only_guard(row: pd.Series, event_by_code: pd.DataFrame) -> tuple[list[str], str]:
    selected = parse_codes(row.get("selected", ""))
    top5 = choose_top5(row)
    if len(selected) != 2 or len(top5) < 3:
        return selected, "not_top2_or_no_candidate"

    second = selected[1]
    second_guard = guard_score(event_by_code, second)
    candidates = [code for code in top5[2:5] if code not in selected]
    if not candidates:
        return selected, "keep_no_alt"
    best_alt = max(candidates, key=lambda code: guard_score(event_by_code, code))
    best_guard = guard_score(event_by_code, best_alt)
    if second_guard < 0.16 and best_guard > second_guard + 0.30:
        return [selected[0], best_alt], "swap_cooldown_only"
    return selected, "keep_cooldown_only"


def audit_file(replay_path: Path, event: pd.DataFrame, output: Path) -> pd.DataFrame:
    replay = pd.read_csv(replay_path)
    rows = []
    for _, row in replay.iterrows():
        cutoff = str(row["cutoff"])
        returns = load_returns(ROOT / "data" / "stock_data.csv", str(row["test_start"]), str(row["test_end"]))
        event_by_code = event_scores_for_cutoff(event, cutoff)

        selected = parse_codes(row.get("selected", ""))
        weights = parse_weights(row.get("weights", ""), len(selected))
        baseline_score, baseline_ranks = score_portfolio(selected, weights, returns)

        conservative_codes, conservative_action = apply_conservative_guard(row, event_by_code)
        conservative_score, conservative_ranks = score_portfolio(conservative_codes, weights, returns)

        cooldown_codes, cooldown_action = apply_cooldown_only_guard(row, event_by_code)
        cooldown_score, cooldown_ranks = score_portfolio(cooldown_codes, weights, returns)

        rows.append(
            {
                "window": row["window"],
                "cutoff": cutoff,
                "chosen_lane": row.get("chosen_lane", ""),
                "market_regime_v2": row.get("market_regime_v2", ""),
                "baseline_selected": ",".join(selected),
                "baseline_score": baseline_score,
                "baseline_ranks": baseline_ranks,
                "conservative_selected": ",".join(conservative_codes),
                "conservative_score": conservative_score,
                "conservative_delta": conservative_score - baseline_score,
                "conservative_action": conservative_action,
                "conservative_ranks": conservative_ranks,
                "cooldown_only_selected": ",".join(cooldown_codes),
                "cooldown_only_score": cooldown_score,
                "cooldown_only_delta": cooldown_score - baseline_score,
                "cooldown_only_action": cooldown_action,
                "cooldown_only_ranks": cooldown_ranks,
                "top5": ",".join(choose_top5(row)),
            }
        )

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def print_summary(result: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    for prefix in ["conservative", "cooldown_only"]:
        delta_col = f"{prefix}_delta"
        action_col = f"{prefix}_action"
        print(
            f"{prefix}: windows={len(result)} "
            f"mean_delta={result[delta_col].mean():.6f} "
            f"sum_delta={result[delta_col].sum():.6f} "
            f"changed={(result[action_col].str.startswith('swap')).sum()}"
        )
    display_cols = [
        "window",
        "baseline_selected",
        "baseline_score",
        "conservative_selected",
        "conservative_delta",
        "conservative_action",
        "cooldown_only_selected",
        "cooldown_only_delta",
        "cooldown_only_action",
    ]
    print(result[display_cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit event cooldown as a selector guard on replay summaries.")
    parser.add_argument("--replay", nargs="+", required=True)
    parser.add_argument("--event-file", default=str(ROOT / "data" / "offline_factors" / "event_factors.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "output"))
    args = parser.parse_args()

    event = load_event_table(Path(args.event_file))
    for replay_item in args.replay:
        replay_path = Path(replay_item)
        if not replay_path.is_absolute():
            replay_path = ROOT / replay_path
        output = Path(args.output_dir) / f"{replay_path.stem}_event_guard_audit.csv"
        result = audit_file(replay_path, event, output)
        print_summary(result, replay_path.name)
        print(f"output={output}")


if __name__ == "__main__":
    main()
