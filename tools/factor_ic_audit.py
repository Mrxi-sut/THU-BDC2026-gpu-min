import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_COL = "\u80a1\u7968\u4ee3\u7801"
DATE_COL = "\u65e5\u671f"
OPEN_COL = "\u5f00\u76d8"


def compute_future_return(raw: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    df = raw[[CODE_COL, DATE_COL, OPEN_COL]].copy()
    df[CODE_COL] = df[CODE_COL].astype(str).str.zfill(6)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[OPEN_COL] = pd.to_numeric(df[OPEN_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, OPEN_COL]).sort_values([CODE_COL, DATE_COL])
    grouped = df.groupby(CODE_COL, group_keys=False)
    df["open_t1"] = grouped[OPEN_COL].shift(-1)
    df[f"open_t{hold_days}"] = grouped[OPEN_COL].shift(-hold_days)
    df["future_return"] = (df[f"open_t{hold_days}"] - df["open_t1"]) / (df["open_t1"] + 1e-12)
    return df[[CODE_COL, DATE_COL, "future_return"]].dropna()


def load_factor(path: Path) -> pd.DataFrame:
    factor = pd.read_csv(path, dtype={CODE_COL: str})
    if CODE_COL not in factor.columns or DATE_COL not in factor.columns:
        raise ValueError(f"missing identity columns: {path}")
    factor[CODE_COL] = factor[CODE_COL].astype(str).str.zfill(6)
    factor[DATE_COL] = pd.to_datetime(factor[DATE_COL], errors="coerce")
    factor = factor.dropna(subset=[DATE_COL])
    blocked = {CODE_COL, DATE_COL, "available_date", "publish_datetime", "ts_code", "con_code"}
    value_cols = []
    for col in factor.columns:
        if col in blocked:
            continue
        factor[col] = pd.to_numeric(factor[col], errors="coerce")
        if factor[col].notna().any():
            value_cols.append(col)
    return factor[[CODE_COL, DATE_COL] + value_cols]


def _spearman_by_day(frame: pd.DataFrame, col: str, min_stocks: int) -> pd.Series:
    values = []
    for date, group in frame[[DATE_COL, col, "future_return"]].dropna().groupby(DATE_COL, sort=True):
        if len(group) < min_stocks or group[col].nunique() < 3:
            continue
        ic = group[col].corr(group["future_return"], method="spearman")
        if pd.notna(ic):
            values.append((date, float(ic)))
    if not values:
        return pd.Series(dtype=float)
    return pd.Series({date: ic for date, ic in values}).sort_index()


def _top_bottom_spread(frame: pd.DataFrame, col: str, min_stocks: int, quantile: float = 0.2) -> pd.Series:
    spreads = []
    for date, group in frame[[DATE_COL, col, "future_return"]].dropna().groupby(DATE_COL, sort=True):
        if len(group) < min_stocks or group[col].nunique() < 3:
            continue
        ranked = group.sort_values(col)
        n = max(1, int(len(ranked) * quantile))
        low = ranked.head(n)["future_return"].mean()
        high = ranked.tail(n)["future_return"].mean()
        spreads.append((date, float(high - low)))
    if not spreads:
        return pd.Series(dtype=float)
    return pd.Series({date: spread for date, spread in spreads}).sort_index()


def summarize_factor(frame: pd.DataFrame, family: str, col: str, min_stocks: int) -> dict:
    ic = _spearman_by_day(frame, col, min_stocks)
    spread = _top_bottom_spread(frame, col, min_stocks)
    if ic.empty:
        return {
            "family": family,
            "factor": col,
            "days": 0,
            "mean_ic": np.nan,
            "median_ic": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "positive_ic_ratio": np.nan,
            "mean_top_bottom_spread": np.nan,
            "directional_spread": np.nan,
            "coverage": float(frame[col].notna().mean()),
        }
    mean_ic = float(ic.mean())
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    mean_spread = float(spread.mean()) if not spread.empty else np.nan
    direction = 1.0 if mean_ic >= 0 else -1.0
    return {
        "family": family,
        "factor": col,
        "days": int(len(ic)),
        "mean_ic": mean_ic,
        "median_ic": float(ic.median()),
        "ic_std": ic_std,
        "icir": mean_ic / (ic_std + 1e-12),
        "positive_ic_ratio": float((ic > 0).mean()),
        "mean_top_bottom_spread": mean_spread,
        "directional_spread": direction * mean_spread if pd.notna(mean_spread) else np.nan,
        "coverage": float(frame[col].notna().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank IC audit for offline factor families.")
    parser.add_argument("--factor-dir", default=str(ROOT / "data" / "offline_factors"))
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--min-stocks", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--output", default=str(ROOT / "output" / "factor_ic_audit.csv"))
    args = parser.parse_args()

    raw = pd.read_csv(ROOT / "data" / "stock_data.csv", dtype={CODE_COL: str})
    future = compute_future_return(raw, args.hold_days)
    factor_dir = Path(args.factor_dir)

    rows = []
    for path in sorted(factor_dir.glob("*_factors.csv")):
        family = path.stem.replace("_factors", "")
        factor = load_factor(path)
        value_cols = [col for col in factor.columns if col not in {CODE_COL, DATE_COL}]
        if not value_cols:
            continue
        merged = factor.merge(future, on=[CODE_COL, DATE_COL], how="inner")
        for col in value_cols:
            rows.append(summarize_factor(merged, family, col, args.min_stocks))

    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_mean_ic"] = result["mean_ic"].abs()
        result["abs_icir"] = result["icir"].abs()
        result = result.sort_values(
            ["family", "abs_mean_ic", "directional_spread"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"rows={len(result)} output={out}")
    if not result.empty:
        display_cols = [
            "family",
            "factor",
            "days",
            "mean_ic",
            "icir",
            "positive_ic_ratio",
            "directional_spread",
            "coverage",
        ]
        print(result.sort_values("abs_mean_ic", ascending=False).head(args.top_n)[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
