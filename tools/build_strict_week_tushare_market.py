"""
Build a strict-week Tushare market table and endpoint return ranks.

The tool fetches daily bars by trade_date, filters to the local HS300 universe,
and writes two outputs:

1. Full daily market bars for the requested week.
2. Endpoint returns/ranks from the start open to T+1/T+5 open and close.

It never prints or writes the Tushare token.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
RAW_DAILY_DIR = ROOT / "data" / "raw" / "tushare" / "daily"
HS300_LIST_PATH = ROOT / "data" / "hs300_stock_list.csv"
FIELDS = "ts_code,trade_date,open,close,high,low,vol,amount,pct_chg"


def _to_ts_code(code: str) -> str:
    text = str(code).strip()
    if text.startswith("sh."):
        return text[3:] + ".SH"
    if text.startswith("sz."):
        return text[3:] + ".SZ"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 6:
        return ""
    suffix = ".SH" if digits.startswith(("5", "6", "9")) else ".SZ"
    return digits + suffix


def _load_universe() -> set[str]:
    hs300 = pd.read_csv(HS300_LIST_PATH, dtype=str)
    col = "code" if "code" in hs300.columns else hs300.columns[0]
    universe = {_to_ts_code(code) for code in hs300[col].dropna().tolist()}
    universe = {code for code in universe if code}
    if len(universe) < 250:
        raise RuntimeError(f"HS300 universe too small: {len(universe)}")
    return universe


def _ensure_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        try:
            token = (ts.get_token() or "").strip()
        except Exception:
            token = ""
    if not token:
        raise SystemExit("TUSHARE_TOKEN is not set and no SDK token was found.")
    return token


def _fetch_daily(pro, trade_date: str, universe: set[str], refresh: bool) -> pd.DataFrame:
    RAW_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DAILY_DIR / f"daily_{trade_date}.csv"
    if path.exists() and path.stat().st_size > 0 and not refresh:
        df = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
    else:
        df = pro.daily(trade_date=trade_date, fields=FIELDS)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    df["ts_code"] = df["ts_code"].astype(str)
    df["trade_date"] = df["trade_date"].astype(str)
    out = df[df["ts_code"].isin(universe)].copy()
    if len(out) != len(universe):
        missing = len(universe) - len(out)
        print(f"warn: {trade_date} HS300 rows={len(out)} missing={missing}", flush=True)
    return out


def _endpoint(df: pd.DataFrame, trade_date: str, prefix: str) -> pd.DataFrame:
    cols = ["ts_code", "open", "close"]
    out = df[df["trade_date"].eq(trade_date)][cols].copy()
    out = out.rename(columns={"open": f"{prefix}_open", "close": f"{prefix}_close"})
    return out


def _rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=False).astype("Int64")


def _build_returns(market: pd.DataFrame, start_date: str, t1_date: str, t5_date: str) -> pd.DataFrame:
    start = _endpoint(market, start_date, "start")
    t1 = _endpoint(market, t1_date, "t1")
    t5 = _endpoint(market, t5_date, "t5")
    out = start.merge(t1, on="ts_code", how="inner").merge(t5, on="ts_code", how="inner")
    out = out[out["start_open"] > 1e-12].copy()
    out["stock_id"] = out["ts_code"].str.extract(r"(\d{6})", expand=False).str.zfill(6)
    out["t1_open_return"] = (out["t1_open"] - out["start_open"]) / out["start_open"]
    out["t1_close_return"] = (out["t1_close"] - out["start_open"]) / out["start_open"]
    out["t5_open_return"] = (out["t5_open"] - out["start_open"]) / out["start_open"]
    out["t5_close_return"] = (out["t5_close"] - out["start_open"]) / out["start_open"]
    for col in ("t1_open_return", "t1_close_return", "t5_open_return", "t5_close_return"):
        out[col.replace("_return", "_rank")] = _rank_desc(out[col])
    ordered = [
        "stock_id",
        "ts_code",
        "start_open",
        "t1_open",
        "t1_close",
        "t5_open",
        "t5_close",
        "t1_open_return",
        "t1_open_rank",
        "t1_close_return",
        "t1_close_rank",
        "t5_open_return",
        "t5_open_rank",
        "t5_close_return",
        "t5_close_rank",
    ]
    return out[ordered].sort_values("t5_close_return", ascending=False).reset_index(drop=True)


def _score_portfolio(returns: pd.DataFrame, selected: list[str], weights: list[float], ret_col: str, rank_col: str) -> dict:
    selected_df = pd.DataFrame({"stock_id": [code.zfill(6) for code in selected], "weight": weights})
    scored = selected_df.merge(returns[["stock_id", ret_col, rank_col]], on="stock_id", how="left")
    portfolio_return = float((scored[ret_col] * scored["weight"]).sum())
    rank_vs_single = int((returns[ret_col] > portfolio_return).sum() + 1)
    return {
        "selected": ",".join(selected_df["stock_id"].tolist()),
        "weights": "/".join(f"{w:.2f}" for w in weights),
        "return_col": ret_col,
        "portfolio_return": portfolio_return,
        "rank_vs_single_stock": rank_vs_single,
        "selected_single_ranks": ",".join(scored[rank_col].astype("Int64").astype(str).tolist()),
        "selected_single_returns": ",".join(f"{x:.6f}" if np.isfinite(x) else "" for x in scored[ret_col].tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict-week Tushare daily market/rank files.")
    parser.add_argument("--dates", nargs="+", default=["20260622", "20260623", "20260624", "20260625", "20260626"])
    parser.add_argument("--start-date", default="20260622")
    parser.add_argument("--t1-date", default="20260623")
    parser.add_argument("--t5-date", default="20260626")
    parser.add_argument("--output-prefix", default="output/strict_week_20260622_20260626")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    universe = _load_universe()
    token = _ensure_token()
    ts.set_token(token)
    pro = ts.pro_api()

    frames = [_fetch_daily(pro, date, universe, args.refresh) for date in args.dates]
    market = pd.concat(frames, ignore_index=True, sort=False)
    market = market.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    returns = _build_returns(market, args.start_date, args.t1_date, args.t5_date)

    out_prefix = ROOT / args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    market_path = out_prefix.with_name(out_prefix.name + "_full_market.csv")
    returns_path = out_prefix.with_name(out_prefix.name + "_returns.csv")
    score_path = out_prefix.with_name(out_prefix.name + "_selected_scores.csv")
    top20_path = out_prefix.with_name(out_prefix.name + "_t5_close_top20.csv")

    market.to_csv(market_path, index=False, encoding="utf-8-sig")
    returns.to_csv(returns_path, index=False, encoding="utf-8-sig")
    returns.head(20).to_csv(top20_path, index=False, encoding="utf-8-sig")

    score_rows = []
    portfolios = [
        ("current_guarded", ["688082", "688072"], [0.55, 0.45]),
        ("old_drag_pair", ["688082", "000657"], [0.60, 0.40]),
    ]
    for name, selected, weights in portfolios:
        for ret_col, rank_col in (
            ("t1_open_return", "t1_open_rank"),
            ("t1_close_return", "t1_close_rank"),
            ("t5_open_return", "t5_open_rank"),
            ("t5_close_return", "t5_close_rank"),
        ):
            row = _score_portfolio(returns, selected, weights, ret_col, rank_col)
            row["portfolio_name"] = name
            score_rows.append(row)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")

    print("=== Strict week Tushare market ===", flush=True)
    print(f"market_rows={len(market)} dates={','.join(sorted(market['trade_date'].unique()))}", flush=True)
    print(f"return_rows={len(returns)}", flush=True)
    print(scores.to_string(index=False), flush=True)
    print(f"market={market_path}", flush=True)
    print(f"returns={returns_path}", flush=True)
    print(f"selected_scores={score_path}", flush=True)
    print(f"t5_close_top20={top20_path}", flush=True)


if __name__ == "__main__":
    main()
