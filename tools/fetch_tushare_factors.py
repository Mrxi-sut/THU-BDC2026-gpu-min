"""
Fetch raw snapshots for the 5000-point factor plan.

This is a pre-race data preparation tool. It requires TUSHARE_TOKEN in the
environment and writes raw CSV files only. The training and prediction code do
not import this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "data" / "raw" / "tushare"
STOCK_DATA_PATH = ROOT / "data" / "stock_data.csv"
HS300_LIST_PATH = ROOT / "data" / "hs300_stock_list.csv"

DATE_INTERFACES = {
    "daily_basic": {
        "folder": "daily_basic",
        "fields": [
            "ts_code",
            "trade_date",
            "close",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
        ],
        "extra": {"ts_code": ""},
    },
    "moneyflow": {
        "folder": "moneyflow",
        "fields": [
            "ts_code",
            "trade_date",
            "buy_sm_amount",
            "sell_sm_amount",
            "buy_md_amount",
            "sell_md_amount",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
            "net_mf_amount",
        ],
        "extra": {},
    },
    "sw_daily": {
        "folder": "sw_daily",
        "fields": [
            "ts_code",
            "trade_date",
            "name",
            "open",
            "low",
            "high",
            "close",
            "change",
            "pct_change",
            "vol",
            "amount",
            "pe",
            "pb",
            "float_mv",
            "total_mv",
        ],
        "extra": {},
    },
    "top_list": {
        "folder": "top_list",
        "fields": [
            "trade_date",
            "ts_code",
            "name",
            "close",
            "pct_chg",
            "turnover_rate",
            "amount",
            "net_amount",
        ],
        "extra": {},
    },
    "top_inst": {
        "folder": "top_inst",
        "fields": [
            "trade_date",
            "ts_code",
            "exalter",
            "buy",
            "buy_rate",
            "sell",
            "sell_rate",
            "net_buy",
        ],
        "extra": {},
    },
    "stk_limit": {
        "folder": "stk_limit",
        "fields": [
            "trade_date",
            "ts_code",
            "pre_close",
            "up_limit",
            "down_limit",
        ],
        "extra": {},
    },
}

RANGE_INTERFACES = {
    "index_weight": {
        "folder": "index_weight",
        "fields": ["index_code", "con_code", "trade_date", "weight"],
    },
}

FINANCIAL_INTERFACES = {
    "fina_indicator": [
        "ts_code",
        "ann_date",
        "end_date",
        "roe",
        "roe_dt",
        "grossprofit_margin",
        "netprofit_margin",
        "or_yoy",
        "netprofit_yoy",
        "debt_to_assets",
        "ocf_to_profit",
    ],
    "forecast": ["ts_code", "ann_date", "end_date", "type", "p_change_min", "p_change_max", "net_profit_min", "net_profit_max"],
    "express": ["ts_code", "ann_date", "end_date", "revenue", "operate_profit", "total_profit", "n_income", "total_assets"],
}

STOCK_RANGE_INTERFACES = {
    "cyq_perf": {
        "folder": "cyq_perf",
        "fields": [
            "ts_code",
            "trade_date",
            "his_low",
            "his_high",
            "cost_5pct",
            "cost_15pct",
            "cost_50pct",
            "cost_85pct",
            "cost_95pct",
            "weight_avg",
            "winner_rate",
        ],
    },
}


def to_ts_code(code: str) -> str:
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


def load_universe(max_stocks: int | None) -> list[str]:
    codes: list[str] = []
    if HS300_LIST_PATH.exists():
        hs300 = pd.read_csv(HS300_LIST_PATH, dtype=str)
        source_col = "code" if "code" in hs300.columns else hs300.columns[0]
        codes = [to_ts_code(code) for code in hs300[source_col].dropna().tolist()]
    elif STOCK_DATA_PATH.exists():
        stock_data = pd.read_csv(STOCK_DATA_PATH, usecols=["股票代码"], dtype=str)
        codes = [to_ts_code(code) for code in stock_data["股票代码"].dropna().unique()]
    codes = sorted({code for code in codes if code})
    return codes[:max_stocks] if max_stocks else codes


def load_trade_dates(start_date: str | None, end_date: str | None, max_trade_dates: int | None) -> list[str]:
    if not STOCK_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing base data: {STOCK_DATA_PATH}")

    df = pd.read_csv(STOCK_DATA_PATH, usecols=["日期"])
    dates = pd.to_datetime(df["日期"], errors="coerce").dropna().drop_duplicates().sort_values()
    if start_date:
        dates = dates[dates >= pd.to_datetime(start_date)]
    if end_date:
        dates = dates[dates <= pd.to_datetime(end_date)]
    values = [d.strftime("%Y%m%d") for d in dates.tolist()]
    return values[-max_trade_dates:] if max_trade_dates else values


def ensure_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        try:
            token = (ts.get_token() or "").strip()
        except Exception:
            token = ""
    if not token:
        raise SystemExit("TUSHARE_TOKEN is not set and no SDK token was found.")
    return token


def query(pro, interface: str, **params) -> pd.DataFrame:
    fields = params.pop("fields", None)
    if fields:
        params["fields"] = ",".join(fields)
    return pro.query(interface, **params)


def save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def should_skip_existing(path: Path, skip_existing: bool) -> bool:
    return skip_existing and path.exists() and path.stat().st_size > 0


def filter_universe(df: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ("ts_code", "con_code"):
        if col in df.columns:
            return df[df[col].isin(universe)].copy()
    return df


def fetch_by_trade_date(
    pro,
    interface: str,
    spec: dict,
    dates: list[str],
    universe: set[str],
    raw_root: Path,
    sleep_s: float,
    skip_existing: bool,
) -> list[dict]:
    records = []
    for trade_date in dates:
        params = dict(spec.get("extra", {}))
        params["trade_date"] = trade_date
        params["fields"] = spec["fields"]
        out_path = raw_root / spec["folder"] / f"{interface}_{trade_date}.csv"
        if should_skip_existing(out_path, skip_existing):
            records.append({"interface": interface, "date": trade_date, "status": "skipped_existing", "rows": None, "file": str(out_path)})
            continue
        try:
            df = query(pro, interface, **params)
            if interface != "sw_daily":
                df = filter_universe(df, universe)
            save_frame(df, out_path)
            records.append({"interface": interface, "date": trade_date, "status": "ok", "rows": int(len(df)), "file": str(out_path)})
        except Exception as exc:  # noqa: BLE001 - keep batch running and record failed slices.
            records.append({"interface": interface, "date": trade_date, "status": "error", "rows": 0, "error": str(exc)})
        if sleep_s:
            time.sleep(sleep_s)
    return records


def fetch_index_weight(
    pro,
    start_date: str,
    end_date: str,
    index_code: str,
    universe: set[str],
    raw_root: Path,
    skip_existing: bool,
) -> dict:
    spec = RANGE_INTERFACES["index_weight"]
    out_path = raw_root / spec["folder"] / f"index_weight_{index_code.replace('.', '')}_{start_date}_{end_date}.csv"
    if should_skip_existing(out_path, skip_existing):
        return {"interface": "index_weight", "status": "skipped_existing", "rows": None, "file": str(out_path)}
    try:
        df = query(
            pro,
            "index_weight",
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
            fields=spec["fields"],
        )
        df = filter_universe(df, universe)
        save_frame(df, out_path)
        return {"interface": "index_weight", "status": "ok", "rows": int(len(df)), "file": str(out_path)}
    except Exception as exc:  # noqa: BLE001
        return {"interface": "index_weight", "status": "error", "rows": 0, "error": str(exc)}


def fetch_financial(
    pro,
    interface: str,
    fields: list[str],
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    raw_root: Path,
    sleep_s: float,
    skip_existing: bool,
) -> list[dict]:
    records = []
    for code in stock_codes:
        out_path = raw_root / "financial" / f"{interface}_{code.replace('.', '')}_{start_date}_{end_date}.csv"
        if should_skip_existing(out_path, skip_existing):
            records.append({"interface": interface, "ts_code": code, "status": "skipped_existing", "rows": None, "file": str(out_path)})
            continue
        try:
            df = query(pro, interface, ts_code=code, start_date=start_date, end_date=end_date, fields=fields)
            save_frame(df, out_path)
            records.append({"interface": interface, "ts_code": code, "status": "ok", "rows": int(len(df)), "file": str(out_path)})
        except Exception as exc:  # noqa: BLE001
            records.append({"interface": interface, "ts_code": code, "status": "error", "rows": 0, "error": str(exc)})
        if sleep_s:
            time.sleep(sleep_s)
    return records


def fetch_stock_range(
    pro,
    interface: str,
    spec: dict,
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    raw_root: Path,
    sleep_s: float,
    skip_existing: bool,
) -> list[dict]:
    records = []
    folder = spec["folder"]
    fields = spec["fields"]
    for code in stock_codes:
        out_path = raw_root / folder / f"{interface}_{code.replace('.', '')}_{start_date}_{end_date}.csv"
        if should_skip_existing(out_path, skip_existing):
            records.append({"interface": interface, "ts_code": code, "status": "skipped_existing", "rows": None, "file": str(out_path)})
            continue
        try:
            df = query(pro, interface, ts_code=code, start_date=start_date, end_date=end_date, fields=fields)
            save_frame(df, out_path)
            records.append({"interface": interface, "ts_code": code, "status": "ok", "rows": int(len(df)), "file": str(out_path)})
        except Exception as exc:  # noqa: BLE001
            records.append({"interface": interface, "ts_code": code, "status": "error", "rows": 0, "error": str(exc)})
        if sleep_s:
            time.sleep(sleep_s)
    return records


def fetch_index_classify(
    pro,
    raw_root: Path,
    skip_existing: bool,
    src: str = "SW2021",
) -> list[dict]:
    records = []
    out_path = raw_root / "industry_meta" / f"index_classify_{src}.csv"
    if should_skip_existing(out_path, skip_existing):
        return [{"interface": "index_classify", "status": "skipped_existing", "rows": None, "file": str(out_path)}]

    fields = ["index_code", "industry_name", "level", "industry_code", "src"]
    frames = []
    for level in ("L1", "L2", "L3"):
        try:
            df = query(pro, "index_classify", level=level, src=src, fields=fields)
            df["level_requested"] = level
            frames.append(df)
            records.append({"interface": "index_classify", "level": level, "status": "ok", "rows": int(len(df))})
        except Exception as exc:  # noqa: BLE001
            records.append({"interface": "index_classify", "level": level, "status": "error", "rows": 0, "error": str(exc)})
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        save_frame(combined, out_path)
        records.append({"interface": "index_classify", "status": "ok", "rows": int(len(combined)), "file": str(out_path)})
    return records


def fetch_index_member_all(
    pro,
    stock_codes: list[str],
    raw_root: Path,
    sleep_s: float,
    skip_existing: bool,
) -> list[dict]:
    records = []
    fields = [
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "ts_code",
        "name",
        "in_date",
        "out_date",
        "is_new",
    ]
    for code in stock_codes:
        out_path = raw_root / "industry_member" / f"index_member_all_{code.replace('.', '')}.csv"
        if should_skip_existing(out_path, skip_existing):
            records.append({"interface": "index_member_all", "ts_code": code, "status": "skipped_existing", "rows": None, "file": str(out_path)})
            continue
        try:
            df = query(pro, "index_member_all", ts_code=code, fields=fields)
            save_frame(df, out_path)
            records.append({"interface": "index_member_all", "ts_code": code, "status": "ok", "rows": int(len(df)), "file": str(out_path)})
        except Exception as exc:  # noqa: BLE001
            records.append({"interface": "index_member_all", "ts_code": code, "status": "error", "rows": 0, "error": str(exc)})
        if sleep_s:
            time.sleep(sleep_s)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch raw Tushare factor data.")
    parser.add_argument("--interfaces", nargs="*", default=["daily_basic", "moneyflow", "index_weight"])
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD or YYYYMMDD; defaults to base data min date")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD or YYYYMMDD; defaults to base data max date")
    parser.add_argument("--max-trade-dates", type=int, default=None, help="Use the last N base trading dates for smoke tests.")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit universe for smoke tests.")
    parser.add_argument("--index-code", default="399300.SZ")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--skip-existing", action="store_true", help="已有 raw CSV 分片则跳过，便于断点续跑")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = pd.to_datetime(args.start_date).strftime("%Y%m%d") if args.start_date else None
    end = pd.to_datetime(args.end_date).strftime("%Y%m%d") if args.end_date else None
    dates = load_trade_dates(args.start_date, args.end_date, args.max_trade_dates)
    if not dates:
        raise SystemExit("No trade dates selected.")
    start = start or dates[0]
    end = end or dates[-1]
    stock_codes = load_universe(args.max_stocks)
    universe = set(stock_codes)
    raw_root = Path(args.raw_root)

    plan = {
        "interfaces": args.interfaces,
        "start": start,
        "end": end,
        "trade_dates": len(dates),
        "stocks": len(stock_codes),
        "raw_root": str(raw_root),
        "skip_existing": bool(args.skip_existing),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    token = ensure_token()
    ts.set_token(token)
    pro = ts.pro_api()

    records: list[dict] = []
    for interface in args.interfaces:
        if interface in DATE_INTERFACES:
            records.extend(
                fetch_by_trade_date(
                    pro,
                    interface,
                    DATE_INTERFACES[interface],
                    dates,
                    universe,
                    raw_root,
                    args.sleep,
                    args.skip_existing,
                )
            )
        elif interface == "index_weight":
            records.append(fetch_index_weight(pro, start, end, args.index_code, universe, raw_root, args.skip_existing))
        elif interface in FINANCIAL_INTERFACES:
            records.extend(
                fetch_financial(
                    pro,
                    interface,
                    FINANCIAL_INTERFACES[interface],
                    stock_codes,
                    start,
                    end,
                    raw_root,
                    args.sleep,
                    args.skip_existing,
                )
            )
        elif interface in STOCK_RANGE_INTERFACES:
            records.extend(
                fetch_stock_range(
                    pro,
                    interface,
                    STOCK_RANGE_INTERFACES[interface],
                    stock_codes,
                    start,
                    end,
                    raw_root,
                    args.sleep,
                    args.skip_existing,
                )
            )
        elif interface == "index_classify":
            records.extend(fetch_index_classify(pro, raw_root, args.skip_existing))
        elif interface == "index_member_all":
            records.extend(fetch_index_member_all(pro, stock_codes, raw_root, args.sleep, args.skip_existing))
        else:
            records.append({"interface": interface, "status": "skipped", "rows": 0, "error": "unknown interface"})

    raw_root.mkdir(parents=True, exist_ok=True)
    audit_path = raw_root / "_fetch_audit.csv"
    pd.DataFrame(records).to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"audit: {audit_path}")
    failures = [row for row in records if row.get("status") == "error"]
    if failures:
        print(f"failed slices: {len(failures)}", file=sys.stderr)


if __name__ == "__main__":
    main()
