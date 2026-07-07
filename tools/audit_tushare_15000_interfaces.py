"""
Smoke-test and rank Tushare 15000-point candidate interfaces.

This script is deliberately non-destructive: it only reads a few sample rows
from each API and writes an audit CSV/JSON. It never stores the token, never
downloads bulk raw data, and does not change model outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_PATH = ROOT / "data" / "stock_data.csv"
HS300_LIST_PATH = ROOT / "data" / "hs300_stock_list.csv"
DEFAULT_OUTPUT = ROOT / "output" / "tushare_15000_interface_smoke.csv"
DEFAULT_JSON = ROOT / "output" / "tushare_15000_interface_smoke.json"


@dataclass(frozen=True)
class Candidate:
    interface: str
    priority: str
    phase: str
    min_points: int | None
    independent_permission: bool
    factor_family: str
    visibility: str
    admission: str
    source_url: str
    notes: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        interface="cyq_perf",
        priority="P0",
        phase="15000",
        min_points=5000,
        independent_permission=False,
        factor_family="chip_cost_pressure",
        visibility="18-19点更新；统一按下一交易日可见",
        admission="第一批候选。只接入rank/变化/距离类特征，通过滚动IC与A1/A2/端午/本周回放后再进production。",
        source_url="https://tushare.pro/wctapi/documents/293.md",
        notes="15000后特色数据无总量限制；比cyq_chips轻，优先级最高。",
    ),
    Candidate(
        interface="cyq_chips",
        priority="P1",
        phase="15000",
        min_points=5000,
        independent_permission=False,
        factor_family="chip_distribution_shape",
        visibility="18-19点更新；统一按下一交易日可见",
        admission="候选。只做小样本烟测；全量前必须先设计聚合特征，避免超大行数直接噪声入模。",
        source_url="https://tushare.pro/wctapi/documents/294.md",
        notes="数据量大，适合做筹码集中度/上方套牢盘/下方获利盘等聚合。",
    ),
    Candidate(
        interface="index_classify",
        priority="P0",
        phase="5000",
        min_points=2000,
        independent_permission=False,
        factor_family="industry_mapping",
        visibility="静态分类；按版本可见，禁止未来行业成分回填",
        admission="行业因子前置表。先做SW2021一级/二级/三级映射，再生成点时行业暴露。",
        source_url="https://tushare.pro/wctapi/documents/181.md",
        notes="申万行业分类表，配合index_member_all与sw_daily。",
    ),
    Candidate(
        interface="index_member_all",
        priority="P0",
        phase="5000",
        min_points=2000,
        independent_permission=False,
        factor_family="industry_mapping",
        visibility="成分需按in_date/out_date或is_new做点时过滤",
        admission="行业因子前置表。必须审计in_date/out_date/is_new字段，不能拿未来成分解释过去。",
        source_url="https://tushare.pro/document/1?doc_id=108",
        notes="可按股票代码或行业代码查询所属申万行业。",
    ),
    Candidate(
        interface="sw_daily",
        priority="P0",
        phase="5000",
        min_points=5000,
        independent_permission=False,
        factor_family="industry_rotation",
        visibility="盘后更新；统一按下一交易日可见",
        admission="第一批行业强弱候选。只接入行业动量、行业宽度、个股相对行业强度。",
        source_url="https://tushare.pro/wctapi/documents/327.md",
        notes="申万行业日线行情，默认SW2021。",
    ),
    Candidate(
        interface="dc_index",
        priority="P1",
        phase="15000",
        min_points=6000,
        independent_permission=False,
        factor_family="concept_rotation",
        visibility="盘后数据；统一按下一交易日可见",
        admission="概念热度候选。先做A2尖峰行情识别，若提高A2/不伤A1和端午再晋级。",
        source_url="https://tushare.pro/wctapi/documents/362.md",
        notes="东方财富概念/行业/地域板块，含领涨股、涨跌家数、换手率。",
    ),
    Candidate(
        interface="dc_member",
        priority="P1",
        phase="15000",
        min_points=6000,
        independent_permission=False,
        factor_family="concept_membership",
        visibility="每日成分；必须按trade_date点时匹配",
        admission="概念映射候选。只服务concept_rotation，不直接裸特征入模。",
        source_url="https://tushare.pro/wctapi/documents/363.md",
        notes="每日板块成分，优先于静态概念映射。",
    ),
    Candidate(
        interface="ths_index",
        priority="P1",
        phase="15000",
        min_points=6000,
        independent_permission=False,
        factor_family="concept_catalog",
        visibility="板块目录；按list_date过滤",
        admission="概念目录候选。与dc_index二选一或互证，禁止重复噪声叠加。",
        source_url="https://tushare.pro/wctapi/documents/259.md",
        notes="同花顺概念/行业/特色指数目录。",
    ),
    Candidate(
        interface="ths_member",
        priority="P1",
        phase="15000",
        min_points=6000,
        independent_permission=False,
        factor_family="concept_membership",
        visibility="概念成分；如无in/out日期则仅作当前截面候选，不回填历史",
        admission="候选。若缺少历史点时字段，则不能用于历史回测主链，只能用于当前预测辅助。",
        source_url="https://tushare.pro/wctapi/documents/261.md",
        notes="同花顺概念板块成分。",
    ),
    Candidate(
        interface="moneyflow_hsgt",
        priority="P1",
        phase="5000",
        min_points=2000,
        independent_permission=False,
        factor_family="market_regime_north_money",
        visibility="盘后/次日可见；作为市场状态，不作为个股alpha裸因子",
        admission="市场状态候选。用于Top2/Top3 selector和风险开关，不直接决定单股。",
        source_url="https://tushare.pro/wctapi/documents/47.md",
        notes="北向/南向资金流向，市场级变量。",
    ),
    Candidate(
        interface="broker_recommend",
        priority="P2",
        phase="15000",
        min_points=6000,
        independent_permission=False,
        factor_family="sell_side_attention",
        visibility="月度1-3日更新；按month滞后到可见日",
        admission="慢变量候选。短周期T+5不一定有效，先做弱信号审计。",
        source_url="https://tushare.pro/wctapi/documents/267.md",
        notes="券商月度金股。",
    ),
    Candidate(
        interface="hk_hold",
        priority="P2",
        phase="5000",
        min_points=2000,
        independent_permission=False,
        factor_family="northbound_holding",
        visibility="2024-08-20后改为季度披露；必须按披露频率可见",
        admission="暂缓。当前比赛期日频价值下降，只能作为季度慢变量或历史研究。",
        source_url="https://tushare.pro/wctapi/documents/188.md",
        notes="交易所已停止日度北向持股披露，不能按旧日频逻辑使用。",
    ),
)


def load_base_dates() -> list[str]:
    if not BASE_DATA_PATH.exists():
        return []
    df = pd.read_csv(BASE_DATA_PATH, usecols=["日期"])
    dates = pd.to_datetime(df["日期"], errors="coerce").dropna().drop_duplicates().sort_values()
    return [d.strftime("%Y%m%d") for d in dates.tolist()]


def to_ts_code(code: str) -> str:
    text = str(code).strip()
    if text.startswith("sh."):
        return f"{text[3:]}.SH"
    if text.startswith("sz."):
        return f"{text[3:]}.SZ"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 6:
        return ""
    return f"{digits}.SH" if digits.startswith(("5", "6", "9")) else f"{digits}.SZ"


def load_sample_stock(default: str = "300408.SZ") -> str:
    if HS300_LIST_PATH.exists():
        df = pd.read_csv(HS300_LIST_PATH, dtype=str)
        col = "code" if "code" in df.columns else df.columns[0]
        for code in df[col].dropna().tolist():
            ts_code = to_ts_code(code)
            if ts_code:
                return ts_code
    if BASE_DATA_PATH.exists():
        df = pd.read_csv(BASE_DATA_PATH, usecols=["股票代码"], dtype=str)
        for code in df["股票代码"].dropna().unique().tolist():
            ts_code = to_ts_code(code)
            if ts_code:
                return ts_code
    return default


def month_from_date(date_text: str) -> str:
    return date_text[:6]


def query(pro: Any, interface: str, fields: list[str] | None = None, **params: Any) -> pd.DataFrame:
    clean_params = {key: value for key, value in params.items() if value not in (None, "")}
    if fields:
        clean_params["fields"] = ",".join(fields)
    return pro.query(interface, **clean_params)


def frame_preview(df: pd.DataFrame, max_cols: int = 16) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "columns": [], "sample": {}}
    cols = list(df.columns)
    sample = df.head(1).fillna("").to_dict(orient="records")[0]
    return {
        "rows": int(len(df)),
        "columns": cols,
        "sample": {key: sample[key] for key in cols[:max_cols]},
    }


def ok_record(candidate: Candidate, params: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    preview = frame_preview(df)
    return {
        **asdict(candidate),
        "status": "ok",
        "request_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "rows": preview["rows"],
        "columns": "|".join(preview["columns"]),
        "sample": json.dumps(preview["sample"], ensure_ascii=False),
        "error": "",
    }


def error_record(candidate: Candidate, params: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        **asdict(candidate),
        "status": "error",
        "request_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "rows": 0,
        "columns": "",
        "sample": "",
        "error": str(exc),
    }


def skipped_record(candidate: Candidate, reason: str) -> dict[str, Any]:
    return {
        **asdict(candidate),
        "status": "skipped",
        "request_params": "{}",
        "rows": 0,
        "columns": "",
        "sample": "",
        "error": reason,
    }


def run_smoke(pro: Any, candidate: Candidate, sample_date: str, sample_stock: str) -> tuple[dict[str, Any], pd.DataFrame]:
    fields: list[str] | None = None
    params: dict[str, Any] = {}

    if candidate.interface == "cyq_perf":
        params = {"ts_code": sample_stock, "start_date": sample_date, "end_date": sample_date}
        fields = [
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
        ]
    elif candidate.interface == "cyq_chips":
        params = {"ts_code": sample_stock, "trade_date": sample_date}
        fields = ["ts_code", "trade_date", "price", "percent"]
    elif candidate.interface == "index_classify":
        params = {"level": "L1", "src": "SW2021"}
        fields = ["index_code", "industry_name", "level", "industry_code", "src"]
    elif candidate.interface == "index_member_all":
        params = {"ts_code": sample_stock, "is_new": "Y"}
    elif candidate.interface == "sw_daily":
        params = {"trade_date": sample_date}
        fields = [
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
        ]
    elif candidate.interface == "dc_index":
        params = {"trade_date": sample_date, "idx_type": "概念板块"}
        fields = [
            "ts_code",
            "trade_date",
            "name",
            "leading",
            "leading_code",
            "pct_change",
            "leading_pct",
            "turnover_rate",
            "up_num",
            "down_num",
            "idx_type",
            "level",
        ]
    elif candidate.interface == "dc_member":
        dc = query(pro, "dc_index", trade_date=sample_date, idx_type="概念板块", fields=["ts_code", "name"])
        if dc.empty or "ts_code" not in dc.columns:
            return skipped_record(candidate, "dc_index returned empty; cannot choose a sample concept"), pd.DataFrame()
        concept_code = str(dc.iloc[0]["ts_code"])
        params = {"trade_date": sample_date, "ts_code": concept_code}
        fields = ["trade_date", "ts_code", "con_code", "name"]
    elif candidate.interface == "ths_index":
        params = {"exchange": "A", "type": "N"}
        fields = ["ts_code", "name", "count", "exchange", "list_date", "type"]
    elif candidate.interface == "ths_member":
        ths = query(pro, "ths_index", exchange="A", type="N", fields=["ts_code", "name", "count"])
        if ths.empty or "ts_code" not in ths.columns:
            return skipped_record(candidate, "ths_index returned empty; cannot choose a sample concept"), pd.DataFrame()
        concept_code = str(ths.iloc[0]["ts_code"])
        params = {"ts_code": concept_code}
        fields = ["ts_code", "con_code", "con_name", "weight", "in_date", "out_date", "is_new"]
    elif candidate.interface == "moneyflow_hsgt":
        params = {"trade_date": sample_date}
        fields = ["trade_date", "ggt_ss", "ggt_sz", "hgt", "sgt", "north_money", "south_money"]
    elif candidate.interface == "broker_recommend":
        params = {"month": month_from_date(sample_date)}
        fields = ["month", "broker", "ts_code", "name"]
    elif candidate.interface == "hk_hold":
        params = {"ts_code": sample_stock, "start_date": sample_date, "end_date": sample_date}
        fields = ["code", "trade_date", "ts_code", "name", "vol", "ratio", "exchange"]
    else:
        return skipped_record(candidate, "unknown interface"), pd.DataFrame()

    try:
        df = query(pro, candidate.interface, fields=fields, **params)
        return ok_record(candidate, params, df), df
    except Exception as exc:  # noqa: BLE001 - audit should keep going.
        return error_record(candidate, params, exc), pd.DataFrame()


def write_outputs(records: list[dict[str, Any]], output: Path, json_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, encoding="utf-8-sig")
    json_output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Tushare 15000 candidate interfaces.")
    parser.add_argument("--sample-date", default=None, help="YYYYMMDD; defaults to latest date in data/stock_data.csv")
    parser.add_argument("--sample-stock", default=None, help="TS code, e.g. 300408.SZ; defaults to first HS300 code")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dates = load_base_dates()
    sample_date = args.sample_date or (dates[-1] if dates else "20260622")
    sample_stock = args.sample_stock or load_sample_stock()

    plan = {
        "sample_date": sample_date,
        "sample_stock": sample_stock,
        "interfaces": [candidate.interface for candidate in CANDIDATES],
        "output": args.output,
        "json_output": args.json_output,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        records = [
            {
                **asdict(candidate),
                "status": "planned",
                "request_params": "",
                "rows": "",
                "columns": "",
                "sample": "",
                "error": "",
            }
            for candidate in CANDIDATES
        ]
        write_outputs(records, Path(args.output), Path(args.json_output))
        print(f"planned audit: {args.output}")
        return

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        try:
            token = (ts.get_token() or "").strip()
        except Exception:
            token = ""
    if not token:
        raise SystemExit("TUSHARE_TOKEN is not set and no SDK token was found.")

    ts.set_token(token)
    pro = ts.pro_api()

    records: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        record, _ = run_smoke(pro, candidate, sample_date, sample_stock)
        records.append(record)
        print(f"{candidate.interface}: {record['status']} rows={record['rows']}")
        if args.sleep:
            time.sleep(args.sleep)

    write_outputs(records, Path(args.output), Path(args.json_output))
    failures = [row for row in records if row["status"] == "error"]
    print(f"audit: {args.output}")
    if failures:
        print(f"failed interfaces: {len(failures)}", file=sys.stderr)


if __name__ == "__main__":
    main()
