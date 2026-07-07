"""
Build local factor CSVs from frozen raw data.

This script never calls online APIs. It reads files under data/raw and writes
model-ready CSVs under data/offline_factors. The model can run without these
files, so every new factor layer remains easy to ablate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "data" / "raw" / "tushare"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "offline_factors"
BASE_DATA_PATH = ROOT / "data" / "stock_data.csv"
INDEX_WEIGHT_ASOF_POLICY = "trade_date"


def read_raw_folder(raw_root: Path, name: str) -> pd.DataFrame:
    folder = raw_root / name
    if not folder.exists():
        return pd.DataFrame()

    frames = []
    for path in sorted(folder.glob("*.csv")):
        if path.name.startswith("_"):
            continue
        try:
            frame = pd.read_csv(path, dtype=str)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frame["_source_file"] = path.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def stock_code_from_ts(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    return parsed.fillna(fallback)


def normalize_trade_table(df: pd.DataFrame, code_col: str, date_col: str) -> pd.DataFrame:
    if df.empty or code_col not in df.columns or date_col not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["股票代码"] = stock_code_from_ts(out[code_col])
    out["日期"] = parse_yyyymmdd(out[date_col])
    out = out[(out["股票代码"].str.len() == 6) & out["日期"].notna()].copy()
    return out


def to_number(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pct_rank_by_date(df: pd.DataFrame, source: str, target: str, high_is_good: bool = True) -> None:
    if source not in df.columns:
        return
    df[target] = (
        df.groupby("日期")[source]
        .rank(pct=True, ascending=high_is_good, method="average")
        .fillna(0.5)
    )


def base_stock_dates() -> pd.DataFrame:
    if not BASE_DATA_PATH.exists():
        return pd.DataFrame(columns=["股票代码", "日期"])
    base = pd.read_csv(BASE_DATA_PATH, usecols=["股票代码", "日期"], dtype={"股票代码": str})
    base["股票代码"] = base["股票代码"].astype(str).str.zfill(6)
    base["日期"] = pd.to_datetime(base["日期"], errors="coerce")
    return base.dropna(subset=["日期"]).drop_duplicates(["股票代码", "日期"]).sort_values(["股票代码", "日期"])


def base_stock_frame() -> pd.DataFrame:
    if not BASE_DATA_PATH.exists():
        return pd.DataFrame(columns=["股票代码", "日期"])
    base = pd.read_csv(BASE_DATA_PATH, dtype={"股票代码": str})
    base["股票代码"] = base["股票代码"].astype(str).str.zfill(6)
    base["日期"] = pd.to_datetime(base["日期"], errors="coerce")
    for col in ("开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率", "涨跌幅"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
    return base.dropna(subset=["日期"]).sort_values(["股票代码", "日期"]).reset_index(drop=True)


def zscore_by_date(df: pd.DataFrame, source: str, target: str) -> None:
    if source not in df.columns:
        return
    grouped = df.groupby("日期")[source]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    df[target] = (df[source] - mean) / (std + 1e-12)


def safe_rank_by_date(df: pd.DataFrame, source: str, target: str, high_is_good: bool = True) -> None:
    if source not in df.columns:
        df[target] = 0.5
        return
    pct_rank_by_date(df, source, target, high_is_good=high_is_good)
    df[target] = df[target].fillna(0.5)


def write_factor(df: pd.DataFrame, output_dir: Path, filename: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if df.empty:
        return {"file": str(path), "status": "empty", "rows": 0, "stocks": 0}

    df = df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).drop_duplicates(["股票代码", "日期"], keep="last")
    df = df.sort_values(["日期", "股票代码"]).reset_index(drop=True)
    for col in (
        "日期",
        "available_date",
        "publish_datetime",
        "weight_trade_date",
        "weight_available_date",
        "weight_month_end_date",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "file": str(path),
        "status": "ok",
        "rows": int(len(df)),
        "stocks": int(df["股票代码"].nunique()),
        "min_date": str(df["日期"].min()),
        "max_date": str(df["日期"].max()),
    }


def build_daily_basic(raw_root: Path, output_dir: Path) -> dict:
    raw = normalize_trade_table(read_raw_folder(raw_root, "daily_basic"), "ts_code", "trade_date")
    if raw.empty:
        return write_factor(raw, output_dir, "daily_basic_factors.csv")

    numeric = [
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
    ]
    raw = to_number(raw, numeric)

    out = raw[["股票代码", "日期"]].copy()
    out["available_date"] = out["日期"]
    for col in numeric:
        if col in raw.columns:
            out[f"ts_db_{col}"] = raw[col]

    for col in ("total_mv", "circ_mv", "free_share", "float_share"):
        source = f"ts_db_{col}"
        if source in out.columns:
            out[f"ts_db_log_{col}"] = np.log1p(out[source].clip(lower=0))
            pct_rank_by_date(out, f"ts_db_log_{col}", f"ts_db_log_{col}_rank")

    for col in ("pb", "pe_ttm", "ps_ttm", "pe", "ps"):
        source = f"ts_db_{col}"
        if source in out.columns:
            pct_rank_by_date(out, source, f"ts_db_{col}_value_rank", high_is_good=False)

    for col in ("turnover_rate", "turnover_rate_f", "volume_ratio", "dv_ttm"):
        source = f"ts_db_{col}"
        if source in out.columns:
            pct_rank_by_date(out, source, f"ts_db_{col}_rank", high_is_good=True)

    return write_factor(out, output_dir, "daily_basic_factors.csv")


def build_daily_basic_core(raw_root: Path, output_dir: Path) -> dict:
    """
    Low-noise daily_basic subset.

    The raw daily_basic file is useful but too wide: valuation, liquidity and
    size columns fight the price model when dumped in directly. This file keeps
    only rank-normalized, interpretable composites for ablation.
    """
    raw = normalize_trade_table(read_raw_folder(raw_root, "daily_basic"), "ts_code", "trade_date")
    if raw.empty:
        return write_factor(raw, output_dir, "daily_basic_core_factors.csv")

    numeric = [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ttm",
        "free_share",
        "float_share",
        "total_mv",
        "circ_mv",
    ]
    raw = to_number(raw, numeric)
    out = raw[["股票代码", "日期"]].copy()
    out["available_date"] = out["日期"]

    eps = 1e-12
    size_source = raw["circ_mv"].where(raw.get("circ_mv").notna(), raw.get("total_mv"))
    out["ts_dbc_log_circ_mv"] = np.log1p(pd.to_numeric(size_source, errors="coerce").clip(lower=0))
    out["ts_dbc_log_total_mv"] = np.log1p(pd.to_numeric(raw.get("total_mv"), errors="coerce").clip(lower=0))
    out["ts_dbc_free_share_log"] = np.log1p(pd.to_numeric(raw.get("free_share"), errors="coerce").clip(lower=0))
    out["ts_dbc_float_share_log"] = np.log1p(pd.to_numeric(raw.get("float_share"), errors="coerce").clip(lower=0))

    pe_ttm = pd.to_numeric(raw.get("pe_ttm"), errors="coerce")
    pe = pd.to_numeric(raw.get("pe"), errors="coerce")
    pb = pd.to_numeric(raw.get("pb"), errors="coerce")
    ps_ttm = pd.to_numeric(raw.get("ps_ttm"), errors="coerce")
    dv_ttm = pd.to_numeric(raw.get("dv_ttm"), errors="coerce")
    turnover = pd.to_numeric(raw.get("turnover_rate_f"), errors="coerce").fillna(
        pd.to_numeric(raw.get("turnover_rate"), errors="coerce")
    )
    volume_ratio = pd.to_numeric(raw.get("volume_ratio"), errors="coerce")

    # Negative PE is usually not "cheap"; neutralize before low-valuation rank.
    out["ts_dbc_pe_ttm_pos"] = pe_ttm.where(pe_ttm > 0)
    out["ts_dbc_pe_pos"] = pe.where(pe > 0)
    out["ts_dbc_pb_pos"] = pb.where(pb > 0)
    out["ts_dbc_ps_ttm_pos"] = ps_ttm.where(ps_ttm > 0)
    out["ts_dbc_dv_ttm"] = dv_ttm
    out["ts_dbc_turnover_rate_f"] = turnover
    out["ts_dbc_volume_ratio"] = volume_ratio

    turnover_target = np.log1p(2.2)
    out["ts_dbc_turnover_moderate"] = 1.0 - (
        (np.log1p(turnover.clip(lower=0)) - turnover_target).abs() / np.log1p(7.0)
    ).clip(0.0, 1.0)
    volume_target = np.log1p(1.25)
    out["ts_dbc_volume_ratio_moderate"] = 1.0 - (
        (np.log1p(volume_ratio.clip(lower=0)) - volume_target).abs() / np.log1p(4.0)
    ).clip(0.0, 1.0)

    safe_rank_by_date(out, "ts_dbc_log_circ_mv", "ts_dbc_size_core_rank")
    safe_rank_by_date(out, "ts_dbc_log_total_mv", "ts_dbc_total_mv_rank")
    safe_rank_by_date(out, "ts_dbc_free_share_log", "ts_dbc_free_share_rank")
    safe_rank_by_date(out, "ts_dbc_float_share_log", "ts_dbc_float_share_rank")
    safe_rank_by_date(out, "ts_dbc_pb_pos", "ts_dbc_pb_value_rank", high_is_good=False)
    safe_rank_by_date(out, "ts_dbc_pe_ttm_pos", "ts_dbc_pe_ttm_value_rank", high_is_good=False)
    safe_rank_by_date(out, "ts_dbc_pe_pos", "ts_dbc_pe_value_rank", high_is_good=False)
    safe_rank_by_date(out, "ts_dbc_ps_ttm_pos", "ts_dbc_ps_ttm_value_rank", high_is_good=False)
    safe_rank_by_date(out, "ts_dbc_dv_ttm", "ts_dbc_dividend_rank")
    safe_rank_by_date(out, "ts_dbc_turnover_rate_f", "ts_dbc_turnover_rank")
    safe_rank_by_date(out, "ts_dbc_volume_ratio", "ts_dbc_volume_ratio_rank")
    safe_rank_by_date(out, "ts_dbc_turnover_moderate", "ts_dbc_turnover_moderate_rank")
    safe_rank_by_date(out, "ts_dbc_volume_ratio_moderate", "ts_dbc_volume_moderate_rank")

    out["ts_dbc_liquidity_quality"] = (
        0.34 * out["ts_dbc_size_core_rank"].fillna(0.5)
        + 0.24 * out["ts_dbc_free_share_rank"].fillna(0.5)
        + 0.22 * out["ts_dbc_turnover_moderate_rank"].fillna(0.5)
        + 0.20 * out["ts_dbc_volume_moderate_rank"].fillna(0.5)
    )
    out["ts_dbc_value_quality"] = (
        0.35 * out["ts_dbc_pb_value_rank"].fillna(0.5)
        + 0.25 * out["ts_dbc_pe_ttm_value_rank"].fillna(0.5)
        + 0.20 * out["ts_dbc_ps_ttm_value_rank"].fillna(0.5)
        + 0.20 * out["ts_dbc_dividend_rank"].fillna(0.5)
    )
    out["ts_dbc_core_value_liquidity"] = (
        0.45 * out["ts_dbc_liquidity_quality"].fillna(0.5)
        + 0.35 * out["ts_dbc_value_quality"].fillna(0.5)
        + 0.20 * out["ts_dbc_total_mv_rank"].fillna(0.5)
    )

    keep = [
        "股票代码",
        "日期",
        "available_date",
        "ts_dbc_size_core_rank",
        "ts_dbc_total_mv_rank",
        "ts_dbc_free_share_rank",
        "ts_dbc_float_share_rank",
        "ts_dbc_pb_value_rank",
        "ts_dbc_pe_ttm_value_rank",
        "ts_dbc_ps_ttm_value_rank",
        "ts_dbc_dividend_rank",
        "ts_dbc_turnover_rank",
        "ts_dbc_volume_ratio_rank",
        "ts_dbc_turnover_moderate_rank",
        "ts_dbc_volume_moderate_rank",
        "ts_dbc_liquidity_quality",
        "ts_dbc_value_quality",
        "ts_dbc_core_value_liquidity",
    ]
    return write_factor(out[keep], output_dir, "daily_basic_core_factors.csv")


def rolling_sum(group: pd.Series, window: int) -> pd.Series:
    return group.rolling(window, min_periods=max(2, window // 2)).sum()


def build_moneyflow(raw_root: Path, output_dir: Path) -> dict:
    raw = normalize_trade_table(read_raw_folder(raw_root, "moneyflow"), "ts_code", "trade_date")
    if raw.empty:
        return write_factor(raw, output_dir, "moneyflow_factors.csv")

    amount_cols = [
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_md_amount",
        "sell_md_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    raw = to_number(raw, amount_cols).sort_values(["股票代码", "日期"]).reset_index(drop=True)

    out = raw[["股票代码", "日期"]].copy()
    out["available_date"] = out["日期"]
    buy_main = raw.get("buy_lg_amount", 0).fillna(0) + raw.get("buy_elg_amount", 0).fillna(0)
    sell_main = raw.get("sell_lg_amount", 0).fillna(0) + raw.get("sell_elg_amount", 0).fillna(0)
    all_amount = sum(raw.get(col, 0).fillna(0) for col in amount_cols if col != "net_mf_amount")

    out["ts_mf_main_net_amount"] = buy_main - sell_main
    out["ts_mf_large_order_amount"] = buy_main + sell_main
    out["ts_mf_all_order_amount"] = all_amount
    if "net_mf_amount" in raw.columns:
        out["ts_mf_net_mf_amount"] = raw["net_mf_amount"]
    out["ts_mf_large_order_share"] = out["ts_mf_large_order_amount"] / (out["ts_mf_all_order_amount"].abs() + 1e-12)

    grouped = out.groupby("股票代码", group_keys=False)
    for window in (5, 10, 20):
        out[f"ts_mf_main_net_{window}"] = grouped["ts_mf_main_net_amount"].transform(lambda s: rolling_sum(s, window))
        out[f"ts_mf_all_order_{window}"] = grouped["ts_mf_all_order_amount"].transform(lambda s: rolling_sum(s.abs(), window))
        out[f"ts_mf_main_net_ratio_{window}"] = out[f"ts_mf_main_net_{window}"] / (
            out[f"ts_mf_all_order_{window}"].abs() + 1e-12
        )
        pct_rank_by_date(out, f"ts_mf_main_net_{window}", f"ts_mf_main_net_{window}_rank")
        pct_rank_by_date(out, f"ts_mf_main_net_ratio_{window}", f"ts_mf_main_net_ratio_{window}_rank")

    pct_rank_by_date(out, "ts_mf_large_order_share", "ts_mf_large_order_share_rank")
    if "ts_mf_net_mf_amount" in out.columns:
        pct_rank_by_date(out, "ts_mf_net_mf_amount", "ts_mf_net_mf_amount_rank")

    return write_factor(out, output_dir, "moneyflow_factors.csv")


def _next_base_date_series(dates: pd.Series) -> pd.Series:
    next_dates = next_base_date_map()
    return dates.map(next_dates)


def _date_z(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-12)


def build_moneyflow_residual(raw_root: Path, output_dir: Path) -> dict:
    """
    Lagged/residualized moneyflow.

    This intentionally does not reuse raw amount columns. Moneyflow for trade
    date T is attached to T+1, then converted into residual/rank features so it
    can only assist price-confirmed spikes instead of overpowering the model.
    """
    raw = normalize_trade_table(read_raw_folder(raw_root, "moneyflow"), "ts_code", "trade_date")
    if raw.empty:
        return write_factor(raw, output_dir, "moneyflow_residual_factors.csv")

    amount_cols = [
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_md_amount",
        "sell_md_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    raw = to_number(raw, amount_cols).sort_values(["股票代码", "日期"]).reset_index(drop=True)
    base = base_stock_frame()
    if not base.empty:
        base = base.copy()
        grouped_base = base.groupby("股票代码", group_keys=False)
        base["base_ret_1"] = grouped_base["收盘"].pct_change(1)
        base["base_ret_5"] = grouped_base["收盘"].pct_change(5)
        base["base_ret_20"] = grouped_base["收盘"].pct_change(20)
        base["base_amount_ratio_5_20"] = (
            grouped_base["成交额"].transform(lambda s: s.rolling(5, min_periods=3).mean())
            / (grouped_base["成交额"].transform(lambda s: s.rolling(20, min_periods=10).mean()) + 1e-12)
        )
        base["base_turnover"] = pd.to_numeric(base.get("换手率"), errors="coerce")
        merge_cols = [
            "股票代码",
            "日期",
            "base_ret_1",
            "base_ret_5",
            "base_ret_20",
            "base_amount_ratio_5_20",
            "base_turnover",
        ]
        raw = raw.merge(base[merge_cols], on=["股票代码", "日期"], how="left")

    buy_main = raw.get("buy_lg_amount", 0).fillna(0) + raw.get("buy_elg_amount", 0).fillna(0)
    sell_main = raw.get("sell_lg_amount", 0).fillna(0) + raw.get("sell_elg_amount", 0).fillna(0)
    buy_small = raw.get("buy_sm_amount", 0).fillna(0) + raw.get("buy_md_amount", 0).fillna(0)
    sell_small = raw.get("sell_sm_amount", 0).fillna(0) + raw.get("sell_md_amount", 0).fillna(0)
    total_order = sum(raw.get(col, 0).fillna(0).abs() for col in amount_cols if col != "net_mf_amount")

    flow = raw[["股票代码", "日期"]].copy()
    flow["mf_main_net_ratio"] = (buy_main - sell_main) / (total_order + 1e-12)
    flow["mf_small_net_ratio"] = (buy_small - sell_small) / (total_order + 1e-12)
    flow["mf_large_order_share"] = (buy_main.abs() + sell_main.abs()) / (total_order + 1e-12)
    if "net_mf_amount" in raw.columns:
        flow["mf_net_mf_ratio"] = raw["net_mf_amount"] / (total_order + 1e-12)
    else:
        flow["mf_net_mf_ratio"] = flow["mf_main_net_ratio"]
    for col in ("base_ret_1", "base_ret_5", "base_ret_20", "base_amount_ratio_5_20", "base_turnover"):
        flow[col] = pd.to_numeric(raw.get(col), errors="coerce")

    for col in (
        "mf_main_net_ratio",
        "mf_net_mf_ratio",
        "mf_large_order_share",
        "base_ret_1",
        "base_ret_5",
        "base_amount_ratio_5_20",
        "base_turnover",
    ):
        zscore_by_date(flow, col, f"{col}_z")

    flow["mf_main_residual"] = (
        flow["mf_main_net_ratio_z"].fillna(0.0)
        - 0.30 * flow["base_ret_1_z"].fillna(0.0)
        - 0.34 * flow["base_ret_5_z"].fillna(0.0)
        - 0.18 * flow["base_amount_ratio_5_20_z"].fillna(0.0)
        - 0.08 * flow["base_turnover_z"].fillna(0.0)
    )
    flow["mf_net_residual"] = (
        flow["mf_net_mf_ratio_z"].fillna(0.0)
        - 0.24 * flow["base_ret_1_z"].fillna(0.0)
        - 0.30 * flow["base_ret_5_z"].fillna(0.0)
        - 0.20 * flow["base_amount_ratio_5_20_z"].fillna(0.0)
    )

    flow = flow.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    grouped = flow.groupby("股票代码", group_keys=False)
    out = flow[["股票代码", "日期"]].copy()
    for source in ("mf_main_residual", "mf_net_residual", "mf_large_order_share"):
        for window in (3, 5, 10, 20):
            out[f"ts_mfr_{source}_{window}"] = grouped[source].transform(
                lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).mean()
            )

    out["ts_mfr_main_residual_slope_5_20"] = (
        out["ts_mfr_mf_main_residual_5"] - out["ts_mfr_mf_main_residual_20"]
    )
    out["ts_mfr_net_residual_slope_5_20"] = (
        out["ts_mfr_mf_net_residual_5"] - out["ts_mfr_mf_net_residual_20"]
    )
    out["base_ret_5"] = flow["base_ret_5"]
    out["base_ret_20"] = flow["base_ret_20"]
    out["base_amount_ratio_5_20"] = flow["base_amount_ratio_5_20"]
    safe_rank_by_date(out, "base_ret_5", "ts_mfr_base_ret5_rank")
    safe_rank_by_date(out, "base_ret_20", "ts_mfr_base_ret20_rank")
    safe_rank_by_date(out, "base_amount_ratio_5_20", "ts_mfr_base_amount_confirm_rank")

    residual_cols = [col for col in out.columns if col.startswith("ts_mfr_mf_") or col.endswith("_slope_5_20")]
    for col in residual_cols:
        safe_rank_by_date(out, col, f"{col}_rank")

    out["ts_mfr_price_confirmed_flow"] = (
        out["ts_mfr_mf_main_residual_5_rank"].fillna(0.5)
        * out["ts_mfr_base_ret5_rank"].fillna(0.5)
        * out["ts_mfr_base_amount_confirm_rank"].fillna(0.5)
    )
    out["ts_mfr_flow_divergence_candidate"] = (
        out["ts_mfr_mf_main_residual_10_rank"].fillna(0.5)
        * (1.0 - out["ts_mfr_base_ret5_rank"].fillna(0.5))
        * out["ts_mfr_base_amount_confirm_rank"].fillna(0.5)
    )
    out["ts_mfr_distribution_risk"] = (
        out["ts_mfr_mf_main_residual_5_rank"].fillna(0.5)
        * out["ts_mfr_base_ret20_rank"].fillna(0.5)
        * (1.0 - out["ts_mfr_base_amount_confirm_rank"].fillna(0.5))
    )

    # Lag1 availability: moneyflow from trade_date T can be used on next base date.
    out["moneyflow_trade_date"] = out["日期"]
    out["日期"] = _next_base_date_series(out["moneyflow_trade_date"])
    out = out.dropna(subset=["日期"]).copy()
    out["available_date"] = out["日期"]
    keep = [
        "股票代码",
        "日期",
        "available_date",
        "moneyflow_trade_date",
        "ts_mfr_mf_main_residual_3_rank",
        "ts_mfr_mf_main_residual_5_rank",
        "ts_mfr_mf_main_residual_10_rank",
        "ts_mfr_mf_main_residual_20_rank",
        "ts_mfr_mf_net_residual_5_rank",
        "ts_mfr_mf_net_residual_10_rank",
        "ts_mfr_mf_large_order_share_5_rank",
        "ts_mfr_mf_large_order_share_20_rank",
        "ts_mfr_main_residual_slope_5_20_rank",
        "ts_mfr_net_residual_slope_5_20_rank",
        "ts_mfr_price_confirmed_flow",
        "ts_mfr_flow_divergence_candidate",
        "ts_mfr_distribution_risk",
    ]
    return write_factor(out[keep], output_dir, "moneyflow_residual_factors.csv")


def build_index_weight(raw_root: Path, output_dir: Path) -> dict:
    raw = normalize_trade_table(read_raw_folder(raw_root, "index_weight"), "con_code", "trade_date")
    if raw.empty or "weight" not in raw.columns:
        return write_factor(pd.DataFrame(), output_dir, "index_weight_factors.csv")

    base = base_stock_dates()
    base_dates = np.array(sorted(base["日期"].dropna().unique()), dtype="datetime64[ns]") if not base.empty else np.array([])

    weights = raw[["股票代码", "日期", "weight"]].copy().rename(columns={"日期": "weight_trade_date"})
    weights["ts_idx_weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    weights = weights.drop(columns=["weight"]).dropna(subset=["ts_idx_weight", "weight_trade_date"])
    policy = INDEX_WEIGHT_ASOF_POLICY.strip().lower()
    if policy in {"trade", "same_day", "effective_date"}:
        policy = "trade_date"
    elif policy in {"next", "next_day"}:
        policy = "next_trade_day"
    elif policy in {"month_end", "month_end_lag"}:
        policy = "month_end_next_trade_day"
    if policy not in {"trade_date", "next_trade_day", "month_end_next_trade_day"}:
        raise ValueError(f"unknown index weight as-of policy: {INDEX_WEIGHT_ASOF_POLICY}")

    month_end_dates = weights.groupby(weights["weight_trade_date"].dt.to_period("M"))["weight_trade_date"].transform("max")
    weights["weight_month_end_date"] = month_end_dates

    def first_base_after(date: pd.Timestamp) -> pd.Timestamp:
        if not len(base_dates):
            return pd.Timestamp(date)
        pos = int(np.searchsorted(base_dates, np.datetime64(pd.Timestamp(date)), side="right"))
        if pos < len(base_dates):
            return pd.Timestamp(base_dates[pos])
        return pd.Timestamp(date)

    if policy == "trade_date":
        weights["available_date"] = weights["weight_trade_date"]
    elif policy == "next_trade_day":
        available_dates = []
        for date in weights["weight_trade_date"]:
            available_dates.append(first_base_after(pd.Timestamp(date)))
        weights["available_date"] = available_dates
    else:
        weights["available_date"] = [first_base_after(pd.Timestamp(date)) for date in weights["weight_month_end_date"]]
    weights["weight_available_date"] = weights["available_date"]
    weights = weights.sort_values(["股票代码", "weight_trade_date"])
    if base.empty:
        out = weights.rename(columns={"weight_trade_date": "日期"})
    else:
        pieces = []
        for stock_code, base_part in base.groupby("股票代码", sort=False):
            weight_part = weights[weights["股票代码"] == stock_code].sort_values("weight_trade_date")
            if weight_part.empty:
                continue
            merged = pd.merge_asof(
                base_part.sort_values("日期"),
                weight_part[
                    [
                        "weight_trade_date",
                        "weight_month_end_date",
                        "available_date",
                        "weight_available_date",
                        "ts_idx_weight",
                    ]
                ],
                left_on="日期",
                right_on="weight_trade_date",
                direction="backward",
            )
            pieces.append(merged)
        out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    if out.empty:
        return write_factor(out, output_dir, "index_weight_factors.csv")
    out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    out = out.dropna(subset=["ts_idx_weight"]).copy()
    if out.empty:
        return write_factor(out, output_dir, "index_weight_factors.csv")
    out["weight_trade_date"] = pd.to_datetime(out["weight_trade_date"], errors="coerce")
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    out["weight_available_date"] = out["available_date"]
    out["ts_idx_weight_age_days"] = (out["日期"] - out["weight_trade_date"]).dt.days
    out["ts_idx_visible_lag_days"] = (out["available_date"] - out["weight_trade_date"]).dt.days
    invalid_visible_rows = int(out["available_date"].gt(out["日期"]).sum())
    out = out[out["available_date"].isna() | out["available_date"].le(out["日期"])].copy()
    if out.empty:
        result = write_factor(out, output_dir, "index_weight_factors.csv")
        result.update(
            {
                "asof_policy": policy,
                "invalid_visible_rows_dropped": invalid_visible_rows,
            }
        )
        return result
    out["ts_idx_asof_valid"] = 1.0
    grouped_out = out.groupby("股票代码", group_keys=False)
    out["ts_idx_weight_change_5"] = grouped_out["ts_idx_weight"].diff(5).fillna(0.0)
    out["ts_idx_weight_change_20"] = grouped_out["ts_idx_weight"].diff(20).fillna(0.0)
    out["ts_idx_weight_change_60"] = grouped_out["ts_idx_weight"].diff(60).fillna(0.0)
    out["ts_idx_weight_delta_abs_20"] = out["ts_idx_weight_change_20"].abs()
    out["ts_idx_weight_accel_20_60"] = out["ts_idx_weight_change_20"] - out["ts_idx_weight_change_60"]
    pct_rank_by_date(out, "ts_idx_weight", "ts_idx_weight_rank")
    pct_rank_by_date(out, "ts_idx_weight_change_5", "ts_idx_weight_change_5_rank")
    pct_rank_by_date(out, "ts_idx_weight_change_20", "ts_idx_weight_change_20_rank")
    pct_rank_by_date(out, "ts_idx_weight_change_60", "ts_idx_weight_change_60_rank")
    pct_rank_by_date(out, "ts_idx_weight_delta_abs_20", "ts_idx_weight_delta_abs_20_rank")
    pct_rank_by_date(out, "ts_idx_weight_accel_20_60", "ts_idx_weight_accel_20_60_rank")
    zscore_by_date(out, "ts_idx_weight", "ts_idx_weight_z")
    out["ts_idx_weight_core_flag"] = (out["ts_idx_weight_rank"] >= 0.80).astype(float)
    out["ts_idx_weight_edge_flag"] = (out["ts_idx_weight_rank"] <= 0.20).astype(float)

    base_full = base_stock_frame()
    if not base_full.empty:
        base_full = base_full.copy()
        base_full["base_ret_20"] = base_full.groupby("股票代码")["收盘"].pct_change(20)
        ret = base_full[["股票代码", "日期", "base_ret_20"]].copy()
        out = out.merge(ret, on=["股票代码", "日期"], how="left")
        pct_rank_by_date(out, "base_ret_20", "ts_idx_base_ret20_rank")
        out["ts_idx_weight_change_20_x_ret20"] = (
            out["ts_idx_weight_change_20_rank"].fillna(0.5)
            * out["ts_idx_base_ret20_rank"].fillna(0.5)
        )
    enhanced_result = write_factor(out, output_dir, "index_weight_enhanced_factors.csv")
    lean_columns = [
        "股票代码",
        "日期",
        "weight_trade_date",
        "weight_month_end_date",
        "available_date",
        "weight_available_date",
        "ts_idx_weight",
        "ts_idx_weight_age_days",
        "ts_idx_visible_lag_days",
        "ts_idx_asof_valid",
        "ts_idx_weight_change_20",
        "ts_idx_weight_rank",
        "ts_idx_weight_change_20_rank",
    ]
    lean_columns = [col for col in lean_columns if col in out.columns]
    result = write_factor(out[lean_columns], output_dir, "index_weight_factors.csv")
    result.update(
        {
            "asof_policy": policy,
            "invalid_visible_rows_dropped": invalid_visible_rows,
            "max_visible_lag_days": int(out["ts_idx_visible_lag_days"].max()) if out["ts_idx_visible_lag_days"].notna().any() else None,
            "max_weight_age_days": int(out["ts_idx_weight_age_days"].max()) if out["ts_idx_weight_age_days"].notna().any() else None,
            "enhanced_file": enhanced_result.get("file"),
            "enhanced_status": enhanced_result.get("status"),
        }
    )
    return result


def next_base_date_map() -> dict[pd.Timestamp, pd.Timestamp]:
    base = base_stock_dates()
    dates = sorted(base["日期"].dropna().unique())
    return {pd.Timestamp(dates[i]): pd.Timestamp(dates[i + 1]) for i in range(len(dates) - 1)}


def shift_event_to_next_base_date(df: pd.DataFrame, next_dates: dict[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Event data is treated as visible on the next base trading day."""
    if df.empty or "日期" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["日期"] = out["日期"].map(next_dates)
    out = out.dropna(subset=["日期"]).copy()
    out["available_date"] = out["日期"]
    return out


def build_event(raw_root: Path, output_dir: Path) -> dict:
    top_list = normalize_trade_table(read_raw_folder(raw_root, "top_list"), "ts_code", "trade_date")
    top_inst = normalize_trade_table(read_raw_folder(raw_root, "top_inst"), "ts_code", "trade_date")
    stk_limit = normalize_trade_table(read_raw_folder(raw_root, "stk_limit"), "ts_code", "trade_date")
    if top_list.empty and top_inst.empty and stk_limit.empty:
        return write_factor(pd.DataFrame(), output_dir, "event_factors.csv")

    next_dates = next_base_date_map()
    frames = []
    if not top_list.empty:
        top_list = to_number(top_list, ["amount", "net_amount", "pct_chg"])
        event = top_list[["股票代码", "日期"]].copy()
        event["ts_event_top_list_flag"] = 1.0
        for col in ("amount", "net_amount", "pct_chg"):
            if col in top_list.columns:
                event[f"ts_event_top_list_{col}"] = top_list[col]
        event = shift_event_to_next_base_date(event, next_dates)
        if not event.empty:
            frames.append(event)
    if not top_inst.empty:
        top_inst = to_number(top_inst, ["buy", "sell", "net_buy", "amount"])
        inst = top_inst[["股票代码", "日期"]].copy()
        if "net_buy" in top_inst.columns:
            inst["ts_event_inst_net_buy"] = top_inst["net_buy"]
        elif {"buy", "sell"} <= set(top_inst.columns):
            inst["ts_event_inst_net_buy"] = top_inst["buy"] - top_inst["sell"]
        if "amount" in top_inst.columns:
            inst["ts_event_inst_amount"] = top_inst["amount"]
        inst = shift_event_to_next_base_date(inst, next_dates)
        if not inst.empty:
            frames.append(inst)

    if not stk_limit.empty:
        stk_limit = to_number(stk_limit, ["pre_close", "up_limit", "down_limit"])
        base = base_stock_frame()
        if not base.empty:
            base = base.sort_values(["股票代码", "日期"]).copy()
            base["base_prev_close"] = base.groupby("股票代码")["收盘"].shift(1)
            base["base_close_pct_vs_prev"] = (base["收盘"] / (base["base_prev_close"] + 1e-12) - 1.0) * 100.0
            base["base_high_pct_vs_prev"] = (base["最高"] / (base["base_prev_close"] + 1e-12) - 1.0) * 100.0
            base["base_low_pct_vs_prev"] = (base["最低"] / (base["base_prev_close"] + 1e-12) - 1.0) * 100.0
        price_cols = [
            "股票代码",
            "日期",
            "涨跌幅",
            "base_close_pct_vs_prev",
            "base_high_pct_vs_prev",
            "base_low_pct_vs_prev",
        ]
        price_cols = [col for col in price_cols if col in base.columns]
        limit = stk_limit[["股票代码", "日期", "pre_close", "up_limit", "down_limit"]].merge(
            base[price_cols],
            on=["股票代码", "日期"],
            how="left",
        )
        eps = 1e-12
        def numeric_or_nan(col: str) -> pd.Series:
            if col not in limit.columns:
                return pd.Series(np.nan, index=limit.index)
            return pd.to_numeric(limit[col], errors="coerce")

        pct_chg = numeric_or_nan("涨跌幅")
        close_pct = pct_chg.fillna(numeric_or_nan("base_close_pct_vs_prev"))
        high_pct = numeric_or_nan("base_high_pct_vs_prev")
        low_pct = numeric_or_nan("base_low_pct_vs_prev")
        up_limit = pd.to_numeric(limit["up_limit"], errors="coerce")
        down_limit = pd.to_numeric(limit["down_limit"], errors="coerce")
        pre_close = pd.to_numeric(limit["pre_close"], errors="coerce")

        up_limit_pct = (up_limit / (pre_close + eps) - 1.0) * 100.0
        down_limit_pct = (down_limit / (pre_close + eps) - 1.0) * 100.0
        limit["ts_event_limit_width"] = (up_limit_pct - down_limit_pct) / 100.0
        limit["ts_event_up_limit_gap_close"] = (up_limit_pct - close_pct) / 100.0
        limit["ts_event_up_limit_gap_high"] = (up_limit_pct - high_pct) / 100.0
        limit["ts_event_down_limit_gap_low"] = (low_pct - down_limit_pct) / 100.0
        limit["ts_event_limit_up_flag"] = (close_pct >= up_limit_pct - 0.05).astype(float)
        limit["ts_event_near_limit_up_flag"] = (
            (high_pct >= up_limit_pct - 1.5)
            | (close_pct >= up_limit_pct - 3.0)
            | (close_pct >= 7.0)
        ).astype(float)
        limit["ts_event_limit_down_flag"] = (close_pct <= down_limit_pct + 0.05).astype(float)

        limit = limit.sort_values(["股票代码", "日期"]).reset_index(drop=True)
        for window in (3, 5, 10):
            limit[f"ts_event_limit_up_count_{window}"] = (
                limit.groupby("股票代码")["ts_event_limit_up_flag"]
                .transform(lambda values: values.rolling(window, min_periods=1).sum())
            )
            limit[f"ts_event_near_limit_up_count_{window}"] = (
                limit.groupby("股票代码")["ts_event_near_limit_up_flag"]
                .transform(lambda values: values.rolling(window, min_periods=1).sum())
            )
            limit[f"ts_event_limit_down_count_{window}"] = (
                limit.groupby("股票代码")["ts_event_limit_down_flag"]
                .transform(lambda values: values.rolling(window, min_periods=1).sum())
            )

        keep_cols = [
            "股票代码",
            "日期",
            "ts_event_limit_width",
            "ts_event_up_limit_gap_close",
            "ts_event_up_limit_gap_high",
            "ts_event_down_limit_gap_low",
            "ts_event_limit_up_flag",
            "ts_event_near_limit_up_flag",
            "ts_event_limit_down_flag",
            "ts_event_limit_up_count_3",
            "ts_event_limit_up_count_5",
            "ts_event_limit_up_count_10",
            "ts_event_near_limit_up_count_3",
            "ts_event_near_limit_up_count_5",
            "ts_event_near_limit_up_count_10",
            "ts_event_limit_down_count_3",
            "ts_event_limit_down_count_5",
            "ts_event_limit_down_count_10",
        ]
        limit_event = shift_event_to_next_base_date(limit[keep_cols], next_dates)
        if not limit_event.empty:
            frames.append(limit_event)

    if not frames:
        return write_factor(pd.DataFrame(), output_dir, "event_factors.csv")

    combined = pd.concat(frames, ignore_index=True)
    value_cols = [col for col in combined.columns if col not in {"股票代码", "日期", "available_date"}]
    out = combined.groupby(["股票代码", "日期"], as_index=False)[value_cols].sum(min_count=1)
    out["available_date"] = out["日期"]
    for col in value_cols:
        if col != "ts_event_top_list_flag":
            lower_is_better = col in {
                "ts_event_up_limit_gap_close",
                "ts_event_up_limit_gap_high",
                "ts_event_down_limit_gap_low",
                "ts_event_limit_down_flag",
                "ts_event_limit_down_count_3",
                "ts_event_limit_down_count_5",
                "ts_event_limit_down_count_10",
            }
            pct_rank_by_date(out, col, f"{col}_rank", high_is_good=not lower_is_better)
    cooldown_specs = {
        "ts_event_limit_up_count_3": "ts_event_limit_up_cooldown_3_rank",
        "ts_event_limit_up_count_5": "ts_event_limit_up_cooldown_5_rank",
        "ts_event_limit_up_count_10": "ts_event_limit_up_cooldown_10_rank",
        "ts_event_near_limit_up_count_3": "ts_event_near_limit_cooldown_3_rank",
        "ts_event_near_limit_up_count_5": "ts_event_near_limit_cooldown_5_rank",
        "ts_event_near_limit_up_count_10": "ts_event_near_limit_cooldown_10_rank",
    }
    for source, target in cooldown_specs.items():
        pct_rank_by_date(out, source, target, high_is_good=False)
    return write_factor(out, output_dir, "event_factors.csv")


def _load_sw_daily(raw_root: Path) -> pd.DataFrame:
    raw = read_raw_folder(raw_root, "sw_daily")
    if raw.empty:
        raw = read_raw_folder(raw_root, "industry")
    if raw.empty or "ts_code" not in raw.columns or "trade_date" not in raw.columns:
        return pd.DataFrame()
    raw = raw.copy()
    raw["index_code"] = raw["ts_code"].astype(str)
    raw["日期"] = parse_yyyymmdd(raw["trade_date"])
    raw = raw.dropna(subset=["日期"]).copy()
    numeric = [
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
    raw = to_number(raw, numeric)
    return raw


def _load_industry_classify(raw_root: Path) -> pd.DataFrame:
    classify = read_raw_folder(raw_root, "industry_meta")
    if classify.empty or "index_code" not in classify.columns:
        return pd.DataFrame(columns=["index_code", "industry_name", "level"])
    keep = [col for col in ("index_code", "industry_name", "level", "src") if col in classify.columns]
    classify = classify[keep].drop_duplicates("index_code", keep="last").copy()
    return classify


def _load_industry_members(raw_root: Path) -> pd.DataFrame:
    members = read_raw_folder(raw_root, "industry_member")
    if members.empty or "ts_code" not in members.columns:
        return pd.DataFrame()
    members = members.copy()
    members["股票代码"] = stock_code_from_ts(members["ts_code"])
    for col in ("in_date", "out_date"):
        if col in members.columns:
            members[col] = parse_yyyymmdd(members[col])
        else:
            members[col] = pd.NaT
    members["in_date"] = members["in_date"].fillna(pd.Timestamp("1900-01-01"))
    members = members[members["股票代码"].str.len() == 6].copy()
    keep = [
        "股票代码",
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "in_date",
        "out_date",
        "is_new",
    ]
    keep = [col for col in keep if col in members.columns]
    return members[keep].drop_duplicates().sort_values(["股票代码", "in_date"])


def _stock_industry_asof(base: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if base.empty or members.empty:
        return pd.DataFrame()
    pieces = []
    for stock_code, base_part in base.groupby("股票代码", sort=False):
        member_part = members[members["股票代码"] == stock_code].sort_values("in_date")
        if member_part.empty:
            continue
        merged = pd.merge_asof(
            base_part.sort_values("日期"),
            member_part,
            left_on="日期",
            right_on="in_date",
            direction="backward",
        )
        if "股票代码_x" in merged.columns:
            merged["股票代码"] = merged["股票代码_x"]
            merged = merged.drop(columns=[col for col in ("股票代码_x", "股票代码_y") if col in merged.columns])
        valid = merged["out_date"].isna() | merged["out_date"].ge(merged["日期"])
        pieces.append(merged[valid].copy())
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def build_industry(raw_root: Path, output_dir: Path) -> dict:
    sw = _load_sw_daily(raw_root)
    members = _load_industry_members(raw_root)
    if sw.empty or members.empty:
        return write_factor(pd.DataFrame(), output_dir, "industry_factors.csv")

    classify = _load_industry_classify(raw_root)
    if not classify.empty:
        sw = sw.merge(classify, on="index_code", how="left")
    if "level" not in sw.columns:
        sw["level"] = ""
    sw["level"] = sw["level"].fillna("")

    sw = sw.sort_values(["index_code", "日期"]).reset_index(drop=True)
    grouped = sw.groupby("index_code", sort=False)
    if "pct_change" in sw.columns:
        sw["ts_ind_daily_ret"] = sw["pct_change"] / 100.0
    else:
        sw["ts_ind_daily_ret"] = grouped["close"].pct_change()
    for window in (3, 5, 10, 20):
        if "close" in sw.columns:
            sw[f"ts_ind_ret_{window}"] = grouped["close"].transform(lambda s, w=window: s / s.shift(w) - 1.0)
        else:
            sw[f"ts_ind_ret_{window}"] = grouped["ts_ind_daily_ret"].transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
    if "amount" in sw.columns:
        amount_roll5 = grouped["amount"].transform(lambda s: s.rolling(5, min_periods=3).mean())
        amount_roll20 = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=8).mean())
        sw["ts_ind_amount_ratio_5_20"] = amount_roll5 / (amount_roll20 + 1e-12) - 1.0
    if "vol" in sw.columns:
        vol_roll5 = grouped["vol"].transform(lambda s: s.rolling(5, min_periods=3).mean())
        vol_roll20 = grouped["vol"].transform(lambda s: s.rolling(20, min_periods=8).mean())
        sw["ts_ind_vol_ratio_5_20"] = vol_roll5 / (vol_roll20 + 1e-12) - 1.0

    industry_rank_cols = [
        "ts_ind_ret_3",
        "ts_ind_ret_5",
        "ts_ind_ret_10",
        "ts_ind_ret_20",
        "ts_ind_amount_ratio_5_20",
        "ts_ind_vol_ratio_5_20",
    ]
    for col in industry_rank_cols:
        if col in sw.columns:
            sw[f"{col}_rank"] = (
                sw.groupby(["日期", "level"])[col]
                .rank(pct=True, ascending=True, method="average")
                .fillna(0.5)
            )

    base = base_stock_frame()
    if base.empty:
        return write_factor(pd.DataFrame(), output_dir, "industry_factors.csv")
    base_cols = ["股票代码", "日期"]
    if "涨跌幅" in base.columns:
        base_cols.append("涨跌幅")
    stock_ind = _stock_industry_asof(base[base_cols], members)
    if stock_ind.empty:
        return write_factor(pd.DataFrame(), output_dir, "industry_factors.csv")
    stock_ind["ts_stock_daily_ret"] = pd.to_numeric(stock_ind.get("涨跌幅", np.nan), errors="coerce") / 100.0

    out = stock_ind[["股票代码", "日期"]].copy()
    out["available_date"] = out["日期"]

    for level in ("l1", "l2", "l3"):
        code_col = f"{level}_code"
        if code_col not in stock_ind.columns:
            continue
        level_upper = level.upper()
        daily_cols = ["index_code", "日期", "ts_ind_daily_ret"] + [
            f"{col}_rank" for col in industry_rank_cols if f"{col}_rank" in sw.columns
        ]
        level_daily = sw[sw["level"].eq(level_upper)][daily_cols].drop_duplicates(["index_code", "日期"], keep="last")
        rename_map = {
            "ts_ind_daily_ret": f"ts_ind_{level}_daily_ret",
        }
        for col in industry_rank_cols:
            rank_col = f"{col}_rank"
            if rank_col in level_daily.columns:
                rename_map[rank_col] = f"ts_ind_{level}_{col.replace('ts_ind_', '')}_rank"
        merged = stock_ind[["股票代码", "日期", code_col]].merge(
            level_daily.rename(columns=rename_map),
            left_on=[code_col, "日期"],
            right_on=["index_code", "日期"],
            how="left",
        )
        for col in [c for c in merged.columns if c.startswith(f"ts_ind_{level}_") and c != f"ts_ind_{level}_daily_ret"]:
            out[col] = merged[col].values

        if f"ts_ind_{level}_daily_ret" in merged.columns:
            excess = stock_ind[["股票代码", "日期", "ts_stock_daily_ret"]].copy()
            excess[f"ts_ind_{level}_excess_1"] = excess["ts_stock_daily_ret"] - merged[f"ts_ind_{level}_daily_ret"].values
            excess = excess.sort_values(["股票代码", "日期"]).reset_index(drop=True)
            for window in (3, 5, 10, 20):
                source = f"ts_ind_{level}_excess_{window}"
                excess[source] = (
                    excess.groupby("股票代码")[f"ts_ind_{level}_excess_1"]
                    .transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
                )
                pct_rank_by_date(excess, source, f"{source}_rank", high_is_good=True)
                out[f"{source}_rank"] = excess[f"{source}_rank"].values

        breadth_base = stock_ind[["股票代码", "日期", code_col, "ts_stock_daily_ret"]].dropna(subset=[code_col]).copy()
        breadth = (
            breadth_base.groupby(["日期", code_col], as_index=False)
            .agg(
                ts_ind_member_count=("股票代码", "nunique"),
                ts_ind_pos_ratio=("ts_stock_daily_ret", lambda s: float((s > 0).mean())),
                ts_ind_member_mean_ret=("ts_stock_daily_ret", "mean"),
            )
            .sort_values([code_col, "日期"])
        )
        for window in (3, 5):
            breadth[f"ts_ind_breadth_{window}"] = (
                breadth.groupby(code_col)["ts_ind_pos_ratio"]
                .transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
            )
            breadth[f"ts_ind_member_mean_ret_{window}"] = (
                breadth.groupby(code_col)["ts_ind_member_mean_ret"]
                .transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
            )
        for col in (
            "ts_ind_pos_ratio",
            "ts_ind_breadth_3",
            "ts_ind_breadth_5",
            "ts_ind_member_mean_ret_3",
            "ts_ind_member_mean_ret_5",
        ):
            pct_rank_by_date(breadth, col, f"{col}_rank", high_is_good=True)
        breadth_cols = [
            code_col,
            "日期",
            "ts_ind_member_count",
            "ts_ind_pos_ratio_rank",
            "ts_ind_breadth_3_rank",
            "ts_ind_breadth_5_rank",
            "ts_ind_member_mean_ret_3_rank",
            "ts_ind_member_mean_ret_5_rank",
        ]
        breadth_cols = [col for col in breadth_cols if col in breadth.columns]
        merged_breadth = stock_ind[["股票代码", "日期", code_col]].merge(
            breadth[breadth_cols],
            on=[code_col, "日期"],
            how="left",
        )
        for col in breadth_cols:
            if col in {code_col, "日期"}:
                continue
            out[f"ts_ind_{level}_{col.replace('ts_ind_', '')}"] = merged_breadth[col].values

    feature_cols = [col for col in out.columns if col not in {"股票代码", "日期", "available_date"}]
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.5 if col.endswith("_rank") else 0.0)

    for level in ("l1", "l2", "l3"):
        member_col = f"ts_ind_{level}_member_count"
        if member_col in out.columns:
            pct_rank_by_date(out, member_col, f"{member_col}_rank", high_is_good=True)
        for window in (3, 5, 10, 20):
            excess_rank = f"ts_ind_{level}_excess_{window}_rank"
            if excess_rank in out.columns:
                out[f"ts_ind_{level}_excess_{window}_cooldown_rank"] = 1.0 - out[excess_rank]
        for window in (3, 5):
            mean_rank = f"ts_ind_{level}_member_mean_ret_{window}_rank"
            if mean_rank in out.columns:
                out[f"ts_ind_{level}_member_mean_ret_{window}_cooldown_rank"] = 1.0 - out[mean_rank]

    core_cols = ["股票代码", "日期", "available_date"]
    for level in ("l1", "l2", "l3"):
        core_cols.extend(
            [
                f"ts_ind_{level}_member_count_rank",
                f"ts_ind_{level}_ret_3_rank",
                f"ts_ind_{level}_ret_5_rank",
                f"ts_ind_{level}_ret_10_rank",
                f"ts_ind_{level}_breadth_3_rank",
                f"ts_ind_{level}_breadth_5_rank",
                f"ts_ind_{level}_member_mean_ret_3_cooldown_rank",
                f"ts_ind_{level}_member_mean_ret_5_cooldown_rank",
                f"ts_ind_{level}_excess_3_cooldown_rank",
                f"ts_ind_{level}_excess_5_cooldown_rank",
                f"ts_ind_{level}_excess_10_cooldown_rank",
            ]
        )
    core_cols = [col for col in core_cols if col in out.columns]
    core_result = write_factor(out[core_cols], output_dir, "industry_core_factors.csv")
    result = write_factor(out, output_dir, "industry_factors.csv")
    result.update(
        {
            "core_file": core_result.get("file"),
            "core_status": core_result.get("status"),
            "core_rows": core_result.get("rows"),
            "core_feature_count": max(0, len(core_cols) - 3),
        }
    )
    return result


def build_cyq_perf(raw_root: Path, output_dir: Path) -> dict:
    raw = normalize_trade_table(read_raw_folder(raw_root, "cyq_perf"), "ts_code", "trade_date")
    if raw.empty:
        return write_factor(pd.DataFrame(), output_dir, "cyq_perf_factors.csv")

    numeric = [
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
    raw = to_number(raw, numeric)
    out = raw[["股票代码", "日期"]].copy()
    out["available_date"] = out["日期"]

    eps = 1e-12
    for col in numeric:
        if col in raw.columns:
            out[f"ts_cyq_{col}"] = raw[col]

    weight_avg = raw.get("weight_avg", pd.Series(np.nan, index=raw.index)).abs() + eps
    cost_5 = raw.get("cost_5pct", pd.Series(np.nan, index=raw.index))
    cost_15 = raw.get("cost_15pct", pd.Series(np.nan, index=raw.index))
    cost_50 = raw.get("cost_50pct", pd.Series(np.nan, index=raw.index))
    cost_85 = raw.get("cost_85pct", pd.Series(np.nan, index=raw.index))
    cost_95 = raw.get("cost_95pct", pd.Series(np.nan, index=raw.index))
    his_low = raw.get("his_low", pd.Series(np.nan, index=raw.index))
    his_high = raw.get("his_high", pd.Series(np.nan, index=raw.index))

    out["ts_cyq_cost_band_width"] = (cost_95 - cost_5) / weight_avg
    out["ts_cyq_cost_85_15_width"] = (cost_85 - cost_15) / weight_avg
    out["ts_cyq_upper_cost_spread"] = (cost_95 - cost_50) / weight_avg
    out["ts_cyq_lower_cost_spread"] = (cost_50 - cost_5) / weight_avg
    out["ts_cyq_cost_skew"] = (
        out["ts_cyq_upper_cost_spread"] - out["ts_cyq_lower_cost_spread"]
    ) / (out["ts_cyq_upper_cost_spread"].abs() + out["ts_cyq_lower_cost_spread"].abs() + eps)
    out["ts_cyq_weight_avg_history_position"] = (raw.get("weight_avg", np.nan) - his_low) / (his_high - his_low + eps)

    out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    grouped = out.groupby("股票代码", sort=False)
    for col in ("ts_cyq_winner_rate", "ts_cyq_cost_band_width", "ts_cyq_cost_skew"):
        if col in out.columns:
            for window in (3, 5, 10):
                out[f"{col}_chg_{window}"] = grouped[col].transform(lambda values, w=window: values - values.shift(w))

    rank_specs = {
        "ts_cyq_winner_rate": True,
        "ts_cyq_winner_rate_chg_3": True,
        "ts_cyq_winner_rate_chg_5": True,
        "ts_cyq_winner_rate_chg_10": True,
        "ts_cyq_cost_band_width": True,
        "ts_cyq_cost_band_width_chg_5": True,
        "ts_cyq_cost_85_15_width": True,
        "ts_cyq_upper_cost_spread": True,
        "ts_cyq_lower_cost_spread": True,
        "ts_cyq_cost_skew": True,
        "ts_cyq_cost_skew_chg_5": True,
        "ts_cyq_weight_avg_history_position": True,
    }
    for source, high_is_good in rank_specs.items():
        if source in out.columns:
            pct_rank_by_date(out, source, f"{source}_rank", high_is_good=high_is_good)
    if "ts_cyq_winner_rate" in out.columns:
        pct_rank_by_date(out, "ts_cyq_winner_rate", "ts_cyq_winner_rate_low_rank", high_is_good=False)
    if "ts_cyq_cost_band_width" in out.columns:
        pct_rank_by_date(out, "ts_cyq_cost_band_width", "ts_cyq_cost_concentration_rank", high_is_good=False)

    keep_cols = [
        "股票代码",
        "日期",
        "available_date",
        "ts_cyq_winner_rate",
        "ts_cyq_winner_rate_rank",
        "ts_cyq_winner_rate_low_rank",
        "ts_cyq_winner_rate_chg_3_rank",
        "ts_cyq_winner_rate_chg_5_rank",
        "ts_cyq_winner_rate_chg_10_rank",
        "ts_cyq_cost_band_width",
        "ts_cyq_cost_band_width_rank",
        "ts_cyq_cost_band_width_chg_5_rank",
        "ts_cyq_cost_concentration_rank",
        "ts_cyq_cost_85_15_width_rank",
        "ts_cyq_upper_cost_spread_rank",
        "ts_cyq_lower_cost_spread_rank",
        "ts_cyq_cost_skew_rank",
        "ts_cyq_cost_skew_chg_5_rank",
        "ts_cyq_weight_avg_history_position_rank",
    ]
    keep_cols = [col for col in keep_cols if col in out.columns]
    return write_factor(out[keep_cols], output_dir, "cyq_perf_factors.csv")


def build_financial(raw_root: Path, output_dir: Path) -> dict:
    raw = read_raw_folder(raw_root, "financial")
    if raw.empty or "ts_code" not in raw.columns:
        return write_factor(pd.DataFrame(), output_dir, "financial_factors.csv")

    raw["股票代码"] = stock_code_from_ts(raw["ts_code"])
    available_source = None
    for col in ("f_ann_date", "ann_date", "ann_date_x", "ann_date_y"):
        if col in raw.columns:
            available_source = raw[col] if available_source is None else available_source.fillna(raw[col])
    if available_source is None:
        return write_factor(pd.DataFrame(), output_dir, "financial_factors.csv")

    raw["available_date"] = parse_yyyymmdd(available_source)
    raw = raw.dropna(subset=["available_date"]).sort_values(["股票代码", "available_date"])
    numeric = [
        "roe",
        "roe_dt",
        "grossprofit_margin",
        "netprofit_margin",
        "or_yoy",
        "netprofit_yoy",
        "debt_to_assets",
        "ocf_to_profit",
    ]
    raw = to_number(raw, numeric)
    keep = ["股票代码", "available_date"] + [col for col in numeric if col in raw.columns]
    raw = raw[keep].drop_duplicates(["股票代码", "available_date"], keep="last")

    base = base_stock_dates()
    pieces = []
    for stock_code, base_part in base.groupby("股票代码", sort=False):
        fin_part = raw[raw["股票代码"] == stock_code].sort_values("available_date")
        if fin_part.empty:
            continue
        merged = pd.merge_asof(
            base_part.sort_values("日期"),
            fin_part,
            left_on="日期",
            right_on="available_date",
            by="股票代码",
            direction="backward",
        )
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if out.empty:
        return write_factor(out, output_dir, "financial_factors.csv")

    rename = {col: f"ts_fin_{col}" for col in numeric if col in out.columns}
    out = out.rename(columns=rename)
    for col in rename.values():
        pct_rank_by_date(out, col, f"{col}_rank")
    return write_factor(out, output_dir, "financial_factors.csv")


BUILDERS = {
    "daily_basic": build_daily_basic,
    "daily_basic_core": build_daily_basic_core,
    "moneyflow": build_moneyflow,
    "moneyflow_residual": build_moneyflow_residual,
    "index_weight": build_index_weight,
    "event": build_event,
    "industry": build_industry,
    "cyq_perf": build_cyq_perf,
    "financial": build_financial,
}


def main() -> None:
    global BASE_DATA_PATH, INDEX_WEIGHT_ASOF_POLICY
    parser = argparse.ArgumentParser(description="Build audited local factor files from raw CSV snapshots.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--base-data",
        default=str(BASE_DATA_PATH),
        help="Base stock data calendar/universe used for as-of joins. Defaults to the frozen training data.",
    )
    parser.add_argument("--families", nargs="*", default=list(BUILDERS), choices=list(BUILDERS))
    parser.add_argument("--manifest", default="factor_manifest.json")
    parser.add_argument(
        "--index-weight-asof-policy",
        default=INDEX_WEIGHT_ASOF_POLICY,
        choices=["trade_date", "next_trade_day", "month_end_next_trade_day"],
        help="Visibility policy for index_weight trade_date when true publish time is unknown.",
    )
    args = parser.parse_args()
    BASE_DATA_PATH = Path(args.base_data)
    INDEX_WEIGHT_ASOF_POLICY = args.index_weight_asof_policy

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    manifest = []
    for family in args.families:
        result = BUILDERS[family](raw_root, output_dir)
        result["family"] = family
        manifest.append(result)
        print(f"{family}: {result['status']} rows={result['rows']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(manifest).to_csv(output_dir / "factor_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
