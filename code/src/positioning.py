import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("开盘", "收盘", "最高", "最低")
BASE_TOP2_WEIGHTS = (0.60, 0.40)
BASE_TOP3_WEIGHTS = (0.40, 0.35, 0.25)
TOP2_BALANCED_WEIGHTS = (0.55, 0.45)
TOP2_CONVICTION_WEIGHTS = (0.65, 0.35)

INDUSTRY_CONFIRMATION_WEIGHTS = (
    ("ts_ind_l2_ret_5_rank", 0.16),
    ("ts_ind_l2_ret_10_rank", 0.10),
    ("ts_ind_l2_breadth_3_rank", 0.12),
    ("ts_ind_l2_breadth_5_rank", 0.14),
    ("ts_ind_l3_ret_5_rank", 0.12),
    ("ts_ind_l3_breadth_3_rank", 0.10),
    ("ts_ind_l3_breadth_5_rank", 0.10),
    ("ts_ind_l1_ret_5_rank", 0.08),
    ("ts_ind_l1_breadth_5_rank", 0.08),
)

INDUSTRY_COOLDOWN_WEIGHTS = (
    ("ts_ind_l2_member_mean_ret_5_cooldown_rank", 0.18),
    ("ts_ind_l2_excess_5_cooldown_rank", 0.18),
    ("ts_ind_l3_member_mean_ret_5_cooldown_rank", 0.16),
    ("ts_ind_l3_excess_5_cooldown_rank", 0.16),
    ("ts_ind_l1_member_mean_ret_5_cooldown_rank", 0.12),
    ("ts_ind_l1_excess_5_cooldown_rank", 0.12),
    ("ts_ind_l2_excess_10_cooldown_rank", 0.08),
)

MONEYFLOW_CONFIRMATION_WEIGHTS = (
    ("ts_mfr_price_confirmed_flow", 0.20),
    ("ts_mfr_mf_main_residual_3_rank", 0.12),
    ("ts_mfr_mf_main_residual_5_rank", 0.16),
    ("ts_mfr_mf_main_residual_10_rank", 0.13),
    ("ts_mfr_mf_net_residual_5_rank", 0.11),
    ("ts_mfr_mf_large_order_share_5_rank", 0.10),
    ("ts_mfr_main_residual_slope_5_20_rank", 0.10),
    ("ts_mfr_net_residual_slope_5_20_rank", 0.08),
)


def _normalize_market_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])
    for col in (*PRICE_COLUMNS, "成交量", "成交额", "换手率", "涨跌幅"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["股票代码", "日期"]).reset_index(drop=True)


def _cross_section_return(
    df: pd.DataFrame,
    dates: List[pd.Timestamp],
    current_idx: int,
    window: int,
) -> pd.Series:
    if current_idx - window < 0:
        return pd.Series(dtype=float)

    current_date = dates[current_idx]
    previous_date = dates[current_idx - window]
    current = df[df["日期"].eq(current_date)][["股票代码", "收盘"]].rename(
        columns={"收盘": "current_close"}
    )
    previous = df[df["日期"].eq(previous_date)][["股票代码", "收盘"]].rename(
        columns={"收盘": "previous_close"}
    )
    merged = current.merge(previous, on="股票代码", how="inner")
    merged = merged[merged["previous_close"] > 1e-12].copy()
    return (merged["current_close"] - merged["previous_close"]) / merged["previous_close"]


def compute_market_snapshot(
    raw_df: pd.DataFrame,
    asof_date=None,
    selected_stock_ids: List[str] | None = None,
) -> Dict[str, float]:
    """只使用已知行情，计算预测时点的市场状态和候选股热度。"""
    df = _normalize_market_df(raw_df)
    dates = sorted(df["日期"].dropna().unique())
    if not dates:
        raise ValueError("无法计算市场状态：数据为空")

    if asof_date is None:
        current_date = pd.Timestamp(dates[-1])
    else:
        current_date = pd.Timestamp(asof_date)
        eligible = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) <= current_date]
        if not eligible:
            raise ValueError(f"无法计算市场状态：{current_date.date()} 前没有行情")
        current_date = eligible[-1]

    current_idx = [pd.Timestamp(d) for d in dates].index(current_date)
    snapshot: Dict[str, float] = {"asof_date": current_date.date().isoformat()}

    for window in (3, 5, 10, 20):
        returns = _cross_section_return(df, dates, current_idx, window)
        if returns.empty:
            continue
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            continue
        snapshot[f"market_ret_{window}"] = float(returns.mean())
        snapshot[f"ret{window}_median"] = float(returns.median())
        snapshot[f"ret{window}_p90"] = float(returns.quantile(0.90))
        snapshot[f"ret{window}_p95"] = float(returns.quantile(0.95))
        snapshot[f"ret{window}_p95_minus_median"] = float(returns.quantile(0.95) - returns.median())
        snapshot[f"ret{window}_p90_minus_median"] = float(returns.quantile(0.90) - returns.median())
        snapshot[f"ret{window}_skew"] = float(returns.skew()) if len(returns) >= 5 else 0.0
        snapshot[f"up_ratio_{window}"] = float((returns > 0).mean())
        snapshot[f"strong_ratio_{window}"] = float((returns > 0.03).mean())
        snapshot[f"strong_stock_count_{window}"] = float((returns > 0.03).sum())
        snapshot[f"weak_ratio_{window}"] = float((returns < -0.03).mean())

    recent = df[df["日期"].isin(dates[max(0, current_idx - 25) : current_idx + 1])].copy()
    recent["daily_return"] = recent.groupby("股票代码")["收盘"].pct_change()
    daily_market = recent.groupby("日期")["daily_return"].mean().dropna()
    last20 = daily_market.tail(20)
    if not last20.empty:
        equity_curve = (1.0 + last20).cumprod()
        snapshot["daily_market_mean_5"] = float(daily_market.tail(5).mean())
        snapshot["daily_market_mean_20"] = float(last20.mean())
        snapshot["daily_market_vol_20"] = float(last20.std())
        snapshot["market_drawdown_20"] = float((equity_curve / equity_curve.cummax() - 1.0).min())

    if "成交额" in df.columns:
        amount_recent = df[df["日期"].isin(dates[max(0, current_idx - 25) : current_idx + 1])].copy()
        amount_recent["amount_ma20"] = amount_recent.groupby("股票代码")["成交额"].transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
        amount_latest = amount_recent[amount_recent["日期"].eq(current_date)].copy()
        amount_latest["amount_ratio20"] = amount_latest["成交额"] / (amount_latest["amount_ma20"] + 1e-12)
        amount_ratio = amount_latest["amount_ratio20"].replace([np.inf, -np.inf], np.nan).dropna()
        if not amount_ratio.empty:
            p50 = float(amount_ratio.quantile(0.50))
            snapshot["amount_ratio20_p90"] = float(amount_ratio.quantile(0.90))
            snapshot["amount_ratio20_p50"] = p50
            snapshot["amount_ratio20_p90_p50"] = float(amount_ratio.quantile(0.90) / (p50 + 1e-12))
            snapshot["amount_ratio20_p95_p50"] = float(amount_ratio.quantile(0.95) / (p50 + 1e-12))

    if selected_stock_ids:
        selected = [str(stock_id).zfill(6) for stock_id in selected_stock_ids]
        latest = df[df["日期"].eq(current_date)][["股票代码", "收盘"]]
        for window in (5, 20):
            if current_idx - window < 0:
                continue
            previous_date = dates[current_idx - window]
            previous = df[df["日期"].eq(previous_date)][["股票代码", "收盘"]].rename(
                columns={"收盘": "previous_close"}
            )
            merged = latest.merge(previous, on="股票代码", how="inner")
            merged = merged[merged["股票代码"].isin(selected)]
            merged = merged[merged["previous_close"] > 1e-12].copy()
            if not merged.empty:
                returns = (merged["收盘"] - merged["previous_close"]) / merged["previous_close"]
                snapshot[f"selected_ret_{window}_mean"] = float(returns.mean())
                snapshot[f"selected_ret_{window}_max"] = float(returns.max())

    return snapshot


def classify_market_regime(
    snapshot: Dict[str, float],
    score_profile: Dict[str, float] | None = None,
) -> str:
    """Fine-grained regime label used by TopK and factor-lane gates."""
    score_profile = score_profile or {}
    market_ret_5 = float(snapshot.get("market_ret_5", 0.0) or 0.0)
    market_ret_20 = float(snapshot.get("market_ret_20", 0.0) or 0.0)
    up_ratio_5 = float(snapshot.get("up_ratio_5", 0.5) or 0.5)
    up_ratio_20 = float(snapshot.get("up_ratio_20", 0.5) or 0.5)
    drawdown_20 = float(snapshot.get("market_drawdown_20", 0.0) or 0.0)
    ret5_dispersion = float(snapshot.get("ret5_p95_minus_median", 0.0) or 0.0)
    ret20_dispersion = float(snapshot.get("ret20_p95_minus_median", 0.0) or 0.0)
    ret5_skew = float(snapshot.get("ret5_skew", 0.0) or 0.0)
    amount_p90_p50 = float(snapshot.get("amount_ratio20_p90_p50", 1.0) or 1.0)
    selected_ret_5 = float(snapshot.get("selected_ret_5_mean", 0.0) or 0.0)
    selected_ret_20 = float(snapshot.get("selected_ret_20_mean", 0.0) or 0.0)
    top2_vs_top10_gap = float(score_profile.get("top2_vs_top10_gap", 0.0) or 0.0)
    top2_concentration = float(score_profile.get("top2_concentration", 0.0) or 0.0)

    weak_breadth = up_ratio_5 < 0.42 or up_ratio_20 < 0.45
    severe_weak_breadth = up_ratio_5 < 0.34 or up_ratio_20 < 0.36
    not_crash = market_ret_5 > -0.045 and drawdown_20 > -0.090
    high_dispersion = (
        ret5_dispersion > 0.070
        or ret20_dispersion > 0.140
        or ret5_skew > 0.60
        or amount_p90_p50 > 1.85
    )
    candidate_hot = (
        selected_ret_5 > 0.035
        or selected_ret_20 > 0.120
        or top2_vs_top10_gap > 0.018
        or top2_concentration > 0.460
    )

    if weak_breadth and not_crash and high_dispersion and candidate_hot:
        return "weak_local_spike"
    if market_ret_20 < -0.035 and up_ratio_20 < 0.38 and candidate_hot:
        return "deep_weak_hot"
    if market_ret_20 > 0.035 and up_ratio_20 > 0.56:
        return "broad_uptrend"
    if selected_ret_20 > 0.22 and weak_breadth:
        return "narrow_hot_theme"
    if market_ret_5 < -0.035 and drawdown_20 < -0.075 and severe_weak_breadth:
        return "crash_defensive"
    if market_ret_5 > 0.010 and market_ret_20 < 0.0 and up_ratio_5 > 0.50:
        return "fragile_rebound"
    return "balanced"


def choose_factor_lane(
    snapshot: Dict[str, float],
    score_profile: Dict[str, float] | None = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Guarded production lane for 5000-point factors.

    The no-factor baseline is the production trunk. index_weight is a guarded
    specialist that should only be used when the pre-window state is broad and
    index-friendly enough.
    """
    score_profile = score_profile or {}
    regime = classify_market_regime(snapshot, score_profile)
    market_ret_20 = float(snapshot.get("market_ret_20", 0.0) or 0.0)
    up_ratio_5 = float(snapshot.get("up_ratio_5", 0.5) or 0.5)
    selected_ret_20 = float(snapshot.get("selected_ret_20_mean", 0.0) or 0.0)
    top2_vs_top10_gap = float(score_profile.get("top2_vs_top10_gap", 0.0) or 0.0)

    index_friendly = (
        regime == "broad_uptrend"
        and market_ret_20 > 0.018
        and up_ratio_5 > 0.46
        and top2_vs_top10_gap > 0.010
    )
    a1_like_index_specialist = (
        market_ret_20 > 0.030
        and up_ratio_5 > 0.50
        and selected_ret_20 > 0.080
        and top2_vs_top10_gap > 0.014
    )

    lane = "no_factors"
    reason = "baseline_trunk_default"
    if index_friendly:
        lane = "index_weight"
        reason = "broad_uptrend_index_weight_specialist"
    elif a1_like_index_specialist and regime not in {"weak_local_spike", "deep_weak_hot"}:
        lane = "index_weight"
        reason = "a1_like_index_weight_specialist"
    elif regime == "weak_local_spike":
        reason = "weak_local_spike_baseline_trunk"

    out = dict(snapshot)
    out["market_regime_v2"] = regime
    out["factor_lane"] = lane
    out["factor_lane_reason"] = reason
    return lane, out


def _profile_value(profile: Dict[str, float] | None, key: str, default: float = 0.0) -> float:
    if not profile:
        return default
    value = profile.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _candidate_quality(score_profile: Dict[str, float] | None, prefix: str) -> float:
    """Quality proxy for the rank-N candidate using only current prediction features."""
    profile = score_profile or {}
    return float(
        0.24 * _profile_value(profile, f"{prefix}_relative_strength_20_rank", 0.5)
        + 0.20 * _profile_value(profile, f"{prefix}_hotspot_strength_rank", 0.5)
        + 0.18 * _profile_value(profile, f"{prefix}_controlled_hotspot", 0.5)
        + 0.16 * _profile_value(profile, f"{prefix}_selector_consensus", 0.5)
        + 0.12 * _profile_value(profile, f"{prefix}_amount_moderate_20_rank", 0.5)
        + 0.10 * _profile_value(profile, f"{prefix}_no_overheat_20_rank", 0.5)
    )


def _profile_heat(score_profile: Dict[str, float] | None, prefix: str) -> float:
    profile = score_profile or {}
    return float(
        0.35 * _profile_value(profile, f"{prefix}_relative_strength_20_rank", 0.5)
        + 0.25 * _profile_value(profile, f"{prefix}_hotspot_strength_rank", 0.5)
        + 0.20 * _profile_value(profile, f"{prefix}_controlled_hotspot", 0.5)
        + 0.20 * _profile_value(profile, f"{prefix}_ret_20_rank", 0.5)
    )


def _profile_overheat_risk(score_profile: Dict[str, float] | None, prefix: str) -> float:
    """Overheat proxy for lane selection; only uses current score-profile fields."""
    no_overheat = _profile_value(score_profile, f"{prefix}_no_overheat_20_rank", 0.5)
    amount_moderate = _profile_value(score_profile, f"{prefix}_amount_moderate_20_rank", 0.5)
    return float(
        0.44 * (1.0 - no_overheat)
        + 0.34 * (1.0 - amount_moderate)
        + 0.22 * max(0.0, _profile_heat(score_profile, prefix) - 0.78) / 0.22
    )


def _row_value(row: pd.Series | Dict[str, float], key: str, default: float = 0.5) -> float:
    try:
        value = row.get(key, default)
    except AttributeError:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _row_has_numeric(row: pd.Series | Dict[str, float], key: str) -> bool:
    try:
        value = row.get(key, np.nan)
    except AttributeError:
        value = row[key] if key in row else np.nan
    if value is None or pd.isna(value):
        return False
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_weighted_metric(
    row: pd.Series | Dict[str, float],
    specs: Tuple[Tuple[str, float], ...],
    default: float = 0.5,
) -> float:
    total = 0.0
    weight_sum = 0.0
    for col, weight in specs:
        if hasattr(row, "index") and col not in row.index:
            continue
        value = _row_value(row, col, np.nan)
        if not np.isfinite(value):
            continue
        total += value * weight
        weight_sum += weight
    if weight_sum <= 1e-12:
        return default
    return float(total / weight_sum)


def _row_candidate_quality(row: pd.Series | Dict[str, float]) -> float:
    return float(
        0.24 * _row_value(row, "relative_strength_20_rank", 0.5)
        + 0.20 * _row_value(row, "hotspot_strength_rank", 0.5)
        + 0.18 * _row_value(row, "controlled_hotspot", 0.5)
        + 0.16 * _row_value(row, "selector_consensus", 0.5)
        + 0.12 * _row_value(row, "amount_moderate_20_rank", 0.5)
        + 0.10 * _row_value(row, "no_overheat_20_rank", 0.5)
    )


def _row_industry_confirmation(row: pd.Series | Dict[str, float]) -> float:
    return _row_weighted_metric(row, INDUSTRY_CONFIRMATION_WEIGHTS, default=0.5)


def _row_industry_cooldown(row: pd.Series | Dict[str, float]) -> float:
    return _row_weighted_metric(row, INDUSTRY_COOLDOWN_WEIGHTS, default=0.5)


def _row_moneyflow_confirmation(row: pd.Series | Dict[str, float]) -> float:
    return _row_weighted_metric(row, MONEYFLOW_CONFIRMATION_WEIGHTS, default=0.5)


def _row_heat(row: pd.Series | Dict[str, float]) -> float:
    return float(
        0.35 * _row_value(row, "relative_strength_20_rank", 0.5)
        + 0.25 * _row_value(row, "hotspot_strength_rank", 0.5)
        + 0.20 * _row_value(row, "controlled_hotspot", 0.5)
        + 0.20 * _row_value(row, "ret_20_rank", 0.5)
    )


def _row_overheat_risk(row: pd.Series | Dict[str, float]) -> float:
    no_overheat = _row_value(row, "no_overheat_20_rank", 0.5)
    amount_moderate = _row_value(row, "amount_moderate_20_rank", 0.5)
    cooldown = _row_industry_cooldown(row)
    return float(
        0.38 * (1.0 - no_overheat)
        + 0.26 * (1.0 - amount_moderate)
        + 0.22 * (1.0 - cooldown)
        + 0.14 * max(0.0, _row_heat(row) - 0.78) / 0.22
    )


def _row_fresh_continuation(row: pd.Series | Dict[str, float]) -> float:
    ret5 = _row_value(row, "ret_5_rank", 0.5)
    ret10 = _row_value(row, "ret_10_rank", 0.5)
    ret20 = _row_value(row, "ret_20_rank", 0.5)
    not_matured = float(np.clip((ret10 - ret20 + 0.08) / 0.18, 0.0, 1.0))
    short_still_hot = float(np.clip((ret5 - 0.76) / 0.18, 0.0, 1.0))
    mid_hot = float(np.clip((ret10 - 0.88) / 0.10, 0.0, 1.0))
    controlled_long = float(np.clip((0.98 - ret20) / 0.16, 0.0, 1.0))
    return float(0.34 * not_matured + 0.24 * short_still_hot + 0.24 * mid_hot + 0.18 * controlled_long)


def _row_micro_industry_leader_score(row: pd.Series | Dict[str, float]) -> float:
    """
    Tiny L2 industry breakout shape: fresh industry leadership with stock-level
    5/10-day strength, but not yet a fully exhausted 20-day tail.
    """
    ret5 = _row_value(row, "ret_5_rank", 0.5)
    ret10 = _row_value(row, "ret_10_rank", 0.5)
    ret20 = _row_value(row, "ret_20_rank", 0.5)
    amount_rank = _row_value(row, "amount_ratio_20_rank", 0.5)
    amount_moderate = _row_value(row, "amount_moderate_20_rank", 0.5)
    near_high = _row_value(row, "near_high_20_rank", 0.5)
    up20 = _row_value(row, "up_day_ratio_20_rank", 0.5)
    l2_ret3 = _row_value(row, "ts_ind_l2_ret_3_rank", 0.5)
    l2_ret5 = _row_value(row, "ts_ind_l2_ret_5_rank", 0.5)
    l2_breadth3 = _row_value(row, "ts_ind_l2_breadth_3_rank", 0.5)
    l2_breadth5 = _row_value(row, "ts_ind_l2_breadth_5_rank", 0.5)
    l2_members = _row_value(row, "ts_ind_l2_member_count_rank", 0.5)
    l3_ret3 = _row_value(row, "ts_ind_l3_ret_3_rank", 0.5)
    l3_breadth3 = _row_value(row, "ts_ind_l3_breadth_3_rank", 0.5)
    moneyflow = _row_moneyflow_confirmation(row)
    score = (
        0.17 * l2_ret3
        + 0.15 * l2_ret5
        + 0.10 * l2_breadth3
        + 0.10 * l2_breadth5
        + 0.12 * ret10
        + 0.09 * ret5
        + 0.09 * (1.0 - l2_members)
        + 0.05 * l3_ret3
        + 0.03 * l3_breadth3
        + 0.04 * amount_moderate
        + 0.06 * moneyflow
        - 0.10 * np.clip((near_high - 0.94) / 0.06, 0.0, 1.0)
        - 0.08 * np.clip((amount_rank - 0.90) / 0.10, 0.0, 1.0)
        - 0.07 * np.clip((ret20 - 0.985) / 0.015, 0.0, 1.0)
        - 0.06 * np.clip((up20 - 0.970) / 0.030, 0.0, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def _row_micro_industry_leader(row: pd.Series | Dict[str, float]) -> bool:
    required = (
        "amount_ratio_20_rank",
        "near_high_20_rank",
        "up_day_ratio_20_rank",
        "ts_ind_l2_ret_3_rank",
        "ts_ind_l2_ret_5_rank",
        "ts_ind_l2_breadth_5_rank",
        "ts_ind_l2_member_count_rank",
    )
    if not all(_row_has_numeric(row, col) for col in required):
        return False
    return bool(
        _row_micro_industry_leader_score(row) >= 0.86
        and _row_value(row, "ts_ind_l2_member_count_rank", 0.5) <= 0.18
        and _row_value(row, "ts_ind_l2_ret_3_rank", 0.5) >= 0.95
        and _row_value(row, "ts_ind_l2_ret_5_rank", 0.5) >= 0.95
        and _row_value(row, "ts_ind_l2_breadth_5_rank", 0.5) >= 0.92
        and _row_value(row, "ret_5_rank", 0.5) >= 0.94
        and _row_value(row, "ret_10_rank", 0.5) >= 0.94
        and _row_value(row, "ret_20_rank", 0.5) <= 0.985
        and _row_value(row, "near_high_20_rank", 0.5) <= 0.93
        and _row_value(row, "amount_ratio_20_rank", 0.5) <= 0.90
        and _row_value(row, "up_day_ratio_20_rank", 0.5) <= 0.97
    )


def _resolve_score_col(df: pd.DataFrame, preferred: str) -> str:
    for col in (preferred, "final_score", "selector_score", "tabular_score"):
        if col in df.columns:
            return col
    return preferred


def _prepare_challenger_lookup(
    challenger_df: pd.DataFrame | None,
    score_col: str,
) -> Tuple[pd.DataFrame | None, Dict[str, pd.Series]]:
    if challenger_df is None or challenger_df.empty:
        return None, {}
    code_col = "stock_id" if "stock_id" in challenger_df.columns else "股票代码"
    if code_col not in challenger_df.columns:
        return None, {}
    local = challenger_df.copy()
    local[code_col] = local[code_col].astype(str).str.zfill(6)
    local_score_col = _resolve_score_col(local, score_col)
    if local_score_col in local.columns:
        local = local.sort_values(local_score_col, ascending=False)
    local = local.reset_index(drop=True)
    local["_challenger_rank"] = np.arange(1, len(local) + 1)
    local["_challenger_rank_score"] = 1.0 - ((local["_challenger_rank"] - 1).clip(lower=0) / 4.0).clip(0.0, 1.0)
    return local, {str(row[code_col]).zfill(6): row for _, row in local.iterrows()}


def _merge_challenger_evidence(
    row: pd.Series,
    challenger_lookup: Dict[str, pd.Series],
    code_col: str,
) -> pd.Series:
    code = str(row[code_col]).zfill(6)
    challenger = challenger_lookup.get(code)
    if challenger is None:
        return row.copy()
    merged = row.copy()
    for col, value in challenger.items():
        if col in {"stock_id", "股票代码", "日期"}:
            continue
        if col not in merged.index or pd.isna(merged.get(col)):
            merged[col] = value
    return merged


def compute_lane_overlap_features(
    index_pred: pd.DataFrame,
    nofactor_pred: pd.DataFrame,
    *,
    score_col: str = "final_score",
    code_col: str = "股票代码",
) -> Dict[str, float]:
    """Compare two lane rankings without looking at future returns."""
    def top_codes(df: pd.DataFrame, n: int) -> List[str]:
        local_score = score_col if score_col in df.columns else "selector_score"
        if local_score not in df.columns:
            local_score = "tabular_score"
        codes = (
            df.sort_values(local_score, ascending=False)[code_col]
            .astype(str)
            .str.zfill(6)
            .head(n)
            .tolist()
        )
        return codes

    index_top2 = top_codes(index_pred, 2)
    nofactor_top2 = top_codes(nofactor_pred, 2)
    index_top5 = top_codes(index_pred, 5)
    nofactor_top5 = top_codes(nofactor_pred, 5)
    return {
        "lane_top2_overlap": float(len(set(index_top2) & set(nofactor_top2))),
        "lane_top5_overlap": float(len(set(index_top5) & set(nofactor_top5))),
        "lane_index_top2": ",".join(index_top2),
        "lane_nofactor_top2": ",".join(nofactor_top2),
        "lane_index_top5": ",".join(index_top5),
        "lane_nofactor_top5": ",".join(nofactor_top5),
    }


def choose_factor_lane_v2(
    index_snapshot: Dict[str, float],
    index_profile: Dict[str, float] | None,
    nofactor_snapshot: Dict[str, float],
    nofactor_profile: Dict[str, float] | None,
    lane_overlap: Dict[str, float] | None = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Live dual-lane selector.

    no_factors is the trunk. index_weight can overrule it only when pre-window
    market state and head structure show an A1-like broad/index-friendly setup.
    """
    lane_overlap = lane_overlap or {}
    index_profile = index_profile or {}
    nofactor_profile = nofactor_profile or {}

    index_regime = classify_market_regime(index_snapshot, index_profile)
    nofactor_regime = classify_market_regime(nofactor_snapshot, nofactor_profile)
    index_market_ret_20 = float(index_snapshot.get("market_ret_20", 0.0) or 0.0)
    index_up_ratio_20 = float(index_snapshot.get("up_ratio_20", 0.5) or 0.5)
    index_up_ratio_5 = float(index_snapshot.get("up_ratio_5", 0.5) or 0.5)
    index_selected_ret_20 = float(index_snapshot.get("selected_ret_20_mean", 0.0) or 0.0)
    nofactor_market_ret_20 = float(nofactor_snapshot.get("market_ret_20", 0.0) or 0.0)
    nofactor_up_ratio_20 = float(nofactor_snapshot.get("up_ratio_20", 0.5) or 0.5)

    index_gap = _profile_value(index_profile, "top2_vs_top10_gap")
    nofactor_gap = _profile_value(nofactor_profile, "top2_vs_top10_gap")
    index_conc = _profile_value(index_profile, "top2_concentration")
    nofactor_conc = _profile_value(nofactor_profile, "top2_concentration")
    index_top2_quality = 0.5 * (
        _candidate_quality(index_profile, "top1") + _candidate_quality(index_profile, "top2")
    )
    nofactor_top2_quality = 0.5 * (
        _candidate_quality(nofactor_profile, "top1") + _candidate_quality(nofactor_profile, "top2")
    )
    top2_overlap = float(lane_overlap.get("lane_top2_overlap", 0.0) or 0.0)

    broad_index_setup = (
        index_regime == "broad_uptrend"
        and index_market_ret_20 > 0.018
        and index_up_ratio_20 > 0.52
        and index_up_ratio_5 > 0.46
    )
    a1_like_index_setup = (
        index_market_ret_20 > 0.030
        and index_up_ratio_20 > 0.50
        and index_selected_ret_20 > 0.080
        and index_gap > 0.012
    )
    index_head_advantage = (
        index_gap > max(0.010, nofactor_gap - 0.002)
        and index_conc >= nofactor_conc - 0.030
        and index_top2_quality >= max(0.52, nofactor_top2_quality - 0.020)
    )
    weak_spike_guard = (
        index_regime in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        or nofactor_regime in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
    )

    lane = "no_factors"
    reason = "baseline_trunk_default"
    if (broad_index_setup or a1_like_index_setup) and index_head_advantage and not weak_spike_guard:
        lane = "index_weight"
        reason = "a1_like_broad_index_weight_specialist"
    elif broad_index_setup and index_gap > nofactor_gap + 0.006 and index_top2_quality > 0.56:
        lane = "index_weight"
        reason = "broad_uptrend_index_gap_advantage"
    elif top2_overlap >= 2 and weak_spike_guard:
        reason = "lanes_agree_but_weak_spike_keep_baseline"
    elif weak_spike_guard:
        reason = "weak_spike_baseline_trunk"

    chosen_snapshot = dict(index_snapshot if lane == "index_weight" else nofactor_snapshot)
    chosen_snapshot.update(lane_overlap)
    chosen_snapshot.update(
        {
            "factor_lane": lane,
            "factor_lane_reason": reason,
            "market_regime_v2": index_regime if lane == "index_weight" else nofactor_regime,
            "index_market_regime_v2": index_regime,
            "nofactor_market_regime_v2": nofactor_regime,
            "index_top2_vs_top10_gap": index_gap,
            "nofactor_top2_vs_top10_gap": nofactor_gap,
            "index_top2_quality": index_top2_quality,
            "nofactor_top2_quality": nofactor_top2_quality,
            "index_market_ret_20": index_market_ret_20,
            "nofactor_market_ret_20": nofactor_market_ret_20,
            "index_up_ratio_20": index_up_ratio_20,
            "nofactor_up_ratio_20": nofactor_up_ratio_20,
        }
    )
    return lane, chosen_snapshot


def _first_code(text: str | None) -> str:
    if not text:
        return ""
    return str(text).split(",")[0].strip().zfill(6)


def _code_list(text: str | None) -> List[str]:
    if not text:
        return []
    return [part.strip().zfill(6) for part in str(text).split(",") if part.strip()]


def choose_factor_lane_v3(
    index_snapshot: Dict[str, float],
    index_profile: Dict[str, float] | None,
    nofactor_snapshot: Dict[str, float],
    nofactor_profile: Dict[str, float] | None,
    industry_snapshot: Dict[str, float] | None = None,
    industry_profile: Dict[str, float] | None = None,
    lane_overlap: Dict[str, float] | None = None,
    industry_overlap: Dict[str, float] | None = None,
    industry_lane_name: str = "index_industry_core",
) -> Tuple[str, Dict[str, float]]:
    """
    Conservative three-lane selector.

    index_industry_core is a weak-local-spike challenger. It is not allowed to
    replace the A1-like index_weight specialist by default and is not used when
    the base lanes already agree on the same Top2. A narrow broad-uptrend
    reorder guard lets industry_core overrule index_weight only when both lanes
    are looking at the same cluster but industry_core clearly changes the head.
    """
    base_lane, base_snapshot = choose_factor_lane_v2(
        index_snapshot,
        index_profile,
        nofactor_snapshot,
        nofactor_profile,
        lane_overlap,
    )
    if industry_snapshot is None or industry_profile is None:
        return base_lane, base_snapshot

    lane_overlap = lane_overlap or {}
    industry_overlap = industry_overlap or {}
    industry_profile = industry_profile or {}
    nofactor_profile = nofactor_profile or {}
    index_profile = index_profile or {}

    industry_regime = classify_market_regime(industry_snapshot, industry_profile)
    industry_gap = _profile_value(industry_profile, "top2_vs_top10_gap")
    nofactor_gap = _profile_value(nofactor_profile, "top2_vs_top10_gap")
    index_gap = _profile_value(index_profile, "top2_vs_top10_gap")
    industry_quality = 0.5 * (
        _candidate_quality(industry_profile, "top1") + _candidate_quality(industry_profile, "top2")
    )
    nofactor_quality = 0.5 * (
        _candidate_quality(nofactor_profile, "top1") + _candidate_quality(nofactor_profile, "top2")
    )
    index_quality = 0.5 * (
        _candidate_quality(index_profile, "top1") + _candidate_quality(index_profile, "top2")
    )
    base_top2_overlap = float(lane_overlap.get("lane_top2_overlap", 0.0) or 0.0)
    industry_top2_overlap = float(industry_overlap.get("lane_top2_overlap", 0.0) or 0.0)
    industry_top5_overlap = float(industry_overlap.get("lane_top5_overlap", 0.0) or 0.0)
    industry_top1 = _first_code(industry_overlap.get("lane_index_top2", ""))
    nofactor_top1 = _first_code(industry_overlap.get("lane_nofactor_top2", ""))
    top1_agree = industry_top1 and industry_top1 == nofactor_top1
    index_top5 = _code_list(lane_overlap.get("lane_index_top5", ""))
    industry_top5 = _code_list(industry_overlap.get("lane_index_top5", ""))
    index_industry_top5_overlap = float(len(set(index_top5[:5]) & set(industry_top5[:5])))
    index_top1 = index_top5[0] if index_top5 else ""
    industry_reorders_index_head = bool(index_top1 and industry_top1 and index_top1 != industry_top1)
    nofactor_top1_overheat = _profile_overheat_risk(nofactor_profile, "top1")
    nofactor_top2_overheat = _profile_overheat_risk(nofactor_profile, "top2")
    nofactor_top3_overheat = _profile_overheat_risk(nofactor_profile, "top3")
    nofactor_top1_amount_moderate = _profile_value(
        nofactor_profile,
        "top1_amount_moderate_20_rank",
        0.5,
    )
    industry_top1_overheat = _profile_overheat_risk(industry_profile, "top1")
    industry_top2_overheat = _profile_overheat_risk(industry_profile, "top2")

    weak_spike_challenger = (
        base_lane == "no_factors"
        and base_snapshot.get("factor_lane_reason") == "weak_spike_baseline_trunk"
        and base_snapshot.get("market_regime_v2") in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and industry_regime in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and base_top2_overlap <= 1.0
        and industry_top2_overlap < 2.0
        and industry_top5_overlap >= 2.0
        and top1_agree
        and industry_quality >= max(0.70, nofactor_quality - 0.08)
        and industry_gap >= nofactor_gap - 0.015
    )
    weak_spike_reorder = (
        base_lane == "no_factors"
        and base_snapshot.get("factor_lane_reason")
        in {"weak_spike_baseline_trunk", "lanes_agree_but_weak_spike_keep_baseline"}
        and base_snapshot.get("market_regime_v2") in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and industry_regime in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and base_top2_overlap >= 2.0
        and 1.0 <= industry_top2_overlap < 2.0
        and industry_top5_overlap >= 3.0
        and not top1_agree
        and industry_quality >= max(0.74, nofactor_quality - 0.04)
        and industry_gap >= nofactor_gap - 0.020
    )
    weak_spike_hot_head_reorder = (
        base_lane == "no_factors"
        and base_snapshot.get("factor_lane_reason")
        in {"weak_spike_baseline_trunk", "lanes_agree_but_weak_spike_keep_baseline"}
        and base_snapshot.get("market_regime_v2") in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and industry_regime in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and base_top2_overlap <= 1.0
        and industry_top5_overlap >= 3.0
        and index_industry_top5_overlap >= 3.0
        and industry_top1
        and nofactor_top1
        and index_top1
        and industry_top1 != nofactor_top1
        and industry_top1 != index_top1
        and industry_quality >= max(0.86, nofactor_quality + 0.035)
        and nofactor_top1_overheat >= 0.72
        and nofactor_top1_amount_moderate <= 0.12
    )
    weak_spike_cross_lane_overheat_reorder = (
        base_lane == "no_factors"
        and base_snapshot.get("factor_lane_reason")
        in {"weak_spike_baseline_trunk", "lanes_agree_but_weak_spike_keep_baseline"}
        and base_snapshot.get("market_regime_v2") in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and industry_regime in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and base_top2_overlap <= 1.0
        and index_top1
        and industry_top1
        and nofactor_top1
        and index_top1 == industry_top1
        and industry_top1 != nofactor_top1
        and industry_top5_overlap >= 4.0
        and index_industry_top5_overlap >= 4.0
        and nofactor_top2_overheat >= 0.80
        and nofactor_top3_overheat >= 0.80
        and industry_quality >= nofactor_quality - 0.060
    )
    weak_spike_dehot_industry_bailout = (
        base_lane == "no_factors"
        and base_snapshot.get("factor_lane_reason")
        in {"weak_spike_baseline_trunk", "lanes_agree_but_weak_spike_keep_baseline"}
        and base_snapshot.get("market_regime_v2") in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and industry_regime in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme"}
        and base_top2_overlap <= 1.0
        and nofactor_quality < 0.56
        and industry_quality >= nofactor_quality - 0.020
        and industry_quality <= 0.62
        and industry_top5_overlap >= 2.0
        and index_industry_top5_overlap <= 1.0
        and industry_top1
        and nofactor_top1
        and industry_top1 != nofactor_top1
        and industry_top1_overheat <= 0.50
        and industry_top2_overheat <= 0.50
    )
    broad_index_reorder = (
        base_lane == "index_weight"
        and base_snapshot.get("factor_lane_reason")
        in {"a1_like_broad_index_weight_specialist", "broad_uptrend_index_gap_advantage"}
        and base_snapshot.get("market_regime_v2") == "broad_uptrend"
        and index_industry_top5_overlap >= 2.0
        and industry_reorders_index_head
        and industry_quality >= max(0.72, index_quality - 0.08)
        and industry_gap >= index_gap - 0.020
    )

    if not (
        weak_spike_challenger
        or weak_spike_reorder
        or weak_spike_hot_head_reorder
        or weak_spike_cross_lane_overheat_reorder
        or weak_spike_dehot_industry_bailout
        or broad_index_reorder
    ):
        base_snapshot.update(
            {
                "industry_market_regime_v2": industry_regime,
                "industry_challenger_action": "rejected",
                "industry_challenger_reason": "guard_not_satisfied",
                "industry_nofactor_top2_overlap": industry_top2_overlap,
                "industry_nofactor_top5_overlap": industry_top5_overlap,
                "industry_index_top5_overlap": index_industry_top5_overlap,
                "industry_top2_quality": industry_quality,
                "index_top2_quality": index_quality,
                "nofactor_top1_overheat_risk": nofactor_top1_overheat,
                "nofactor_top2_overheat_risk": nofactor_top2_overheat,
                "nofactor_top3_overheat_risk": nofactor_top3_overheat,
                "industry_top1_overheat_risk": industry_top1_overheat,
                "industry_top2_overheat_risk": industry_top2_overheat,
            }
        )
        return base_lane, base_snapshot

    chosen_snapshot = dict(industry_snapshot)
    chosen_snapshot.update(lane_overlap)
    chosen_snapshot.update(
        {
            "factor_lane": industry_lane_name,
            "factor_lane_reason": (
                "broad_uptrend_index_industry_core_reorder"
                if broad_index_reorder
                else (
                    "weak_spike_hot_head_industry_core_reorder"
                    if weak_spike_hot_head_reorder
                    else (
                        "weak_spike_cross_lane_overheat_reorder"
                        if weak_spike_cross_lane_overheat_reorder
                        else (
                            "weak_spike_dehot_industry_core_bailout"
                            if weak_spike_dehot_industry_bailout
                            else (
                                "weak_spike_index_industry_core_reorder"
                                if weak_spike_reorder
                                else "weak_spike_index_industry_core_challenger"
                            )
                        )
                    )
                )
            ),
            "market_regime_v2": industry_regime,
            "index_market_regime_v2": base_snapshot.get("index_market_regime_v2", ""),
            "nofactor_market_regime_v2": base_snapshot.get("nofactor_market_regime_v2", ""),
            "industry_market_regime_v2": industry_regime,
            "industry_challenger_action": "accepted",
            "industry_challenger_reason": (
                "broad_index_reorder_guard"
                if broad_index_reorder
                else (
                    "weak_spike_hot_head_guard"
                    if weak_spike_hot_head_reorder
                    else (
                        "weak_spike_cross_lane_overheat_guard"
                        if weak_spike_cross_lane_overheat_reorder
                        else (
                            "weak_spike_dehot_industry_guard"
                            if weak_spike_dehot_industry_bailout
                            else (
                                "weak_spike_reorder_agreed_base"
                                if weak_spike_reorder
                                else "guard_satisfied"
                            )
                        )
                    )
                )
            ),
            "industry_nofactor_top2_overlap": industry_top2_overlap,
            "industry_nofactor_top5_overlap": industry_top5_overlap,
            "industry_index_top5_overlap": index_industry_top5_overlap,
            "industry_top2_quality": industry_quality,
            "nofactor_top2_quality": nofactor_quality,
            "index_top2_quality": index_quality,
            "industry_top2_vs_top10_gap": industry_gap,
            "nofactor_top2_vs_top10_gap": nofactor_gap,
            "index_top2_vs_top10_gap": index_gap,
            "nofactor_top1_overheat_risk": nofactor_top1_overheat,
            "nofactor_top2_overheat_risk": nofactor_top2_overheat,
            "nofactor_top3_overheat_risk": nofactor_top3_overheat,
            "industry_top1_overheat_risk": industry_top1_overheat,
            "industry_top2_overheat_risk": industry_top2_overheat,
        }
    )
    return industry_lane_name, chosen_snapshot


def compute_score_profile(score_df: pd.DataFrame, score_col: str = "final_score") -> Dict[str, float]:
    """用预测分数刻画头部信号强弱，只看当期排序结果，不读取未来数据。"""
    if score_col not in score_df.columns or score_df.empty:
        return {
            "top1_score": 0.0,
            "top2_mean_score": 0.0,
            "top3_mean_score": 0.0,
            "top1_top2_gap": 0.0,
            "top2_top3_gap": 0.0,
            "top1_top3_gap": 0.0,
            "top2_vs_top10_gap": 0.0,
            "top3_vs_top10_gap": 0.0,
            "top2_concentration": 0.0,
            "top3_concentration": 0.0,
        }

    scores = score_df[score_col].astype(float).to_numpy()
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return {
            "top1_score": 0.0,
            "top2_mean_score": 0.0,
            "top3_mean_score": 0.0,
            "top1_top2_gap": 0.0,
            "top2_top3_gap": 0.0,
            "top1_top3_gap": 0.0,
            "top2_vs_top10_gap": 0.0,
            "top3_vs_top10_gap": 0.0,
            "top2_concentration": 0.0,
            "top3_concentration": 0.0,
        }

    top2 = scores[: min(2, len(scores))]
    top3 = scores[: min(3, len(scores))]
    top10 = scores[: min(10, len(scores))]
    top2_mean = float(np.mean(top2))
    top3_mean = float(np.mean(top3))
    top10_tail_mean = float(np.mean(top10[3:])) if len(top10) > 3 else top3_mean
    top10_after_top2_mean = float(np.mean(top10[2:])) if len(top10) > 2 else top2_mean

    shifted = top10 - np.min(top10)
    top2_concentration = float(np.sum(shifted[: len(top2)]) / (np.sum(shifted) + 1e-12))
    concentration = float(np.sum(shifted[: len(top3)]) / (np.sum(shifted) + 1e-12))

    profile = {
        "top1_score": float(top3[0]),
        "top2_mean_score": top2_mean,
        "top3_mean_score": top3_mean,
        "top1_top2_gap": float(top3[0] - top3[1]) if len(top3) > 1 else 0.0,
        "top2_top3_gap": float(top3[1] - top3[2]) if len(top3) > 2 else 0.0,
        "top1_top3_gap": float(top3[0] - top3[-1]) if len(top3) > 1 else 0.0,
        "top2_vs_top10_gap": float(top2_mean - top10_after_top2_mean),
        "top3_vs_top10_gap": float(top3_mean - top10_tail_mean),
        "top2_concentration": top2_concentration,
        "top3_concentration": concentration,
    }
    for rank_idx, prefix in ((0, "top1"), (1, "top2"), (2, "top3")):
        for col in (
            "relative_strength_20_rank",
            "hotspot_strength_rank",
            "ret_5_rank",
            "ret_20_rank",
            "amount_moderate_20_rank",
            "no_overheat_20_rank",
            "controlled_hotspot",
            "selector_consensus",
            "selector_hot_rotation",
            "selector_defensive",
        ):
            profile[f"{prefix}_{col}"] = _candidate_value(score_df, rank_idx, col)
    return profile


def _candidate_value(score_df: pd.DataFrame, rank_idx: int, column: str, default: float = 0.5) -> float:
    if column not in score_df.columns or len(score_df) <= rank_idx:
        return default
    value = pd.to_numeric(pd.Series([score_df.iloc[rank_idx][column]]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return default
    return float(value)


def _scale_weights(base_weights: Tuple[float, ...], total_position: float, ranked_stock_count: int) -> List[float]:
    top_k = min(len(base_weights), ranked_stock_count)
    return [round(weight * total_position, 10) for weight in base_weights[:top_k]]


def _flag(value: bool) -> float:
    return 1.0 if value else 0.0


def _choose_positioning_core(
    snapshot: Dict[str, float],
    ranked_stock_count: int,
    score_profile: Dict[str, float] | None = None,
    *,
    enable_v2: bool,
) -> Tuple[str, List[float], Dict[str, float]]:
    """
    Top2/Top3 进攻型动态仓位。

    只使用预测日前的市场状态和头部排序结构：
    强主题/弱宽度时允许 Top2 集中进攻；普通或信号弱时回到 Top3 容错。
    """
    score_profile = score_profile or {}
    market_ret_20 = snapshot.get("market_ret_20", 0.0)
    market_ret_5 = snapshot.get("market_ret_5", 0.0)
    up_ratio_5 = snapshot.get("up_ratio_5", 0.5)
    up_ratio_20 = snapshot.get("up_ratio_20", 0.5)
    selected_ret_20 = snapshot.get("selected_ret_20_mean", 0.0)
    selected_ret_5 = snapshot.get("selected_ret_5_mean", 0.0)
    drawdown_20 = snapshot.get("market_drawdown_20", 0.0)
    ret5_dispersion = snapshot.get("ret5_p95_minus_median", 0.0)
    ret20_dispersion = snapshot.get("ret20_p95_minus_median", 0.0)
    top1_top3_gap = score_profile.get("top1_top3_gap", 0.0)
    top2_top3_gap = score_profile.get("top2_top3_gap", 0.0)
    top2_vs_top10_gap = score_profile.get("top2_vs_top10_gap", 0.0)
    top3_vs_top10_gap = score_profile.get("top3_vs_top10_gap", 0.0)
    top2_concentration = score_profile.get("top2_concentration", 0.0)
    top3_concentration = score_profile.get("top3_concentration", 0.0)
    top1_top2_gap = score_profile.get("top1_top2_gap", 0.0)
    third_relative = score_profile.get("top3_relative_strength_20_rank", 0.5)
    third_hotspot = score_profile.get("top3_hotspot_strength_rank", 0.5)
    third_amount_moderate = score_profile.get("top3_amount_moderate_20_rank", 0.5)
    third_no_overheat = score_profile.get("top3_no_overheat_20_rank", 0.5)
    second_quality = _candidate_quality(score_profile, "top2")
    third_quality = _candidate_quality(score_profile, "top3")
    selector_regime = str(snapshot.get("selector_regime", ""))
    market_regime_v2 = classify_market_regime(snapshot, score_profile)

    snapshot.update({f"score_{key}": value for key, value in score_profile.items()})

    market_weak = market_ret_20 < -0.035 or market_ret_5 < -0.020 or drawdown_20 < -0.060
    breadth_weak = up_ratio_20 < 0.40 or up_ratio_5 < 0.38
    signal_weak = (
        top3_vs_top10_gap < 0.012
        and top1_top3_gap < 0.012
        and top3_concentration < 0.40
        and selected_ret_20 < 0.08
    )
    strong_theme = (
        selected_ret_20 > 0.22
        or selected_ret_5 > 0.08
        or top3_vs_top10_gap > 0.035
        or top3_concentration > 0.55
    )
    weak_breadth = up_ratio_5 < 0.42 or up_ratio_20 < 0.45
    not_crash = market_ret_5 > -0.045 and drawdown_20 > -0.090
    high_dispersion = ret5_dispersion > 0.070 or ret20_dispersion > 0.140
    candidate_hot = (
        selected_ret_5 > 0.035
        or selected_ret_20 > 0.120
        or top2_vs_top10_gap > 0.018
        or top2_concentration > 0.460
    )
    weak_spike_top2 = (
        ranked_stock_count >= 2
        and weak_breadth
        and not_crash
        and candidate_hot
        and (high_dispersion or selected_ret_5 > 0.050 or top2_vs_top10_gap > 0.025)
        and not signal_weak
    )
    third_overheat_tail = (
        third_no_overheat < 0.12
        and third_amount_moderate < 0.25
        and third_hotspot > 0.85
        and third_relative > 0.80
    )
    third_unsupported_tail = third_relative < 0.50 and third_hotspot < 0.70
    third_candidate_tail_risk = ranked_stock_count >= 3 and (third_overheat_tail or third_unsupported_tail)
    third_tail_risk = (
        ranked_stock_count >= 3
        and not signal_weak
        and (selected_ret_5 > 0.025 or selected_ret_20 > 0.080 or strong_theme)
        and (
            (top2_top3_gap > 0.010 and top2_vs_top10_gap > 0.018)
            or (top2_concentration > 0.480 and top3_concentration < 0.620 and top2_top3_gap > 0.006)
            or (weak_breadth and candidate_hot and top2_top3_gap > 0.004)
            or (third_candidate_tail_risk and top2_vs_top10_gap > 0.012 and selected_ret_20 > 0.18)
        )
    )
    controlled_clean_squeeze = (
        selector_regime == "controlled_theme_squeeze"
        and market_ret_20 <= 0.0
        and up_ratio_20 < 0.34
        and selected_ret_20 > 0.52
    )
    rotation_capitulation = (
        selector_regime == "rotation_hotspot"
        and market_ret_5 < -0.020
        and up_ratio_20 < 0.36
        and selected_ret_20 > 0.25
    )
    weak_defensive_focus = (
        selector_regime == "weak_defensive_hgb_guard"
        and market_ret_20 < -0.015
        and selected_ret_20 > 0.12
    )
    hgb_weak_squeeze = (
        selector_regime == "hgb_anchor"
        and market_ret_20 < -0.025
        and up_ratio_20 < 0.35
        and selected_ret_20 > 0.40
    )
    hgb_strong_not_frothy = (
        selector_regime == "hgb_anchor"
        and market_ret_20 > 0.055
        and 0.35 <= selected_ret_20 <= 0.55
        and up_ratio_20 <= 0.72
    )

    use_top2 = (
        ranked_stock_count >= 2
        and not signal_weak
        and (
            controlled_clean_squeeze
            or rotation_capitulation
            or weak_defensive_focus
            or hgb_weak_squeeze
            or hgb_strong_not_frothy
            or (enable_v2 and weak_spike_top2)
            or (enable_v2 and third_tail_risk)
        )
    )

    p_top2_rule = 0.25
    p_top2_rule += 0.20 if controlled_clean_squeeze or rotation_capitulation else 0.0
    p_top2_rule += 0.15 if weak_defensive_focus or hgb_weak_squeeze or hgb_strong_not_frothy else 0.0
    p_top2_rule += 0.20 if weak_spike_top2 else 0.0
    p_top2_rule += 0.20 if third_tail_risk else 0.0
    p_top2_rule -= 0.20 if signal_weak else 0.0
    p_top2_rule = float(np.clip(p_top2_rule, 0.0, 1.0))
    reasons = []
    if controlled_clean_squeeze:
        reasons.append("controlled_clean_squeeze")
    if rotation_capitulation:
        reasons.append("rotation_capitulation")
    if weak_defensive_focus:
        reasons.append("weak_defensive_focus")
    if hgb_weak_squeeze:
        reasons.append("hgb_weak_squeeze")
    if hgb_strong_not_frothy:
        reasons.append("hgb_strong_not_frothy")
    if enable_v2 and weak_spike_top2:
        reasons.append("weak_spike_top2")
    if enable_v2 and third_tail_risk:
        reasons.append("third_tail_risk")
    if signal_weak:
        reasons.append("signal_weak")
    snapshot.update(
        {
            "positioning_policy": "v2" if enable_v2 else "legacy",
            "market_regime_v2": market_regime_v2,
            "market_weak_flag": _flag(market_weak),
            "breadth_weak_flag": _flag(breadth_weak),
            "weak_breadth_flag": _flag(weak_breadth),
            "signal_weak_flag": _flag(signal_weak),
            "strong_theme_flag": _flag(strong_theme),
            "high_dispersion_flag": _flag(high_dispersion),
            "candidate_hot_flag": _flag(candidate_hot),
            "weak_spike_top2_flag": _flag(enable_v2 and weak_spike_top2),
            "third_tail_risk_flag": _flag(enable_v2 and third_tail_risk),
            "third_overheat_tail_flag": _flag(enable_v2 and third_overheat_tail),
            "third_unsupported_tail_flag": _flag(enable_v2 and third_unsupported_tail),
            "second_quality": second_quality,
            "third_quality": third_quality,
            "p_top2_rule": p_top2_rule,
            "topk_reason": "|".join(reasons) if reasons else "default_top3",
        }
    )

    # 只有主题收缩足够清晰时才用 Top2；仍然禁止 Top1。
    if use_top2 and not (market_weak and breadth_weak and selected_ret_20 < 0.05):
        total_position = 1.0
        if second_quality < 0.53 or top1_top2_gap > 0.025:
            snapshot["top2_weight_reason"] = "top1_conviction_or_second_slot_thin"
            return "attack_top2_conviction_65_35", _scale_weights(TOP2_CONVICTION_WEIGHTS, total_position, ranked_stock_count), snapshot
        if second_quality > 0.62 and top1_top2_gap < 0.008:
            snapshot["top2_weight_reason"] = "top1_top2_balanced_quality"
            return "attack_top2_balanced_55_45", _scale_weights(TOP2_BALANCED_WEIGHTS, total_position, ranked_stock_count), snapshot
        snapshot["top2_weight_reason"] = "top2_standard_quality"
        return "attack_top2_dynamic_60_40", _scale_weights(BASE_TOP2_WEIGHTS, total_position, ranked_stock_count), snapshot

    # 主线明确但不够集中时，满仓三股进攻。
    if strong_theme and not (market_weak and breadth_weak and selected_ret_20 < 0.05):
        total_position = 1.0
        return "attack_top3_strong", _scale_weights(BASE_TOP3_WEIGHTS, total_position, ranked_stock_count), snapshot

    # 三类弱信号同时出现才降仓，防止极弱市硬吃回撤；只看大盘弱不降仓。
    if market_weak and breadth_weak and signal_weak:
        if selected_ret_20 < 0.03 and top3_concentration < 0.35 and market_ret_20 < -0.05:
            total_position = 0.4
            return "attack_top3_very_weak_40pct", _scale_weights(BASE_TOP3_WEIGHTS, total_position, ranked_stock_count), snapshot
        if selected_ret_20 < 0.08 or top3_vs_top10_gap < 0.006:
            total_position = 0.6
            return "attack_top3_weak_60pct", _scale_weights(BASE_TOP3_WEIGHTS, total_position, ranked_stock_count), snapshot
        total_position = 0.8
        return "attack_top3_mid_80pct", _scale_weights(BASE_TOP3_WEIGHTS, total_position, ranked_stock_count), snapshot

    total_position = 1.0
    return "attack_top3_full", _scale_weights(BASE_TOP3_WEIGHTS, total_position, ranked_stock_count), snapshot


def choose_positioning(
    snapshot: Dict[str, float],
    ranked_stock_count: int,
    score_profile: Dict[str, float] | None = None,
) -> Tuple[str, List[float], Dict[str, float]]:
    policy = os.environ.get("POSITIONING_POLICY", "v2").strip().lower()
    enable_v2 = policy not in {"legacy", "old", "v1"}
    return _choose_positioning_core(snapshot, ranked_stock_count, score_profile, enable_v2=enable_v2)


def select_portfolio_from_ranking(
    score_df: pd.DataFrame,
    weights: List[float],
    snapshot: Dict[str, float] | None = None,
    score_profile: Dict[str, float] | None = None,
    *,
    score_col: str = "final_score",
    challenger_df: pd.DataFrame | None = None,
    challenger_score_col: str | None = None,
    challenger_name: str = "challenger",
) -> Tuple[List[str], List[float], Dict[str, float]]:
    """
    Final Top2/Top3 candidate gate.

    It does not use realized returns. It can only reshuffle the second slot when
    rank2 is weak, rank3 is clearly healthier, and model score gap is small.
    """
    if score_df.empty:
        raise ValueError("score_df 为空，无法选择组合")
    snapshot = dict(snapshot or {})
    score_profile = score_profile or {}
    code_col = "stock_id" if "stock_id" in score_df.columns else "股票代码"
    if code_col not in score_df.columns:
        raise ValueError("score_df 缺少股票代码列")

    ranked = score_df.copy()
    score_col = _resolve_score_col(ranked, score_col)
    if score_col in ranked.columns:
        ranked = ranked.sort_values(score_col, ascending=False)
    ranked = ranked.reset_index(drop=True)
    selected_indices = list(range(min(len(weights), len(ranked))))
    action = "head_selection"
    replacement_code = ""
    replacement_reason = ""

    second_quality = _candidate_quality(score_profile, "top2")
    third_quality = _candidate_quality(score_profile, "top3")
    top2_top3_gap = _profile_value(score_profile, "top2_top3_gap")
    top2_vs_top10_gap = _profile_value(score_profile, "top2_vs_top10_gap")
    third_tail_risk = float(snapshot.get("third_tail_risk_flag", 0.0) or 0.0) > 0.5
    third_overheat_tail = float(snapshot.get("third_overheat_tail_flag", 0.0) or 0.0) > 0.5
    third_unsupported_tail = float(snapshot.get("third_unsupported_tail_flag", 0.0) or 0.0) > 0.5

    challenger_ranked, challenger_lookup = _prepare_challenger_lookup(
        challenger_df,
        challenger_score_col or score_col,
    )
    if len(ranked) >= 3:
        second_row = _merge_challenger_evidence(ranked.iloc[1], challenger_lookup, code_col)
        third_row = _merge_challenger_evidence(ranked.iloc[2], challenger_lookup, code_col)
    elif len(ranked) >= 2:
        second_row = _merge_challenger_evidence(ranked.iloc[1], challenger_lookup, code_col)
        third_row = pd.Series(dtype=float)
    else:
        second_row = pd.Series(dtype=float)
        third_row = pd.Series(dtype=float)

    second_row_quality = _row_candidate_quality(second_row) if not second_row.empty else second_quality
    third_row_quality = _row_candidate_quality(third_row) if not third_row.empty else third_quality
    second_quality = max(second_quality, second_row_quality)
    third_quality = max(third_quality, third_row_quality)
    second_industry_confirmation = _row_industry_confirmation(second_row) if not second_row.empty else 0.5
    third_industry_confirmation = _row_industry_confirmation(third_row) if not third_row.empty else 0.5
    second_overheat_risk = _row_overheat_risk(second_row) if not second_row.empty else 0.5
    third_overheat_risk = _row_overheat_risk(third_row) if not third_row.empty else 0.5
    market_regime = str(snapshot.get("market_regime_v2", ""))
    weak_spike_like = (
        market_regime in {"weak_local_spike", "deep_weak_hot", "narrow_hot_theme", "fragile_rebound"}
        or float(snapshot.get("weak_spike_top2_flag", 0.0) or 0.0) > 0.5
    )
    second_slot_stale_tail = False
    low_signal_second_hold = False

    if len(weights) == 2 and len(ranked) >= 3:
        rank2_weak = second_quality < 0.49
        rank3_healthier = third_quality > second_quality + 0.08
        small_model_gap = top2_top3_gap < 0.008 or top2_vs_top10_gap < 0.014
        third_not_banned = not (third_tail_risk or third_overheat_tail or third_unsupported_tail)
        second_fresh = _row_fresh_continuation(second_row) if not second_row.empty else 0.5
        third_fresh = _row_fresh_continuation(third_row) if not third_row.empty else 0.5
        low_signal_second_hold = (
            weak_spike_like
            and second_quality < 0.35
            and second_fresh < 0.45
            and second_overheat_risk < 0.58
        )
        if (
            rank2_weak
            and rank3_healthier
            and small_model_gap
            and third_not_banned
            and not low_signal_second_hold
        ):
            selected_indices = [0, 2]
            action = "swap_rank2_to_rank3_second_slot_gate"
            replacement_code = str(ranked.iloc[2][code_col]).zfill(6)
            replacement_reason = "rank2_weak_rank3_healthier"

        rank2_stale_tail = (
            weak_spike_like
            and _row_value(second_row, "ret_20_rank", 0.5) >= 0.92
            and _row_value(second_row, "relative_strength_20_rank", 0.5) >= 0.88
            and _row_value(second_row, "ret_5_rank", 0.5) <= 0.22
            and second_fresh <= 0.18
            and second_industry_confirmation <= 0.62
        )
        rank3_fresher_tail = (
            third_quality >= second_quality - 0.020
            and third_fresh >= second_fresh + 0.25
            and _row_value(third_row, "ret_10_rank", 0.5) >= 0.85
            and _row_value(third_row, "ret_5_rank", 0.5)
            >= _row_value(second_row, "ret_5_rank", 0.5) + 0.15
        )
        if action == "head_selection" and rank2_stale_tail and rank3_fresher_tail:
            selected_indices = [0, 2]
            action = "swap_rank2_to_rank3_stale_tail_gate"
            replacement_code = str(ranked.iloc[2][code_col]).zfill(6)
            replacement_reason = "rank2_stale_tail_rank3_fresher"
            second_slot_stale_tail = True

        rank2_isolated_hot = (
            _row_heat(second_row) > 0.74
            and second_industry_confirmation < 0.54
            and second_quality < 0.73
        )
        rank2_overheated = second_overheat_risk > 0.62 and second_industry_confirmation < 0.62
        rank2_confirmed_overheat_tail = second_overheat_risk > 0.74 and second_quality < 0.78
        second_slot_risky = rank2_weak or (
            weak_spike_like
            and (rank2_isolated_hot or rank2_overheated or rank2_confirmed_overheat_tail)
        )

        if (
            action == "head_selection"
            and weak_spike_like
            and second_slot_risky
            and not low_signal_second_hold
        ):
            blocked_codes = {
                str(ranked.iloc[0][code_col]).zfill(6),
                str(ranked.iloc[1][code_col]).zfill(6),
            }
            candidate_rows: list[Tuple[str, pd.Series, str, int]] = []
            for idx in range(2, min(20, len(ranked))):
                row = _merge_challenger_evidence(ranked.iloc[idx], challenger_lookup, code_col)
                candidate_rows.append((str(row[code_col]).zfill(6), row, "same_lane", idx))
            if challenger_ranked is not None:
                challenger_code_col = "stock_id" if "stock_id" in challenger_ranked.columns else "股票代码"
                for idx, (_, row) in enumerate(challenger_ranked.head(20).iterrows()):
                    code = str(row[challenger_code_col]).zfill(6)
                    candidate_rows.append((code, row, challenger_name, idx))

            best_code = ""
            best_score = -np.inf
            best_source = ""
            best_quality = 0.0
            best_industry = 0.0
            best_overheat = 1.0
            best_fresh = 0.0
            best_micro = 0.0
            best_moneyflow = 0.5
            seen_codes = set()
            for code, row, source, idx in candidate_rows:
                if code in blocked_codes or code in seen_codes:
                    continue
                seen_codes.add(code)
                quality = _row_candidate_quality(row)
                industry_confirmation = _row_industry_confirmation(row)
                overheat_risk = _row_overheat_risk(row)
                fresh_continuation = _row_fresh_continuation(row)
                moneyflow_confirmation = _row_moneyflow_confirmation(row)
                micro_leader_score = _row_micro_industry_leader_score(row)
                micro_leader_replacement = (
                    second_overheat_risk > 0.74
                    and industry_confirmation >= 0.90
                    and fresh_continuation >= 0.58
                    and _row_micro_industry_leader(row)
                )
                rank_bonus = max(0.0, 1.0 - 0.10 * idx)
                if source != "same_lane":
                    rank_bonus = max(rank_bonus, _row_value(row, "_challenger_rank_score", 0.5))
                healthier = (
                    quality > second_quality + 0.045
                    or industry_confirmation > second_industry_confirmation + 0.095
                )
                fresh_confirmed_replacement = (
                    second_overheat_risk > 0.74
                    and industry_confirmation >= 0.90
                    and fresh_continuation >= 0.58
                    and overheat_risk <= 0.58
                    and _row_value(row, "ret_10_rank", 0.5) >= 0.90
                    and _row_value(row, "ret_20_rank", 0.5) >= 0.90
                    and _row_value(row, "amount_moderate_20_rank", 0.5) >= 0.45
                )
                industry_confirmed_replacement = (
                    industry_confirmation >= 0.70
                    or (industry_confirmation >= 0.57 and fresh_continuation >= 0.50)
                )
                challenger_quality_confirmed = (
                    source != "same_lane"
                    and quality >= 0.70
                    and (
                        industry_confirmation >= 0.70
                        or fresh_continuation >= 0.55
                    )
                )
                confirmed = (
                    industry_confirmed_replacement
                    or challenger_quality_confirmed
                    or fresh_confirmed_replacement
                    or micro_leader_replacement
                )
                not_banned = overheat_risk < 0.74 or industry_confirmation >= 0.67 or micro_leader_replacement
                if rank2_confirmed_overheat_tail and not (fresh_confirmed_replacement or micro_leader_replacement):
                    continue
                if not ((healthier and confirmed and not_banned) or fresh_confirmed_replacement or micro_leader_replacement):
                    continue
                candidate_score = (
                    0.33 * quality
                    + 0.28 * industry_confirmation
                    + 0.21 * fresh_continuation
                    + 0.12 * rank_bonus
                    + 0.04 * moneyflow_confirmation
                    - 0.20 * overheat_risk
                )
                if micro_leader_replacement:
                    candidate_score += 0.24 + 0.18 * micro_leader_score
                if candidate_score > best_score:
                    best_code = code
                    best_score = candidate_score
                    best_source = source
                    best_quality = quality
                    best_industry = industry_confirmation
                    best_overheat = overheat_risk
                    best_fresh = fresh_continuation
                    best_micro = micro_leader_score if micro_leader_replacement else 0.0
                    best_moneyflow = moneyflow_confirmation

            if best_code:
                replacement_code = best_code
                replacement_reason = (
                    f"weak_spike_second_slot_guard:{best_source}:"
                    f"q={best_quality:.3f}:ind={best_industry:.3f}:"
                    f"fresh={best_fresh:.3f}:risk={best_overheat:.3f}:"
                    f"micro={best_micro:.3f}:mf={best_moneyflow:.3f}"
                )
                action = "swap_rank2_to_confirmed_second_slot_gate"

        if (
            action == "head_selection"
            and weak_spike_like
            and second_overheat_risk > 0.74
            and second_quality < 0.78
            and len(weights) == 2
        ):
            weights = list(TOP2_CONVICTION_WEIGHTS)
            action = "haircut_rank2_overheat_65_35"
            replacement_reason = "no_confirmed_replacement_second_overheat_haircut"
            snapshot["top2_weight_reason"] = "second_slot_overheat_haircut"

    if replacement_code:
        selected_stock_ids = [str(ranked.iloc[0][code_col]).zfill(6), replacement_code]
    else:
        selected_stock_ids = (
            ranked.iloc[selected_indices][code_col]
            .astype(str)
            .str.zfill(6)
            .tolist()
        )
    snapshot.update(
        {
            "portfolio_selection_action": action,
            "selected_stock_ids_after_gate": ",".join(selected_stock_ids),
            "second_quality": second_quality,
            "third_quality": third_quality,
            "second_industry_confirmation": second_industry_confirmation,
            "third_industry_confirmation": third_industry_confirmation,
            "second_overheat_risk": second_overheat_risk,
            "third_overheat_risk": third_overheat_risk,
            "second_slot_risky_flag": _flag(weak_spike_like and (second_quality < 0.49 or second_industry_confirmation < 0.54 or second_overheat_risk > 0.62)),
            "second_slot_stale_tail_flag": _flag(second_slot_stale_tail),
            "low_signal_second_hold_flag": _flag(low_signal_second_hold),
            "second_slot_replacement_code": replacement_code,
            "second_slot_replacement_reason": replacement_reason,
        }
    )
    return selected_stock_ids, weights[: len(selected_stock_ids)], snapshot
