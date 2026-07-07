"""
Tushare factor registry for the HS300 competition pipeline.

This module is intentionally declarative. It records what we are allowed to
fetch, how the data becomes model features, and what visibility rule must hold
before a factor can enter training or prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class InterfacePlan:
	name: str
	phase: str
	min_points: int | None
	independent_permission: bool
	visibility: str
	raw_dir: str
	factor_file: str
	factor_family: str
	priority: str
	key_fields: tuple[str, ...]
	model_features: tuple[str, ...]
	research_basis: tuple[str, ...]
	notes: str = ""


PLANS: tuple[InterfacePlan, ...] = (
	InterfacePlan(
		name="daily_basic",
		phase="5000",
		min_points=2000,
		independent_permission=False,
		visibility="trade_date盘后，默认下一交易日可见",
		raw_dir="data/raw/tushare/daily_basic",
		factor_file="data/offline_factors/daily_basic_factors.csv",
		factor_family="valuation_size_liquidity",
		priority="P0",
		key_fields=(
			"ts_code",
			"trade_date",
			"turnover_rate",
			"turnover_rate_f",
			"volume_ratio",
			"pe_ttm",
			"pb",
			"ps_ttm",
			"dv_ttm",
			"total_mv",
			"circ_mv",
			"free_share",
		),
		model_features=(
			"ts_db_log_total_mv_rank",
			"ts_db_log_circ_mv_rank",
			"ts_db_pb_value_rank",
			"ts_db_pe_ttm_value_rank",
			"ts_db_ps_ttm_value_rank",
			"ts_db_turnover_rate_f_rank",
			"ts_db_volume_ratio_rank",
		),
		research_basis=(
			"Fama-French size/value",
			"Amihud liquidity",
		),
	),
	InterfacePlan(
		name="moneyflow",
		phase="5000",
		min_points=2000,
		independent_permission=False,
		visibility="trade_date盘后，默认下一交易日可见",
		raw_dir="data/raw/tushare/moneyflow",
		factor_file="data/offline_factors/moneyflow_factors.csv",
		factor_family="capital_flow",
		priority="P0",
		key_fields=(
			"ts_code",
			"trade_date",
			"buy_lg_amount",
			"sell_lg_amount",
			"buy_elg_amount",
			"sell_elg_amount",
			"net_mf_amount",
		),
		model_features=(
			"ts_mf_main_net_5_rank",
			"ts_mf_main_net_10_rank",
			"ts_mf_main_net_ratio_5_rank",
			"ts_mf_main_net_ratio_10_rank",
			"ts_mf_large_order_share_rank",
		),
		research_basis=(
			"order-flow / liquidity microstructure proxy",
			"short-horizon price-volume alpha",
		),
	),
	InterfacePlan(
		name="index_weight",
		phase="5000",
		min_points=2000,
		independent_permission=False,
		visibility="月度成分发布后可见，禁止未来成分回填",
		raw_dir="data/raw/tushare/index_weight",
		factor_file="data/offline_factors/index_weight_factors.csv",
		factor_family="universe_index_weight",
		priority="P0",
		key_fields=("index_code", "con_code", "trade_date", "weight"),
		model_features=("ts_idx_weight_rank", "ts_idx_weight_change_20_rank"),
		research_basis=("index membership and benchmark pressure control",),
	),
	InterfacePlan(
		name="index_classify,index_member,index_member_all,sw_daily",
		phase="5000",
		min_points=2000,
		independent_permission=False,
		visibility="行业映射按已知版本或生效日；行业行情盘后下一交易日可见",
		raw_dir="data/raw/tushare/industry",
		factor_file="data/offline_factors/industry_factors.csv",
		factor_family="industry_rotation",
		priority="P0",
		key_fields=("index_code", "con_code", "trade_date", "industry_name"),
		model_features=(
			"industry_ret_5_rank",
			"industry_ret_20_rank",
			"industry_breadth_5_rank",
			"stock_industry_excess_20_rank",
		),
		research_basis=(
			"industry momentum",
			"cross-sectional relative strength",
		),
	),
	InterfacePlan(
		name="top_list,top_inst,stk_limit",
		phase="5000",
		min_points=2000,
		independent_permission=False,
		visibility="龙虎榜晚间公开、涨跌停价盘前可得；统一按下一交易日可见",
		raw_dir="data/raw/tushare/top_list + data/raw/tushare/top_inst + data/raw/tushare/stk_limit",
		factor_file="data/offline_factors/event_factors.csv",
		factor_family="event_attention",
		priority="P1",
		key_fields=("ts_code", "trade_date", "net_buy", "amount", "exalter", "up_limit", "down_limit"),
		model_features=(
			"ts_event_top_list_flag",
			"ts_event_inst_net_buy_rank",
			"ts_event_top_list_amount_rank",
			"ts_event_inst_amount_rank",
			"ts_event_up_limit_gap_close_rank",
			"ts_event_near_limit_up_count_5_rank",
			"ts_event_near_limit_cooldown_5_rank",
		),
		research_basis=("event-driven attention, delayed reaction control",),
	),
	InterfacePlan(
		name="cyq_perf",
		phase="15000",
		min_points=5000,
		independent_permission=False,
		visibility="18-19点更新；统一按下一交易日可见",
		raw_dir="data/raw/tushare/cyq_perf",
		factor_file="data/offline_factors/cyq_perf_factors.csv",
		factor_family="chip_cost_pressure",
		priority="P0",
		key_fields=(
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
		),
		model_features=(
			"ts_cyq_winner_rate_rank",
			"ts_cyq_winner_rate_chg_5_rank",
			"ts_cyq_close_to_weight_avg_rank",
			"ts_cyq_close_to_cost_85pct_rank",
			"ts_cyq_cost_band_width_rank",
			"ts_cyq_upper_pressure_rank",
		),
		research_basis=(
			"cost distribution / winner-rate pressure",
			"behavioral finance disposition and overhead supply proxies",
		),
		notes="15000后特色数据无总量限制；第一批15000股票级候选。",
	),
	InterfacePlan(
		name="cyq_chips",
		phase="15000",
		min_points=5000,
		independent_permission=False,
		visibility="18-19点更新；统一按下一交易日可见",
		raw_dir="data/raw/tushare/cyq_chips",
		factor_file="data/offline_factors/cyq_chips_factors.csv",
		factor_family="chip_distribution_shape",
		priority="P1",
		key_fields=("ts_code", "trade_date", "price", "percent"),
		model_features=(
			"ts_cyqc_chip_concentration_rank",
			"ts_cyqc_upper_overhang_rank",
			"ts_cyqc_lower_support_rank",
		),
		research_basis=("chip distribution shape and overhead supply proxies",),
		notes="行数较大，必须先聚合后入模，禁止raw分布直接进训练。",
	),
	InterfacePlan(
		name="dc_index,dc_member,ths_index,ths_member",
		phase="15000",
		min_points=6000,
		independent_permission=False,
		visibility="板块行情盘后下一交易日可见；成分必须按trade_date/in_date/out_date点时匹配",
		raw_dir="data/raw/tushare/concept",
		factor_file="data/offline_factors/concept_factors.csv",
		factor_family="concept_rotation",
		priority="P1",
		key_fields=("ts_code", "trade_date", "con_code", "pct_change", "up_num", "down_num"),
		model_features=(
			"ts_concept_ret_3_rank",
			"ts_concept_ret_5_rank",
			"ts_concept_breadth_rank",
			"ts_concept_leading_pct_rank",
			"ts_stock_top_concept_exposure_rank",
		),
		research_basis=("theme momentum, breadth, and attention rotation",),
		notes="先作为A2尖峰行情识别候选；禁止同花顺/东方财富重复噪声无审计叠加。",
	),
	InterfacePlan(
		name="moneyflow_hsgt",
		phase="15000",
		min_points=2000,
		independent_permission=False,
		visibility="盘后/次日可见；作为市场状态，不作为个股alpha裸因子",
		raw_dir="data/raw/tushare/moneyflow_hsgt",
		factor_file="data/offline_factors/market_regime_factors.csv",
		factor_family="market_regime_north_money",
		priority="P1",
		key_fields=("trade_date", "north_money", "south_money", "hgt", "sgt"),
		model_features=(
			"ts_regime_north_money_5_rank",
			"ts_regime_north_money_chg_3_rank",
			"ts_regime_risk_on_flag",
		),
		research_basis=("market regime and foreign-flow risk appetite",),
		notes="只服务Top2/Top3 selector和风险开关。",
	),
	InterfacePlan(
		name="broker_recommend,hk_hold",
		phase="15000",
		min_points=6000,
		independent_permission=False,
		visibility="券商金股按月滞后；hk_hold在2024-08-20后改季度披露",
		raw_dir="data/raw/tushare/slow_attention",
		factor_file="data/offline_factors/slow_attention_factors.csv",
		factor_family="slow_attention",
		priority="P2",
		key_fields=("ts_code", "trade_date", "month", "broker", "ratio"),
		model_features=(
			"ts_broker_recommend_count_rank",
			"ts_hk_hold_ratio_rank",
		),
		research_basis=("sell-side attention and long-horizon ownership proxies",),
		notes="短周期T+5优先级低；hk_hold不再适合日频主链。",
	),
	InterfacePlan(
		name="fina_indicator,income,cashflow,balancesheet,forecast,express",
		phase="5000",
		min_points=2000,
		independent_permission=False,
		visibility="按ann_date/f_ann_date生成available_date，默认下一交易日可见",
		raw_dir="data/raw/tushare/financial",
		factor_file="data/offline_factors/financial_factors.csv",
		factor_family="quality_growth",
		priority="P1",
		key_fields=(
			"ts_code",
			"ann_date",
			"f_ann_date",
			"end_date",
			"roe",
			"grossprofit_margin",
			"or_yoy",
			"netprofit_yoy",
		),
		model_features=(
			"ts_fin_roe_rank",
			"ts_fin_grossprofit_margin_rank",
			"ts_fin_netprofit_yoy_rank",
			"ts_fin_or_yoy_rank",
			"ts_fin_ocf_to_profit_rank",
		),
		research_basis=(
			"Fama-French profitability/investment",
			"Novy-Marx profitability",
			"Hou-Xue-Zhang q-factor",
		),
	),
)


def iter_plans(phase: str | None = None) -> Iterable[InterfacePlan]:
	for plan in PLANS:
		if phase is None or plan.phase == phase:
			yield plan


def to_records(phase: str | None = None) -> list[dict]:
	return [asdict(plan) for plan in iter_plans(phase)]


def required_factor_files(phase: str | None = None) -> list[str]:
	return sorted({plan.factor_file for plan in iter_plans(phase)})


def main() -> None:
	import argparse
	import csv
	import json
	from pathlib import Path

	parser = argparse.ArgumentParser(description="导出 Tushare 因子接口规划表")
	parser.add_argument("--phase", default=None, help="例如 5000；为空则导出全部")
	parser.add_argument("--output", default=None, help="输出 CSV/JSON 文件；为空则打印 JSON")
	args = parser.parse_args()

	records = to_records(args.phase)
	if args.output is None:
		print(json.dumps(records, ensure_ascii=False, indent=2))
		return

	out = Path(args.output)
	out.parent.mkdir(parents=True, exist_ok=True)
	if out.suffix.lower() == ".json":
		out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
	else:
		fieldnames = list(records[0].keys()) if records else []
		with out.open("w", encoding="utf-8-sig", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for row in records:
				writer.writerow(row)


if __name__ == "__main__":
	main()
