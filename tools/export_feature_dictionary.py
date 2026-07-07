"""
Export a machine-readable feature dictionary.

The model keeps stable English feature names; this file records Chinese names,
sources, visibility, and current admission status for analysis reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


ROWS: list[dict[str, str]] = [
    {"feature_name": "股票代码", "中文名": "股票代码", "来源": "主行情表", "含义": "6位A股代码", "可见性": "原始数据", "当前状态": "身份列"},
    {"feature_name": "日期", "中文名": "交易日期", "来源": "主行情表", "含义": "行情对应交易日", "可见性": "原始数据", "当前状态": "身份列"},
    {"feature_name": "开盘", "中文名": "开盘价", "来源": "主行情表", "含义": "当日开盘价", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "收盘", "中文名": "收盘价", "来源": "主行情表", "含义": "当日收盘价", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "最高", "中文名": "最高价", "来源": "主行情表", "含义": "当日最高价", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "最低", "中文名": "最低价", "来源": "主行情表", "含义": "当日最低价", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "成交量", "中文名": "成交量", "来源": "主行情表", "含义": "当日成交股数", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "成交额", "中文名": "成交额", "来源": "主行情表", "含义": "当日成交金额", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "振幅", "中文名": "振幅", "来源": "主行情表", "含义": "当日最高最低波动幅度", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "涨跌额", "中文名": "涨跌额", "来源": "主行情表", "含义": "当日价格变化额", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "换手率", "中文名": "换手率", "来源": "主行情表", "含义": "当日换手比例", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "涨跌幅", "中文名": "涨跌幅", "来源": "主行情表", "含义": "当日收益率百分比", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "intraday_strength", "中文名": "日内强弱", "来源": "主行情衍生", "含义": "收盘相对开盘的强弱", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "range_pct", "中文名": "日内振幅比例", "来源": "主行情衍生", "含义": "最高最低差相对开盘价", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "upper_shadow", "中文名": "上影线比例", "来源": "主行情衍生", "含义": "上影线相对开盘价", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "lower_shadow", "中文名": "下影线比例", "来源": "主行情衍生", "含义": "下影线相对开盘价", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "vwap_gap", "中文名": "成交均价偏离", "来源": "主行情衍生", "含义": "成交额/成交量近似均价与收盘价偏离", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "stable_mom_20", "中文名": "20日稳定动量", "来源": "主行情衍生", "含义": "20日收益除以20日波动率", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "ret_accel_5_20", "中文名": "5-20日动量加速度", "来源": "主行情衍生", "含义": "5日收益减20日收益", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "turnover_accel_5_20", "中文名": "5-20日换手加速度", "来源": "主行情衍生", "含义": "5日平均换手减20日平均换手", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "volume_price_confirm", "中文名": "成交量价格确认", "来源": "主行情衍生", "含义": "5日收益与成交量放大的交互", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "amount_price_confirm", "中文名": "成交额价格确认", "来源": "主行情衍生", "含义": "5日收益与成交额放大的交互", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "board_hotspot", "中文名": "代码段板块热点", "来源": "主行情衍生", "含义": "用代码段近似板块强度", "可见性": "交易日行情", "当前状态": "已入模"},
    {"feature_name": "selector_score", "中文名": "最终选择器分数", "来源": "模型融合", "含义": "用于正式排序的二层融合信号", "可见性": "离线模型", "当前状态": "已入模"},
    {"feature_name": "selector_consensus", "中文名": "多模型共识分数", "来源": "模型融合", "含义": "多个模型排名的共识", "可见性": "离线模型", "当前状态": "已入模"},
    {"feature_name": "selector_hot_rotation", "中文名": "热点轮动分数", "来源": "模型融合", "含义": "偏热点/轮动行情的选择分支", "可见性": "离线模型", "当前状态": "已入模"},
    {"feature_name": "selector_defensive", "中文名": "防守分数", "来源": "模型融合", "含义": "弱市下偏稳健的选择分支", "可见性": "离线模型", "当前状态": "已入模"},
    {"feature_name": "ts_db_log_total_mv_rank", "中文名": "总市值对数横截面排名", "来源": "daily_basic", "含义": "规模因子", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_db_log_circ_mv_rank", "中文名": "流通市值对数横截面排名", "来源": "daily_basic", "含义": "流通盘规模排名", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_db_pb_value_rank", "中文名": "市净率价值排名", "来源": "daily_basic", "含义": "PB越低排名越靠前", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_db_pe_ttm_value_rank", "中文名": "TTM市盈率价值排名", "来源": "daily_basic", "含义": "PE TTM越低排名越靠前", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_db_ps_ttm_value_rank", "中文名": "TTM市销率价值排名", "来源": "daily_basic", "含义": "PS TTM越低排名越靠前", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_db_turnover_rate_f_rank", "中文名": "自由流通换手率排名", "来源": "daily_basic", "含义": "自由流通口径交易活跃度", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_db_volume_ratio_rank", "中文名": "量比排名", "来源": "daily_basic", "含义": "成交量相对近期均量活跃度", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_mf_main_net_5_rank", "中文名": "5日主力净流入排名", "来源": "moneyflow", "含义": "大单+特大单买入减卖出5日累计", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_mf_main_net_10_rank", "中文名": "10日主力净流入排名", "来源": "moneyflow", "含义": "大单+特大单买入减卖出10日累计", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_mf_main_net_ratio_5_rank", "中文名": "5日主力净流入占比排名", "来源": "moneyflow", "含义": "主力净流入占总订单金额比例", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_mf_main_net_ratio_10_rank", "中文名": "10日主力净流入占比排名", "来源": "moneyflow", "含义": "更平滑的资金持续性指标", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_mf_large_order_share_rank", "中文名": "大单特大单参与度排名", "来源": "moneyflow", "含义": "大单和特大单成交占比", "可见性": "盘后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_idx_weight_rank", "中文名": "沪深300权重排名", "来源": "index_weight", "含义": "指数权重大小，反映基准资金影响", "可见性": "月度成分发布后可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_idx_weight_change_20_rank", "中文名": "20日指数权重变化排名", "来源": "index_weight", "含义": "指数权重变化趋势", "可见性": "月度成分发布后可见", "当前状态": "待取数入模"},
    {"feature_name": "industry_ret_5_rank", "中文名": "行业5日强度排名", "来源": "行业数据", "含义": "行业短期动量", "可见性": "盘后默认T+1可见", "当前状态": "计划中"},
    {"feature_name": "industry_ret_20_rank", "中文名": "行业20日强度排名", "来源": "行业数据", "含义": "行业中期动量", "可见性": "盘后默认T+1可见", "当前状态": "计划中"},
    {"feature_name": "industry_breadth_5_rank", "中文名": "行业5日宽度排名", "来源": "行业数据", "含义": "行业内上涨扩散程度", "可见性": "盘后默认T+1可见", "当前状态": "计划中"},
    {"feature_name": "stock_industry_excess_20_rank", "中文名": "个股相对行业20日超额排名", "来源": "行业数据", "含义": "个股是否跑赢所属行业", "可见性": "盘后默认T+1可见", "当前状态": "计划中"},
    {"feature_name": "ts_event_top_list_flag", "中文名": "龙虎榜上榜标记", "来源": "top_list", "含义": "是否出现龙虎榜事件", "可见性": "晚间公开默认T+1可见", "当前状态": "计划中"},
    {"feature_name": "ts_event_inst_net_buy_rank", "中文名": "龙虎榜机构净买入排名", "来源": "top_inst", "含义": "机构席位净买入力度", "可见性": "晚间公开默认T+1可见", "当前状态": "计划中"},
    {"feature_name": "ts_event_top_list_amount_rank", "中文名": "龙虎榜成交额排名", "来源": "top_list", "含义": "上榜成交活跃度", "可见性": "晚间公开默认T+1可见", "当前状态": "计划中"},
    {"feature_name": "ts_event_inst_amount_rank", "中文名": "龙虎榜机构成交额排名", "来源": "top_inst", "含义": "机构参与规模", "可见性": "晚间公开默认T+1可见", "当前状态": "计划中"},
    {"feature_name": "ts_event_up_limit_gap_close_rank", "中文名": "距涨停收盘距离排名", "来源": "stk_limit", "含义": "收盘价越接近涨停价排名越靠前", "可见性": "涨跌停价盘前可得，收盘确认后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_event_up_limit_gap_high_rank", "中文名": "距涨停最高价距离排名", "来源": "stk_limit", "含义": "日内最高价越接近涨停价排名越靠前", "可见性": "涨跌停价盘前可得，收盘确认后默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_event_limit_up_count_5_rank", "中文名": "近5日涨停次数排名", "来源": "stk_limit", "含义": "短期涨停活跃度和情绪强度", "可见性": "历史事件默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_event_near_limit_up_count_5_rank", "中文名": "近5日接近涨停次数排名", "来源": "stk_limit", "含义": "短期冲板或强势接近涨停的频率", "可见性": "历史事件默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_event_limit_up_cooldown_5_rank", "中文名": "近5日涨停冷却排名", "来源": "stk_limit", "含义": "近期涨停次数越少排名越靠前，用于抑制过热追高", "可见性": "历史事件默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_event_near_limit_cooldown_5_rank", "中文名": "近5日冲板冷却排名", "来源": "stk_limit", "含义": "近期接近涨停次数越少排名越靠前，用于过滤过热但不压制健康强势", "可见性": "历史事件默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_event_limit_down_count_5_rank", "中文名": "近5日跌停风险反向排名", "来源": "stk_limit", "含义": "跌停次数越少排名越靠前，用于过滤极端下行风险", "可见性": "历史事件默认T+1可见", "当前状态": "待取数入模"},
    {"feature_name": "ts_fin_roe_rank", "中文名": "ROE排名", "来源": "财务数据", "含义": "盈利能力", "可见性": "按公告日生成可见日", "当前状态": "计划中"},
    {"feature_name": "ts_fin_grossprofit_margin_rank", "中文名": "毛利率排名", "来源": "财务数据", "含义": "盈利质量", "可见性": "按公告日生成可见日", "当前状态": "计划中"},
    {"feature_name": "ts_fin_netprofit_yoy_rank", "中文名": "净利润同比增速排名", "来源": "财务数据", "含义": "利润成长性", "可见性": "按公告日生成可见日", "当前状态": "计划中"},
    {"feature_name": "ts_fin_or_yoy_rank", "中文名": "营收同比增速排名", "来源": "财务数据", "含义": "收入成长性", "可见性": "按公告日生成可见日", "当前状态": "计划中"},
    {"feature_name": "ts_fin_ocf_to_profit_rank", "中文名": "经营现金流利润质量排名", "来源": "财务数据", "含义": "利润现金含量", "可见性": "按公告日生成可见日", "当前状态": "计划中"},
]


TUSHARE_RAW_FIELDS = {
    "ts_db_log_total_mv_rank": "total_mv",
    "ts_db_log_circ_mv_rank": "circ_mv",
    "ts_db_pb_value_rank": "pb",
    "ts_db_pe_ttm_value_rank": "pe_ttm",
    "ts_db_ps_ttm_value_rank": "ps_ttm",
    "ts_db_turnover_rate_f_rank": "turnover_rate_f",
    "ts_db_volume_ratio_rank": "volume_ratio",
    "ts_mf_main_net_5_rank": "buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount",
    "ts_mf_main_net_10_rank": "buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount",
    "ts_mf_main_net_ratio_5_rank": "buy_*_amount,sell_*_amount",
    "ts_mf_main_net_ratio_10_rank": "buy_*_amount,sell_*_amount",
    "ts_mf_large_order_share_rank": "buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount",
    "ts_idx_weight_rank": "weight",
    "ts_idx_weight_change_20_rank": "weight",
    "industry_ret_5_rank": "sw_daily行情字段",
    "industry_ret_20_rank": "sw_daily行情字段",
    "industry_breadth_5_rank": "行业成员+行情字段",
    "stock_industry_excess_20_rank": "行业成员+行情字段",
    "ts_event_top_list_flag": "ts_code,trade_date",
    "ts_event_inst_net_buy_rank": "net_buy",
    "ts_event_top_list_amount_rank": "amount",
    "ts_event_inst_amount_rank": "amount",
    "ts_event_up_limit_gap_close_rank": "up_limit,close",
    "ts_event_up_limit_gap_high_rank": "up_limit,high",
    "ts_event_limit_up_count_5_rank": "up_limit,close,pct_chg",
    "ts_event_near_limit_up_count_5_rank": "up_limit,high,close,pct_chg",
    "ts_event_limit_up_cooldown_5_rank": "up_limit,close,pct_chg",
    "ts_event_near_limit_cooldown_5_rank": "up_limit,high,close,pct_chg",
    "ts_event_limit_down_count_5_rank": "down_limit,close,pct_chg",
    "ts_fin_roe_rank": "roe",
    "ts_fin_grossprofit_margin_rank": "grossprofit_margin",
    "ts_fin_netprofit_yoy_rank": "netprofit_yoy",
    "ts_fin_or_yoy_rank": "or_yoy",
    "ts_fin_ocf_to_profit_rank": "ocf_to_profit",
}


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched = []
    for row in rows:
        item = dict(row)
        feature_name = item["feature_name"]
        source = item.get("来源", "")
        if source in {"daily_basic", "moneyflow", "index_weight", "top_list", "top_inst", "stk_limit", "财务数据", "行业数据"}:
            item["字段层级"] = "内部衍生特征"
            item["Tushare原始字段"] = TUSHARE_RAW_FIELDS.get(feature_name, "")
        elif source == "主行情表":
            item["字段层级"] = "比赛基础中文字段"
            item["Tushare原始字段"] = ""
        else:
            item["字段层级"] = "模型内部衍生特征"
            item["Tushare原始字段"] = ""
        enriched.append(item)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="导出特征中文词典")
    parser.add_argument("--output", default=str(ROOT / "output" / "feature_dictionary_5000.csv"))
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["feature_name", "中文名", "字段层级", "来源", "Tushare原始字段", "含义", "可见性", "当前状态"]
    pd.DataFrame(enrich_rows(ROWS)).to_csv(out, index=False, encoding="utf-8-sig", columns=columns)
    print(f"feature dictionary: {out}")


if __name__ == "__main__":
    main()
