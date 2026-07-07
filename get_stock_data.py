#!/usr/bin/env python3
"""
Fetch HS300 daily data from Tushare and export the legacy training schema.

This script is for pre-race data freezing only. The official train/predict
pipeline must keep reading local CSV files and must not require network access,
Tushare token, MCP, or Codex skills at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT_COLUMNS = [
	"股票代码",
	"日期",
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="用 Tushare 拉取沪深300成分股后复权日线，并导出当前模型兼容的 CSV。"
	)
	parser.add_argument("--start-date", default="2024-04-09", help="开始日期，YYYY-MM-DD 或 YYYYMMDD")
	parser.add_argument("--end-date", default="2026-06-22", help="结束日期，YYYY-MM-DD 或 YYYYMMDD")
	parser.add_argument(
		"--component-date",
		default=None,
		help="成分股口径日期，默认使用 --end-date。脚本会向前回看月度成分权重。",
	)
	parser.add_argument("--data-dir", default="data", help="输出目录，默认 data")
	parser.add_argument("--output", default="stock_data.csv", help="行情输出文件名")
	parser.add_argument("--stock-list-output", default="hs300_stock_list.csv", help="成分股列表输出文件名")
	parser.add_argument("--index-code", default="399300.SZ", help="沪深300指数代码，默认 399300.SZ")
	parser.add_argument(
		"--fallback-index-code",
		default="000300.SH",
		help="主指数代码取不到成分时使用的备用代码，默认 000300.SH",
	)
	parser.add_argument(
		"--component-lookback-months",
		type=int,
		default=8,
		help="成分股月度权重向前回看月数，默认 8",
	)
	parser.add_argument(
		"--adj",
		default="hfq",
		choices=["none", "qfq", "hfq"],
		help="复权口径：none/qfq/hfq。为兼容旧数据默认 hfq。",
	)
	parser.add_argument("--sleep", type=float, default=0.2, help="每只股票之间的暂停秒数")
	parser.add_argument("--retries", type=int, default=3, help="瞬时错误重试次数")
	parser.add_argument("--max-stocks", type=int, default=None, help="冒烟测试时只拉前 N 只")
	parser.add_argument(
		"--allow-partial",
		action="store_true",
		help="允许部分股票失败时仍写出 partial 文件。正式取数不建议开启。",
	)
	parser.add_argument(
		"--token-env",
		default="TUSHARE_TOKEN",
		help="读取 Tushare token 的环境变量名，默认 TUSHARE_TOKEN",
	)
	return parser.parse_args()


def normalize_ymd(value: str) -> str:
	ts = pd.to_datetime(value, errors="coerce")
	if pd.isna(ts):
		raise ValueError(f"日期格式无效: {value}")
	return ts.strftime("%Y%m%d")


def display_ymd(value: str) -> str:
	return pd.to_datetime(value, format="%Y%m%d").strftime("%Y-%m-%d")


def month_windows(end_date: str, lookback_months: int) -> Iterable[tuple[str, str]]:
	end = pd.to_datetime(end_date, format="%Y%m%d")
	for offset in range(max(lookback_months, 1)):
		base = end - pd.DateOffset(months=offset)
		start = base.replace(day=1)
		month_end = start + pd.offsets.MonthEnd(0)
		if offset == 0:
			month_end = min(month_end, end)
		yield start.strftime("%Y%m%d"), month_end.strftime("%Y%m%d")


def import_tushare():
	try:
		import tushare as ts  # type: ignore
	except ImportError as exc:
		raise RuntimeError(
			"未安装 tushare。请先在赛前取数环境执行 `pip install tushare`，"
			"或用项目环境同步 pyproject.toml。"
		) from exc
	return ts


def build_tushare_api(token_env: str):
	ts = import_tushare()
	token = os.environ.get(token_env)
	if token:
		ts.set_token(token)
		return ts, ts.pro_api(token)
	return ts, ts.pro_api()


def ts_code_to_legacy(ts_code: str) -> str:
	symbol, exchange = ts_code.split(".")
	return f"{exchange.lower()}.{symbol}"


def pure_stock_code(ts_code: str) -> str:
	return ts_code.split(".")[0].zfill(6)


def safe_query(callable_obj, retries: int, sleep_seconds: float, **kwargs) -> pd.DataFrame:
	last_exc: Exception | None = None
	for attempt in range(1, retries + 1):
		try:
			df = callable_obj(**kwargs)
			if df is None:
				return pd.DataFrame()
			return df
		except Exception as exc:  # noqa: BLE001 - third-party API raises mixed exception types
			last_exc = exc
			if attempt < retries:
				time.sleep(max(sleep_seconds, 0.1) * attempt)
	if last_exc is not None:
		raise last_exc
	return pd.DataFrame()


def load_hs300_components(
	pro,
	index_codes: list[str],
	component_date: str,
	lookback_months: int,
	retries: int,
	sleep_seconds: float,
) -> tuple[pd.DataFrame, str, str]:
	for start_date, end_date in month_windows(component_date, lookback_months):
		for index_code in index_codes:
			df = safe_query(
				pro.index_weight,
				retries=retries,
				sleep_seconds=sleep_seconds,
				index_code=index_code,
				start_date=start_date,
				end_date=end_date,
			)
			if df.empty:
				continue
			latest_trade_date = str(df["trade_date"].max())
			latest = df[df["trade_date"].astype(str) == latest_trade_date].copy()
			latest = latest.drop_duplicates(subset=["con_code"], keep="last")
			latest = latest.sort_values("con_code").reset_index(drop=True)
			return latest, index_code, latest_trade_date
	raise RuntimeError(
		f"未取到沪深300成分股。已尝试指数 {index_codes}，"
		f"并从 {display_ymd(component_date)} 向前回看 {lookback_months} 个月。"
	)


def attach_stock_names(pro, components: pd.DataFrame, retries: int, sleep_seconds: float) -> pd.DataFrame:
	basic = safe_query(
		pro.stock_basic,
		retries=retries,
		sleep_seconds=sleep_seconds,
		exchange="",
		list_status="L",
		fields="ts_code,symbol,name",
	)
	if basic.empty:
		components["code_name"] = ""
		return components
	name_map = basic.set_index("ts_code")["name"].to_dict()
	components["code_name"] = components["con_code"].map(name_map).fillna("")
	return components


def export_stock_list(
	components: pd.DataFrame,
	stock_list_path: Path,
	index_code: str,
	component_trade_date: str,
) -> pd.DataFrame:
	out = components.copy()
	out["updateDate"] = display_ymd(component_trade_date)
	out["code"] = out["con_code"].map(ts_code_to_legacy)
	out = out.rename(columns={"con_code": "ts_code"})
	columns = ["updateDate", "code", "code_name", "ts_code", "index_code", "trade_date", "weight"]
	out["index_code"] = index_code
	out = out[[col for col in columns if col in out.columns]]
	out.to_csv(stock_list_path, index=False, encoding="utf-8-sig")
	return out


def fetch_daily_basic_turnover(
	pro,
	ts_code: str,
	start_date: str,
	end_date: str,
	retries: int,
	sleep_seconds: float,
) -> pd.DataFrame:
	try:
		df = safe_query(
			pro.daily_basic,
			retries=retries,
			sleep_seconds=sleep_seconds,
			ts_code=ts_code,
			start_date=start_date,
			end_date=end_date,
			fields="ts_code,trade_date,turnover_rate",
		)
	except Exception as exc:  # noqa: BLE001
		print(f"  警告: daily_basic 换手率获取失败，将尝试 pro_bar 自带字段: {exc}")
		return pd.DataFrame()
	if df.empty:
		return df
	df["trade_date"] = df["trade_date"].astype(str)
	return df[["ts_code", "trade_date", "turnover_rate"]]


def fetch_price_bar(
	ts_module,
	pro,
	ts_code: str,
	start_date: str,
	end_date: str,
	adj: str,
	retries: int,
	sleep_seconds: float,
) -> pd.DataFrame:
	adj_value = None if adj == "none" else adj
	def _query(use_factors: bool) -> pd.DataFrame:
		kwargs = {
			"api": pro,
			"ts_code": ts_code,
			"adj": adj_value,
			"freq": "D",
			"start_date": start_date,
			"end_date": end_date,
		}
		if use_factors:
			kwargs["factors"] = ["tor"]
		return ts_module.pro_bar(**kwargs)

	last_exc: Exception | None = None
	for attempt in range(1, retries + 1):
		try:
			return _query(use_factors=True)
		except TypeError:
			return _query(use_factors=False)
		except Exception as exc:  # noqa: BLE001
			last_exc = exc
			if attempt < retries:
				time.sleep(max(sleep_seconds, 0.1) * attempt)
	if last_exc is not None:
		print(f"  警告: pro_bar factors 查询失败，将退回不带 factors: {last_exc}")
		for attempt in range(1, retries + 1):
			try:
				return _query(use_factors=False)
			except Exception as exc:  # noqa: BLE001
				last_exc = exc
				if attempt < retries:
					time.sleep(max(sleep_seconds, 0.1) * attempt)
	if last_exc is not None:
		raise last_exc
	return pd.DataFrame()


def to_legacy_schema(bars: pd.DataFrame, turnover: pd.DataFrame, ts_code: str) -> pd.DataFrame:
	if bars is None or bars.empty:
		return pd.DataFrame(columns=OUTPUT_COLUMNS)

	df = bars.copy()
	if "ts_code" not in df.columns:
		df["ts_code"] = ts_code
	df["trade_date"] = df["trade_date"].astype(str)

	if not turnover.empty:
		df = df.merge(turnover, on=["ts_code", "trade_date"], how="left", suffixes=("", "_basic"))
		if "turnover_rate_basic" in df.columns:
			df["turnover_rate"] = df["turnover_rate"].combine_first(df["turnover_rate_basic"])

	if "turnover_rate" not in df.columns:
		for candidate in ["tor", "turnover", "换手率"]:
			if candidate in df.columns:
				df["turnover_rate"] = df[candidate]
				break
	if "turnover_rate" not in df.columns:
		df["turnover_rate"] = pd.NA

	for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "turnover_rate"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	df = df.sort_values("trade_date").reset_index(drop=True)
	if "pre_close" not in df.columns or df["pre_close"].isna().all():
		if "change" in df.columns:
			df["pre_close"] = df["close"] - df["change"]
		else:
			df["pre_close"] = df["close"].shift(1)

	pre_close = df["pre_close"].replace(0, pd.NA)
	price_change = df["close"] - df["pre_close"]
	pct_chg = df["pct_chg"] if "pct_chg" in df.columns else price_change / pre_close * 100

	out = pd.DataFrame(
		{
			"股票代码": pure_stock_code(ts_code),
			"日期": pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d"),
			"开盘": df["open"],
			"收盘": df["close"],
			"最高": df["high"],
			"最低": df["low"],
			"成交量": df["vol"] * 100,
			"成交额": df["amount"] * 1000,
			"振幅": ((df["high"] - df["low"]) / pre_close * 100).round(2),
			"涨跌额": price_change.round(2),
			"换手率": df["turnover_rate"],
			"涨跌幅": pct_chg,
		}
	)
	return out[OUTPUT_COLUMNS]


def write_meta(meta_path: Path, payload: dict) -> None:
	meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
	args = parse_args()
	start_date = normalize_ymd(args.start_date)
	end_date = normalize_ymd(args.end_date)
	component_date = normalize_ymd(args.component_date or args.end_date)
	if start_date > end_date:
		raise ValueError("--start-date 不能晚于 --end-date")

	data_dir = Path(args.data_dir)
	data_dir.mkdir(parents=True, exist_ok=True)
	output_path = data_dir / args.output
	partial_path = output_path.with_suffix(".partial.csv")
	stock_list_path = data_dir / args.stock_list_output
	meta_path = data_dir / "tushare_fetch_meta.json"
	failed_path = data_dir / "failed_stocks.csv"

	ts_module, pro = build_tushare_api(args.token_env)
	print("Tushare API 初始化完成")
	print(f"目标行情区间: {display_ymd(start_date)} 至 {display_ymd(end_date)} | 复权: {args.adj}")

	index_codes = [args.index_code]
	if args.fallback_index_code and args.fallback_index_code not in index_codes:
		index_codes.append(args.fallback_index_code)

	components, used_index_code, component_trade_date = load_hs300_components(
		pro=pro,
		index_codes=index_codes,
		component_date=component_date,
		lookback_months=args.component_lookback_months,
		retries=args.retries,
		sleep_seconds=args.sleep,
	)
	components = attach_stock_names(pro, components, args.retries, args.sleep)
	stock_list = export_stock_list(components, stock_list_path, used_index_code, component_trade_date)
	ts_codes = stock_list["ts_code"].tolist()
	if args.max_stocks is not None:
		ts_codes = ts_codes[: args.max_stocks]
	print(
		f"成分股口径: {used_index_code} @ {display_ymd(component_trade_date)}，"
		f"股票数: {len(ts_codes)}"
	)

	all_frames: list[pd.DataFrame] = []
	failed: list[tuple[str, str]] = []
	name_map = stock_list.set_index("ts_code")["code_name"].to_dict()

	for idx, ts_code in enumerate(ts_codes, start=1):
		name = name_map.get(ts_code, "")
		print(f"[{idx}/{len(ts_codes)}] {ts_code} {name}")
		try:
			bars = fetch_price_bar(
				ts_module=ts_module,
				pro=pro,
				ts_code=ts_code,
				start_date=start_date,
				end_date=end_date,
				adj=args.adj,
				retries=args.retries,
				sleep_seconds=args.sleep,
			)
			turnover = fetch_daily_basic_turnover(
				pro=pro,
				ts_code=ts_code,
				start_date=start_date,
				end_date=end_date,
				retries=args.retries,
				sleep_seconds=args.sleep,
			)
			legacy = to_legacy_schema(bars, turnover, ts_code)
			if legacy.empty:
				raise RuntimeError("返回空行情")
			all_frames.append(legacy)
			print(f"  OK: {len(legacy)} 行")
		except Exception as exc:  # noqa: BLE001
			failed.append((ts_code, name))
			print(f"  失败: {exc}")
		time.sleep(max(args.sleep, 0))

	if not all_frames:
		raise RuntimeError("没有成功获取任何股票数据")

	result = pd.concat(all_frames, ignore_index=True)
	result = result.sort_values(["股票代码", "日期"]).reset_index(drop=True)

	target_path = output_path if (args.allow_partial or not failed) else partial_path
	result.to_csv(target_path, index=False, encoding="utf-8-sig")

	if failed:
		pd.DataFrame(failed, columns=["ts_code", "股票名称"]).to_csv(failed_path, index=False, encoding="utf-8-sig")

	write_meta(
		meta_path,
		{
			"source": "tushare",
			"interfaces": ["index_weight", "stock_basic", "pro_bar", "daily_basic"],
			"index_code": used_index_code,
			"component_trade_date": display_ymd(component_trade_date),
			"start_date": display_ymd(start_date),
			"end_date": display_ymd(end_date),
			"adjustment": args.adj,
			"rows": int(len(result)),
			"stocks": int(result["股票代码"].nunique()),
			"failed_stocks": len(failed),
			"output": str(target_path),
		},
	)

	print("=" * 60)
	print(f"输出行情: {target_path}，{len(result)} 行，{result['股票代码'].nunique()} 只股票")
	print(f"输出成分股: {stock_list_path}")
	print(f"输出元信息: {meta_path}")
	if failed:
		print(f"失败股票: {len(failed)}，详见 {failed_path}")
		if not args.allow_partial:
			raise RuntimeError("存在失败股票，已写出 partial 文件；正式取数请修复后重跑。")


if __name__ == "__main__":
	main()
