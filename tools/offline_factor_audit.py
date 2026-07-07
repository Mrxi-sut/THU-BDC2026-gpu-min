"""
Audit offline Tushare factor files before they are allowed into training.

The audit is intentionally conservative:
- factor files must be local CSVs;
- every file needs stock/date identity columns;
- event/financial factors need an explicit available_date or publish_datetime;
- duplicate stock-date rows are rejected;
- missingness and coverage are reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tushare_factor_registry import iter_plans


IDENTITY_COLUMNS = {"股票代码", "日期"}
VISIBILITY_COLUMNS = {"available_date", "publish_datetime"}
EVENT_OR_FINANCIAL_FAMILIES = {"event_attention", "quality_growth"}


class AuditReport:
	def __init__(self) -> None:
		self.rows: list[dict] = []
		self.errors: list[str] = []

	def add(self, **kwargs) -> None:
		self.rows.append(kwargs)

	def fail(self, message: str) -> None:
		self.errors.append(message)

	def write(self, output: Path | None) -> None:
		if output is None:
			return
		output.parent.mkdir(parents=True, exist_ok=True)
		pd.DataFrame(self.rows).to_csv(output, index=False, encoding="utf-8-sig")

	def raise_if_failed(self) -> None:
		if self.errors:
			raise SystemExit("离线因子审计失败:\n" + "\n".join(f"- {msg}" for msg in self.errors))


def parse_date_series(series: pd.Series, file_path: Path, column: str, report: AuditReport) -> pd.Series:
	parsed = pd.to_datetime(series, errors="coerce")
	if parsed.isna().any():
		report.fail(f"{file_path}: {column} 存在无法解析日期 {int(parsed.isna().sum())} 行")
	return parsed


def audit_file(file_path: Path, factor_family: str, report: AuditReport, cutoff_date: str | None) -> None:
	if not file_path.exists():
		report.add(file=str(file_path), status="missing", rows=0, coverage=0.0, max_missing_rate=None)
		report.fail(f"{file_path}: 文件不存在")
		return

	df = pd.read_csv(file_path, dtype={"股票代码": str})
	missing_identity = IDENTITY_COLUMNS - set(df.columns)
	if missing_identity:
		report.fail(f"{file_path}: 缺少身份列 {sorted(missing_identity)}")

	if factor_family in EVENT_OR_FINANCIAL_FAMILIES and not (VISIBILITY_COLUMNS & set(df.columns)):
		report.fail(f"{file_path}: 事件/财务因子必须包含 available_date 或 publish_datetime")

	if {"股票代码", "日期"} <= set(df.columns):
		df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
		df["_日期"] = parse_date_series(df["日期"], file_path, "日期", report)
		duplicate_count = int(df.duplicated(["股票代码", "_日期"]).sum())
		if duplicate_count:
			report.fail(f"{file_path}: 股票代码+日期 重复 {duplicate_count} 行")

		if cutoff_date is not None:
			cutoff = pd.to_datetime(cutoff_date)
			if (df["_日期"] > cutoff).any():
				report.fail(f"{file_path}: 日期超过 cutoff_date={cutoff_date}")
	else:
		duplicate_count = None

	visibility_col = None
	for col in ("available_date", "publish_datetime"):
		if col in df.columns:
			visibility_col = col
			parsed = parse_date_series(df[col], file_path, col, report)
			if "_日期" in df.columns and (parsed > df["_日期"]).any():
				report.fail(f"{file_path}: {col} 晚于因子日期，存在未来可见性风险")
			if cutoff_date is not None and (parsed > pd.to_datetime(cutoff_date)).any():
				report.fail(f"{file_path}: {col} 超过 cutoff_date={cutoff_date}")
			break

	feature_cols = [
		col
		for col in df.columns
		if col not in {"股票代码", "日期", "available_date", "publish_datetime", "_日期"}
	]
	max_missing_rate = float(df[feature_cols].isna().mean().max()) if feature_cols else 1.0
	coverage = 1.0 - max_missing_rate
	if feature_cols and coverage < 0.20:
		report.fail(f"{file_path}: 特征覆盖率过低，最低覆盖约 {coverage:.2%}")

	report.add(
		file=str(file_path),
		status="ok",
		rows=int(len(df)),
		stocks=int(df["股票代码"].nunique()) if "股票代码" in df.columns else None,
		feature_count=len(feature_cols),
		coverage=coverage,
		max_missing_rate=max_missing_rate,
		duplicate_stock_date=duplicate_count,
		visibility_col=visibility_col,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="审计 data/offline_factors 中的 Tushare 离线因子")
	parser.add_argument("--phase", default="5000", help="审计阶段，默认 5000")
	parser.add_argument("--cutoff-date", default=None, help="可选，禁止因子日期或可见日期超过该日期")
	parser.add_argument("--output", default="output/offline_factor_audit.csv", help="审计报告输出路径")
	parser.add_argument("--allow-missing", action="store_true", help="允许尚未生成的因子文件缺失")
	args = parser.parse_args()

	report = AuditReport()
	for plan in iter_plans(args.phase):
		file_path = Path(plan.factor_file)
		if args.allow_missing and not file_path.exists():
			report.add(file=str(file_path), status="missing_allowed", rows=0, coverage=0.0, max_missing_rate=None)
			continue
		audit_file(file_path, plan.factor_family, report, args.cutoff_date)

	report.write(Path(args.output) if args.output else None)
	report.raise_if_failed()
	print(f"离线因子审计通过，报告: {args.output}")


if __name__ == "__main__":
	main()
