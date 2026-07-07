from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CUTOFF = pd.Timestamp("2026-06-26")


class CheckReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_comment_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )


def check_train_test_split(report: CheckReport) -> None:
    train_path = ROOT / "data" / "train.csv"
    test_path = ROOT / "data" / "test.csv"
    report.require(train_path.exists(), "data/train.csv 不存在")
    report.require(test_path.exists(), "data/test.csv 不存在")
    if not train_path.exists() or not test_path.exists():
        return

    train = pd.read_csv(train_path, usecols=["股票代码", "日期"], dtype={"股票代码": str})
    test = pd.read_csv(test_path, usecols=["股票代码", "日期"], dtype={"股票代码": str})
    train["日期"] = pd.to_datetime(train["日期"])
    test["日期"] = pd.to_datetime(test["日期"])

    train_max = train["日期"].max()
    test_min = test["日期"].min()

    report.require(
        train_max >= PRODUCTION_CUTOFF,
        f"data/train.csv 未更新到生产截止日 {PRODUCTION_CUTOFF.date()}，当前最大日期 {train_max.date()}",
    )
    if train_max < test_min:
        pass
    else:
        report.warn(
            False,
            "data/train.csv 是生产全量训练集；data/test.csv 仅作历史验证样本，predict.py 不读取 test.csv",
        )
    report.require(train["股票代码"].nunique() <= 300, "train 股票数超过沪深300候选范围")
    report.require(test["股票代码"].nunique() <= 300, "test 股票数超过沪深300候选范围")


def check_result_csv(report: CheckReport) -> None:
    result_path = ROOT / "output" / "result.csv"
    report.require(result_path.exists(), "output/result.csv 不存在，请先运行 predict.py")
    if not result_path.exists():
        return

    raw_bytes = result_path.read_bytes()
    report.require(b"\x00" not in raw_bytes, "result.csv 含有空字节，可能不是普通文本 CSV")
    try:
        raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        report.require(False, "result.csv 不是合法 UTF-8 编码")

    result = pd.read_csv(result_path, dtype={"stock_id": str})
    required = {"stock_id", "weight"}
    report.require(required.issubset(result.columns), "result.csv 必须包含 stock_id, weight")
    if not required.issubset(result.columns):
        return

    result["stock_id"] = result["stock_id"].astype(str).str.zfill(6)
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    report.require(result["stock_id"].str.fullmatch(r"\d{6}").all(), "result.csv 股票代码必须是 6 位数字")
    report.require(1 <= len(result) <= 5, "result.csv 股票数量必须在 1 到 5 之间")
    report.require(not result["stock_id"].duplicated().any(), "result.csv 存在重复股票代码")
    report.require(result["weight"].notna().all(), "result.csv 存在非法权重")
    report.require((result["weight"] >= 0).all(), "result.csv 权重不能为负")
    report.require(float(result["weight"].sum()) <= 1.0 + 1e-8, "result.csv 权重和不能超过 1")


def check_predict_does_not_read_test(report: CheckReport) -> None:
    predict_text = strip_comment_lines(read_text(ROOT / "code" / "src" / "predict.py")).lower()
    forbidden_patterns = ["test.csv", "data/test", "score_self.py"]
    for pattern in forbidden_patterns:
        report.require(pattern not in predict_text, f"predict.py 不应读取或引用 {pattern}")


def check_scaler_fit_scope(report: CheckReport) -> None:
    train_text = read_text(ROOT / "code" / "src" / "train.py")
    predict_text = read_text(ROOT / "code" / "src" / "predict.py")
    report.require(
        "scaler.fit_transform(train_data[features])" in train_text,
        "StandardScaler 应只在 train_data 上 fit_transform",
    )
    report.require(
        "val_data[features] = scaler.transform(val_data[features])" in train_text,
        "验证集应只 transform，不能 fit",
    )
    report.require(
        "processed[features] = scaler.transform(processed[features])" in predict_text,
        "预测集应只 transform，不能 fit",
    )


def check_negative_shift_usage(report: CheckReport) -> None:
    allowed_context = (
        "future_return",
        "open_t1",
        "open_t5",
        "label",
        "shift(-1)",
        "shift(-5)",
    )
    files = [
        ROOT / "code" / "src" / "tabular_ranker.py",
        ROOT / "code" / "src" / "train.py",
        ROOT / "code" / "src" / "utils.py",
    ]
    pattern = re.compile(r"shift\(\s*-\d+")
    for path in files:
        for line_no, line in enumerate(read_text(path).splitlines(), start=1):
            if pattern.search(line):
                context = line.strip()
                is_allowed = any(token in context for token in allowed_context)
                report.require(
                    is_allowed,
                    f"{path.name}:{line_no} 出现疑似未来特征 shift: {context}",
                )


def check_offline_reproducible(report: CheckReport) -> None:
    code_text = "\n".join(
        strip_comment_lines(read_text(path)).lower()
        for path in (ROOT / "code" / "src").glob("*.py")
        if path.name != "leakage_check.py"
    )
    forbidden = ["import requests", "akshare", "baostock", "tushare", "urllib", "selenium"]
    for token in forbidden:
        report.require(token not in code_text, f"训练/预测核心代码不应联网或依赖在线接口: {token}")

    report.require((ROOT / "train.sh").exists(), "根目录缺少 train.sh")
    report.require((ROOT / "test.sh").exists(), "根目录缺少 test.sh")
    report.require((ROOT / "Dockerfile").exists(), "根目录缺少 Dockerfile")


def main() -> int:
    report = CheckReport()
    check_train_test_split(report)
    check_result_csv(report)
    check_predict_does_not_read_test(report)
    check_scaler_fit_scope(report)
    check_negative_shift_usage(report)
    check_offline_reproducible(report)

    print("=== Leakage / Submission Check ===")
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"  - {error}")
        return 1

    print("PASS: 未发现明显未来函数或提交格式问题。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
