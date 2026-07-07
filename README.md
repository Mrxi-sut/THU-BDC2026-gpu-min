# THU-BDC2026 GPU-min

本项目是面向 2026 中国高校计算机大赛大数据挑战赛的沪深300股票组合预测方案。当前版本已经将赛前取数源升级为 Tushare，正式训练与预测只读取本地离线 CSV，不依赖联网、Token、MCP 或 Codex Skill。

## 1. 核心目标

比赛目标是从沪深300成分股中选出未来一周收益靠前的股票组合。评测口径为 T+1 开盘买入、T+5 开盘卖出，组合不超过 5 只股票，权重累计不超过 1。

当前主策略不是 Top1，也不是固定 Top5 等权，而是：

- 表格排序器为主信号；
- Transformer 序列模型为辅助信号；
- `selector_score` 统一调度多个候选分支；
- 动态仓位只输出 Top2 或 Top3。

常用仓位：

```text
Top2: 0.60 / 0.40
Top3: 0.40 / 0.35 / 0.25
```

## 2. 数据链路

赛前联网取数：

```powershell
$env:TUSHARE_TOKEN="your_token"
python get_stock_data.py --start-date 2024-04-09 --end-date 2026-06-26
python data/split_train_test.py
```

`get_stock_data.py` 使用 Tushare 接口：

- `index_weight`：获取沪深300成分与权重；
- `stock_basic`：补充股票名称；
- `pro_bar`：获取复权日线行情，默认 `hfq`；
- `daily_basic`：补充换手率。

输出仍兼容当前模型 schema：

```text
股票代码,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅
```

正式训练与预测：

- `data/stock_data.csv` 与 `data/train.csv` 当前为生产全量表：300 只股票，2024-04-09 至 2026-06-26，共 160212 行；
- `data/test.csv` 只保留为历史验证样本，不作为正式预测输入；
- `train.sh` 只读取 `data/train.csv`；
- `test.sh` 只读取 `data/train.csv` 和训练产物；
- 不读取 `data/test.csv`；
- 不联网；
- 不需要 Tushare token。

离线扩展因子为可选增强层，赛前预取、赛中断网复现：

```powershell
$env:TUSHARE_TOKEN="your_token"
python tools/fetch_tushare_factors.py --interfaces daily_basic moneyflow index_weight --max-trade-dates 20 --max-stocks 5
python tools/build_offline_factors.py --families daily_basic moneyflow index_weight
python tools/offline_factor_audit.py --phase 5000 --output output/offline_factor_audit.csv
```

通过审计后的文件位于 `data/offline_factors/`。`tabular_ranker.py` 会自动合并这些本地因子；如果文件不存在，则保持当前 baseline 行为。

## 3. 代码结构

- `get_stock_data.py`：Tushare 赛前离线取数脚本。
- `data/split_train_test.py`：按日期切分 `stock_data.csv` 为 `train.csv` 和 `test.csv`。
- `code/src/train.py`：训练 Transformer 与表格排序器。
- `code/src/tabular_ranker.py`：构造日线、横截面、市场状态特征并训练表格模型。
- `code/src/predict.py`：生成 `output/result.csv` 和 `output/ranking_debug.csv`。
- `code/src/positioning.py`：动态 Top2/Top3 仓位选择。
- `code/src/leakage_check.py`：未来函数、提交格式和联网依赖检查。
- `code/src/rolling_backtest.py`：滚动窗口回测。
- `tools/fetch_tushare_factors.py`：赛前 Tushare raw 层取数。
- `tools/build_offline_factors.py`：raw 层到离线因子层转换。
- `tools/offline_factor_audit.py`：离线因子 schema、覆盖率和可见性审计。

## 4. 输出约定

最终提交文件：

```text
output/result.csv
```

格式要求：

- 表头为 `stock_id,weight`；
- 不超过 5 只股票；
- 股票不重复；
- 权重非负；
- 权重和不超过 1；
- UTF-8 编码。

## 5. 本地运行

训练：

```bash
sh train.sh
```

预测：

```bash
sh test.sh
```

检查：

```bash
python code/src/leakage_check.py
```

滚动回测示例：

```bash
python code/src/rolling_backtest.py --windows 15 --step 10 --output output/rolling_backtest_selector_15.csv
```

## 6. 正式封包原则

Docker 复现环境必须满足：

- 固定随机种子后训练与预测结果可复现；
- 训练时间不超过 8 小时；
- 预测时间不超过 5 分钟；
- Docker 文件总大小不超过 10G；
- 复现训练和预测时不得联网；
- `readme.md` 说明算法、辅助数据来源、训练流程、推理流程。

Tushare 只作为赛前离线数据源。任何 token、在线调用配置、MCP 地址、Skill 依赖都不得进入正式运行链路。

官方镜像构建与导出：

```bash
docker buildx build --platform linux/amd64 --build-arg IMAGE_NAME=nvidia/cuda -t bdc2026 .
docker save -o 队伍名称.tar bdc2026:latest
```
