import pandas as pd

from config import config
from tabular_ranker import train_tabular_ranker


def main() -> None:
    raw_df = pd.read_csv(f"{config['data_path']}/train.csv", dtype={"股票代码": str})
    metrics = train_tabular_ranker(raw_df, config["output_dir"], config)
    print(metrics)


if __name__ == "__main__":
    main()
