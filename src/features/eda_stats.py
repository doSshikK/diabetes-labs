
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
REPORT_DIR = BASE_DIR / "reports" / "LAB1"


def main():
    # Загружаем данные
    df = pd.read_csv(DATA_PATH)

    # Считаем основные статистики
    stats = df.describe().T.reset_index()
    stats = stats.rename(
        columns={
            "index": "feature",
            "50%": "median",
        }
    )

    # Добавляем информацию о пропусках и типах
    stats["missing"] = stats["feature"].map(
        df.isna().sum().to_dict()
    )
    stats["dtype"] = stats["feature"].map(
        df.dtypes.astype(str).to_dict()
    )

    # Добавляем ранее рассчитанные значения VIF
    vif = pd.read_csv(REPORT_DIR / "vif_stats.csv")
    stats = stats.merge(vif, on="feature", how="left")

    # Сохраняем результаты
    output = REPORT_DIR / "eda_stats.csv"
    stats.to_csv(output, index=False, float_format="%.6f")

    print("EDA Diabetes")
    print("Количество записей:", len(df))
    print("Количество признаков:", len(df.columns) - 1)
    print("Всего пропусков:", df.isna().sum().sum())

    print("\nСтатистика признаков:")
    print(
        stats[
            ["feature", "mean", "std", "missing", "vif"]
        ].to_string(index=False)
    )

    print("\nТаблица сохранена:", output)


if __name__ == "__main__":
    main()
