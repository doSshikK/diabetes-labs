
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
REPORT_DIR = BASE_DIR / "reports" / "LAB1"


def main():
    df = pd.read_csv(DATA_PATH)
    target = df["target"]

    print("Анализ целевой переменной Diabetes")
    print("Количество записей:", len(df))
    print("Минимальное значение:", target.min())
    print("Максимальное значение:", target.max())
    print("Среднее значение:", round(target.mean(), 3))
    print("Медиана:", round(target.median(), 3))
    print("Пропуски:", target.isna().sum())
    print("Дубликаты строк:", df.duplicated().sum())

    # Ищем статистические выбросы методом IQR
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((target < lower) | (target > upper)).sum()

    print("Количество выбросов по IQR:", outliers)

    # Создаём два графика
    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(
        target,
        bins=25,
        kde=True,
        ax=axes[0],
    )
    axes[0].set_title("Target Distribution")
    axes[0].set_xlabel("Target value")

    sns.boxplot(
        y=target,
        ax=axes[1],
    )
    axes[1].set_title("Target Boxplot")
    axes[1].set_ylabel("Target value")

    plt.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        REPORT_DIR / "target_distribution.png",
        dpi=150,
    )
    plt.close()

    print("\nГрафик успешно сохранён!")


if __name__ == "__main__":
    main()
