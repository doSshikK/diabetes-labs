
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
REPORT_DIR = BASE_DIR / "reports" / "LAB1"


def main():
    # 1. Загружаем данные
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(DATA_PATH, index=False)

    print("Датасет Diabetes успешно загружен!")
    print("Количество пациентов:", df.shape[0])
    print("Количество признаков:", df.shape[1] - 1)
    print("Количество пропусков:", df.isnull().sum().sum())

    # 2. Вычисляем корреляции между признаками
    features = df.drop(columns=["target"])
    corr = features.corr()

    corr.to_csv(REPORT_DIR / "correlation_matrix.csv")

    # 3. Строим тепловую карту корреляций
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
    )

    plt.title("Diabetes: Feature Correlation Matrix")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "correlation_heatmap.png",
        dpi=120,
    )
    plt.close()

    # 4. Находим пару с самой сильной корреляцией
    pairs = []

    columns = list(corr.columns)

    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            value = corr.loc[first, second]
            pairs.append((abs(value), first, second, value))

    strongest = max(pairs)

    print("\nСамая сильная корреляция:")
    print("Признаки:", strongest[1], "и", strongest[2])
    print("Корреляция:", round(strongest[3], 4))

    print("\nГрафик успешно сохранён!")
    print("Папка:", REPORT_DIR)


if __name__ == "__main__":
    main()
