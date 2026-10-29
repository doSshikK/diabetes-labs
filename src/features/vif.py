
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
REPORT_DIR = BASE_DIR / "reports" / "LAB1"


def main():
    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=["target"])

    results = []

    for column in features.columns:
        other_features = features.drop(columns=[column])
        target_feature = features[column]

        model = LinearRegression()
        model.fit(other_features, target_feature)

        r2 = model.score(other_features, target_feature)

        if 1 - r2 < 1e-12:
            vif = float("inf")
        else:
            vif = 1 / (1 - r2)

        results.append({
            "feature": column,
            "vif": round(vif, 3),
        })

    vif_df = pd.DataFrame(results)
    vif_df = vif_df.sort_values("vif", ascending=False)

    print("VIF для признаков Diabetes:")
    print(vif_df.to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    vif_df.to_csv(REPORT_DIR / "vif_stats.csv", index=False)

    # Строим график VIF
    plt.figure(figsize=(10, 6))
    plt.barh(vif_df["feature"], vif_df["vif"])
    plt.axvline(5, color="red", linestyle="--", label="VIF = 5")
    plt.xlabel("VIF")
    plt.ylabel("Признак")
    plt.title("Diabetes: Multicollinearity (VIF)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(REPORT_DIR / "vif_chart.png", dpi=120)
    plt.close()

    print("\nРезультаты и график сохранены!")


if __name__ == "__main__":
    main()
