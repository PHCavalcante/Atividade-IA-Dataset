import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from data_load import load_dataset
from preprocessing import preprocess
from pathlib import Path

def train_and_evaluate(df=None, target_col="Conservation Status"):
    if df is None:
        df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor, ncols, ccols = preprocess(df, target_col)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs")
    }

    results = {}
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        cm = confusion_matrix(y_test, preds)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm
        }
        joblib.dump(model, f"artifacts/model_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

    metrics_df = pd.DataFrame([
        {"Model": name, "Accuracy": v["accuracy"], "F1": v["f1"], "Precision": v["precision"], "Recall": v["recall"]}
        for name, v in results.items()
    ])
    metrics_df.to_csv("artifacts/metrics.csv", index=False)

    plt.figure(figsize=(7,5))
    plt.bar(metrics_df["Model"], metrics_df["Accuracy"])
    plt.ylim(0, 1)
    plt.title("Comparação de Acurácias")
    plt.ylabel("Acurácia")
    for i, val in enumerate(metrics_df["Accuracy"]):
        plt.text(i, val + 0.01, f"{val:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("artifacts/accuracy_comparison.png")
    plt.close()

    for name, v in results.items():
        print("=== Modelo:", name, "===")
        print("Accuracy:", v["accuracy"])
        print("Confusion matrix:\n", v["confusion_matrix"])
        print("Classification report:\n", classification_report(y_test, v["model"].predict(X_test)))

    return results, metrics_df

if __name__ == "__main__":
    train_and_evaluate()
