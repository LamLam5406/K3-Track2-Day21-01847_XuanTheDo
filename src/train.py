import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    if "target" not in df_train.columns or "target" not in df_eval.columns:
        raise ValueError("Both datasets must contain a 'target' column")

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    if list(X_train.columns) != list(X_eval.columns):
        raise ValueError("Training and evaluation feature columns must match")

    labels = [0, 1, 2]
    label_distribution = {
        str(label): float((y_train == label).mean()) for label in labels
    }
    for label, ratio in label_distribution.items():
        if ratio < 0.10:
            print(
                f"WARNING: class {label} only represents {ratio:.2%} "
                "of the training data"
            )

    with mlflow.start_run():

        mlflow.log_params(params)

        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for label, ratio in label_distribution.items():
            mlflow.log_metric(f"label_ratio_{label}", ratio)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "label_distribution": label_distribution,
                },
                f,
                indent=2,
            )

        matrix = confusion_matrix(y_eval, preds, labels=labels)
        precision, recall, _, support = precision_recall_fscore_support(
            y_eval,
            preds,
            labels=labels,
            zero_division=0,
        )
        report_lines = [
            "CONFUSION MATRIX (rows=true, columns=predicted)",
            "labels: 0 1 2",
            *[" ".join(map(str, row)) for row in matrix],
            "",
            "PER-CLASS METRICS",
            "class precision recall support",
        ]
        report_lines.extend(
            f"{label} {precision[index]:.4f} {recall[index]:.4f} "
            f"{int(support[index])}"
            for index, label in enumerate(labels)
        )
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
