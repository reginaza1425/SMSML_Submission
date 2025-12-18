import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from automate_Nalendra import load_and_preprocess_data


def train_with_tuning():

    print("============= Hyperparameter Tuning =============")

    # 1. Load data dari automate (SATU PINTU)
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        "Titanic-raw.csv"
    )

    # 2. Set experiment MLflow
    mlflow.set_experiment("Titanic_Survival_Tuning")

    # 3. Kandidat hyperparameter
    param_grid = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": None},
    ]

    for params in param_grid:
        with mlflow.start_run():

            # --- Log parameter ---
            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])

            # --- Build Model ---
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )

            pipeline = Pipeline(steps=[
                ('classifier', model)
            ])

            # --- Training ---
            pipeline.fit(X_train, y_train)

            # --- Evaluation ---
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # --- Log metrics ---
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # --- Save model ---
            mlflow.sklearn.log_model(pipeline, "model_titanic")

            print(f"Done → n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
            print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train_with_tuning()
