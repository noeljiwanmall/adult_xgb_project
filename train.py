import warnings
warnings.filterwarnings("ignore", message="UserWarning*")

import os
import subprocess
import mlflow
from model_utils import get_data, build_preprocessor, build_pipeline, evaluate_pipeline
from mlflow_wrapper import train_and_log_models
from registry_utils import get_production_model_score, promote_model

mlflow.set_experiment("adult-income-xgb")

def main():
    X_train, X_test, y_train, y_test = get_data()

    preprocessor = build_preprocessor(X_train)
    pipeline = build_pipeline(preprocessor)

    print("📊 Checking for current Production model...")
    try:
        prod_f1, prod_version = get_production_model_score("adult-income-xgb", X_test, y_test)
        has_production = True
        print(f"✔️  Found Production model (version {prod_version}) with F1: {prod_f1:.4f}")
    except Exception:
        prod_f1 = 0.0
        prod_version = None
        has_production = False
        print("ℹ️ No Production model found. Treating this as the first deployment.")

    print("🔁 Training and tuning new model...")
    param_grid = {
        'model__max_depth': [3, 5],
        'model__subsample': [0.1],
        'model__eta': [0.05]
    }

    new_f1, best_run_id = train_and_log_models(pipeline, param_grid, X_train, y_train, X_test, y_test)
    print(f"🏁 Best trained model F1: {new_f1:.4f} (run_id: {best_run_id})")

    if not has_production or new_f1 >= prod_f1:
        print("🚀 New model meets promotion criteria. Promoting...")
        new_version = promote_model("adult-income-xgb", best_run_id)
        print(f"✅ Model promoted to Production (version {new_version})")

        print("🖥️ Starting MLflow model server on port 5002...")
        subprocess.run([
            "mlflow", "models", "serve",
            "-m", "models:/adult-income-xgb/Production",
            "-p", "5002", "--no-conda"
        ])
    else:
        print("🛑 New model did not outperform Production. Keeping current model.")

if __name__ == "__main__":
    main()