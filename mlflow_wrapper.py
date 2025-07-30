import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score
from mlflow.models.signature import infer_signature

def train_and_log_models(pipeline, param_grid, X_train, y_train, X_test, y_test):
    from sklearn.model_selection import ParameterGrid
    best_f1 = 0.0
    best_run_id = None

    print(f"Training {len(list(ParameterGrid(param_grid)))} models...")
    for i, params in enumerate(ParameterGrid(param_grid)):
        model = clone(pipeline).set_params(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)

        with mlflow.start_run(run_name=f"run_{i}") as run:
            signature = infer_signature(X_test, preds)
            mlflow.log_params(params)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="pipeline_model",
                signature=signature,
                input_example=X_test.sample(3)
            )
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run.info.run_id

    return best_f1, best_run_id