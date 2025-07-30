import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

client = MlflowClient()

def get_production_model_score(name, X_test, y_test):
    try:
        latest = client.get_latest_versions(name=name, stages=["Production"])[0]
        model_uri = f"models:/{name}/{latest.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        preds = model.predict(X_test)
        return f1_score(y_test, preds), latest.version
    except IndexError:
        return 0.0, None

def promote_model(name, run_id):
    new_uri = f"runs:/{run_id}/pipeline_model"
    result = mlflow.register_model(new_uri, name)
    version = result.version
    client.transition_model_version_stage(name, version, stage="Production", archive_existing_versions=True)
    return version