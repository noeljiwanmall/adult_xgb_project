# import mlflow
# import mlflow.sklearn
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from mlflow.models.signature import infer_signature
# import pandas as pd

# X, y = load_iris(return_X_y=True, as_frame=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pipeline = Pipeline([
#     ("scale", StandardScaler()),
#     ("clf", LogisticRegression())
# ])
# pipeline.fit(X_train, y_train)

# preds = pipeline.predict(X_test)

# with mlflow.start_run() as run:
#     mlflow.sklearn.log_model(
#         sk_model=pipeline,
#         artifact_path="test_pipeline",
#         input_example=X_test.head(),
#         signature=infer_signature(X_test, preds)
#     )
#     print(f"✅ Model logged to run ID: {run.info.run_id}")


# import mlflow.pyfunc
# model = mlflow.pyfunc.load_model("mlruns/0/43a4ba4e0ffd4e62bb309de50d33dd7d/artifacts/pipeline_model/")

import mlflow
print(mlflow.get_tracking_uri())