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
import requests
import pandas as pd

# Sample input (match this to your model's expected columns)
data = pd.DataFrame([{
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}])

# Convert to dataframe_split format
payload = {
    "dataframe_split": {
        "columns": data.columns.tolist(),
        "data": data.values.tolist()
    }
}

# Send POST request to MLflow model server
response = requests.post(
    url="http://127.0.0.1:5002/invocations",
    headers={"Content-Type": "application/json"},
    json=payload
)

print("Prediction:", response.json())

# or the following curl command will work also


