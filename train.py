"""
train.py  – grid-search over PARAM_GRID, log each run to MLflow,
write best_run.txt, then exit 0 so docker-compose can continue.
"""

import warnings, os, json, time, mlflow, xgboost as xgb
from pathlib import Path
from itertools import product
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from fairlearn.datasets import fetch_adult

warnings.filterwarnings("ignore", message="UserWarning*")

# ─────────────────── config ───────────────────────────────────────────────
TRACKING_URI  = "file:/app/mlruns"
BEST_RUN_FILE = Path("/app/best_run.txt")
NUM_JOBS      = int(os.getenv("N_JOBS", 1))   # 1 = serial (safe), >1 = threaded
PARAM_GRID = {
    "model__max_depth": [3, 5, 7],
    "model__subsample": [0.6, 0.8],
    "model__eta": [0.01, 0.05],
}

# ─────────────────── data / pipeline ───────────────────────────────────────
X, y = fetch_adult(as_frame=True, return_X_y=True)
y    = LabelEncoder().fit_transform(y)

num_cols = ["age", "capital-gain", "capital-loss", "hours-per-week"]
pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"),
          [c for c in X.columns if c not in num_cols])])

base_pipe = Pipeline([
        ("prep", pre),
        ("model", xgb.XGBClassifier(
                     eval_metric="logloss",
                     n_estimators=200,
                     random_state=42))
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)

# ─────────────────── MLflow setup ──────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("adult-income-xgb")

def run_one(param_dict):
    """Train + log one parameter combo; return (f1, run_id)."""
    # Needed when executed in a Joblib worker
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("adult-income-xgb")

    with mlflow.start_run():
        model = base_pipe.set_params(**param_dict)
        model.fit(X_tr, y_tr)
        f1 = f1_score(y_te, model.predict(X_te))

        mlflow.log_params(param_dict)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, artifact_path="pipeline_model")

        return f1, mlflow.active_run().info.run_id

# ─────────────────── run grid search ───────────────────────────────────────
grid = [dict(zip(PARAM_GRID, v)) for v in product(*PARAM_GRID.values())]
print(f"🚀 training {len(grid)} configs  (jobs={NUM_JOBS})")

if NUM_JOBS == 1:
    results = [run_one(p) for p in grid]
else:
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=NUM_JOBS, backend="threading")(
        delayed(run_one)(p) for p in grid)

best_f1, best_run = max(results, key=lambda r: r[0])
print(f"🏆 best F1 = {best_f1:.4f}   run_id = {best_run}")

BEST_RUN_FILE.write_text(best_run + "\n")
os.sync()
time.sleep(1)
print("✅ best_run.txt written – exiting")
