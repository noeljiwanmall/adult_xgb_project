# 🧠 Adult Income Prediction with XGBoost

A complete, containerized machine learning pipeline that predicts whether an individual earns over $50K per year based on demographic features using the UCI Adult Income dataset.  
The project includes training, parameter search, model tracking, serving, and an interactive Streamlit UI for inference.

---

## 🚀 Architecture Overview

- **Trainer** (`train.py`) — Grid-search over hyperparameters, logs runs via MLflow, and writes best run ID to `best_run.txt`.
- **Model Server** (`serve_model.sh`) — Waits for the best run, then serves the model via MLflow's REST API.
- **UI** (`streamlit_app.py`) — Streamlit front-end for users to input features and get predictions.
- **Orchestration** — `docker-compose` builds and manages services: `trainer`, `model_server`, and `streamlit`.

```
                   ┌──────────┐
                   │ Trainer  │
                   └────┬─────┘
                        │ (logs best run ID)
                        ▼
               ┌────────────────────┐
               │   MLflow Server    │
               └────────┬───────────┘
                        │ (REST API)
                        ▼
               ┌────────────────────┐
               │   Streamlit UI     │
               └────────────────────┘
```

---

## 📦 Features

| Capability               | Description                                         |
|--------------------------|-----------------------------------------------------|
| 🔍 Hyperparameter Search | Grid search over multiple model configs            |
| 🧪 MLflow Tracking        | Logs params, metrics, models per run               |
| 📈 Auto-Best Selection    | Best run is picked based on F1 score               |
| 📦 Dockerized             | Easy to launch everything with Docker Compose      |
| 🌐 UI for Predictions     | Streamlit form interface → model predictions       |

---

## 🧰 Technologies

- Python 3.9
- XGBoost
- Scikit-learn
- MLflow
- Streamlit
- Docker + Docker Compose

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/noeljiwanmall/adult_xgb_project.git
cd adult_xgb_project
```

### 2. Launch the pipeline

```bash
docker compose up --build
```

> ⏳ This will:
> - Train and log multiple models
> - Identify the best run
> - Start MLflow model server
> - Launch Streamlit UI

---

## 🌐 Access the App

| Component        | URL                            |
|------------------|---------------------------------|
| Streamlit UI     | [http://localhost:18501](http://localhost:18501) |
| MLflow REST API  | `http://mlflow_server:5002/invocations` *(internal)* |

---

## 📁 Project Structure

```
adult_xgb_project/
├── train.py               # Trains multiple models, logs to MLflow
├── serve_model.sh         # Waits for best run & serves it via MLflow
├── streamlit_app.py       # Web UI for model input/output
├── docker-compose.yml     # Orchestrates the full stack
├── Dockerfile             # Builds shared image for all services
├── best_run.txt           # Contains run_id of the best performing model
├── mlruns/                # MLflow logs, metrics, models
├── model_utils.py         # Preprocessing & model setup
├── mlflow_wrapper.py      # Utilities for logging runs
├── registry_utils.py      # Model promotion logic
└── requirements.txt       # Python dependencies
```

---

## 🧪 Example Inference (manual)

```bash
curl -X POST http://localhost:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["age", "workclass", "fnlwgt", ...],
          "data": [[39, "State-gov", 77516, "Bachelors", 13, ...]]
        }
      }'
```

---

## 🔄 Customization

- Modify search space in `PARAM_GRID` inside `train.py`
- Set `N_JOBS` env var for parallelism (`docker-compose.yml`)
- Streamlit layout can be adjusted in `streamlit_app.py`

---

## 👨‍💻 Author

**Noel Jiwanmall**  
[github.com/noeljiwanmall](https://github.com/noeljiwanmall)

---

## 📜 License

This project is licensed under the MIT License.
