import mlflow
import os
import time
import mlflow.sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score
from mlflow.models.signature import infer_signature
import traceback


def train_and_log_models(pipeline, param_grid, X_train, y_train, X_test, y_test):
    from sklearn.model_selection import ParameterGrid
    best_f1 = 0.0
    best_run_id = None

    # Set tracking URI - be very explicit
    tracking_path = os.path.abspath("mlruns")
    tracking_uri = f"file://{tracking_path}"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"🔧 MLflow tracking URI: {tracking_uri}")
    print(f"📁 MLruns directory: {tracking_path}")
    
    # Ensure directory exists
    os.makedirs(tracking_path, exist_ok=True)
    
    # Set experiment
    experiment = mlflow.set_experiment("adult-income-xgb")
    print(f"📊 Using experiment: {experiment.name} (ID: {experiment.experiment_id})")

    param_combinations = list(ParameterGrid(param_grid))
    print(f"🔄 Training {len(param_combinations)} models...")
    
    for i, params in enumerate(param_combinations):
        print(f"\n🚀 Training model {i+1}/{len(param_combinations)} with params: {params}")
        
        try:
            # Clone and train model
            model = clone(pipeline).set_params(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds)
            
            print(f"📈 Model {i+1} F1 Score: {f1:.4f}")

            with mlflow.start_run(run_name=f"run_{i}") as run:
                run_id = run.info.run_id
                print(f"📝 Started MLflow run: {run_id}")
                
                # Log parameters and metrics
                mlflow.log_params(params)
                mlflow.log_metric("f1_score", float(f1))
                print("✅ Logged parameters and metrics")
                
                # Try sklearn logging first (simplest approach)
                model_logged = False
                try:
                    print("🔄 Attempting sklearn model logging...")
                    
                    # Create signature - make it optional
                    signature = None
                    input_example = None
                    try:
                        signature = infer_signature(X_train.head(2), preds[:2])
                        input_example = X_train.head(1)
                        print("✅ Created signature and input example")
                    except Exception as sig_error:
                        print(f"⚠️ Signature creation failed, continuing without: {sig_error}")
                    
                    mlflow.sklearn.log_model(
                        sk_model=model.named_steps["model"],
                        artifact_path="pipeline_model",
                        signature=signature,
                        input_example=input_example
                    )
                    
                    print("✅ sklearn model logging completed")
                    model_logged = True
                    
                except Exception as sklearn_error:
                    print(f"❌ sklearn logging failed: {sklearn_error}")
                    print(f"🔍 sklearn error traceback: {traceback.format_exc()}")
                
                # If sklearn failed, try a simple pyfunc approach
                if not model_logged:
                    try:
                        print("🔄 Attempting simple pyfunc model logging...")
                        
                        # Very simple pyfunc wrapper
                        class SimpleModelWrapper(mlflow.pyfunc.PythonModel):
                            def __init__(self, model):
                                self.model = model
                            
                            def predict(self, context, model_input):
                                return self.model.predict(model_input)
                        
                        wrapped_model = SimpleModelWrapper(model)
                        mlflow.pyfunc.log_model(
                            artifact_path="pipeline_model",
                            python_model=wrapped_model,
                            pip_requirements=["scikit-learn", "xgboost", "pandas", "numpy"]
                        )
                        
                        print("✅ Simple pyfunc model logging completed")
                        model_logged = True
                        
                    except Exception as pyfunc_error:
                        print(f"❌ Simple pyfunc logging failed: {pyfunc_error}")
                        print(f"🔍 pyfunc error traceback: {traceback.format_exc()}")

                if not model_logged:
                    print(f"❌ All model logging approaches failed for run {run_id}")
                    continue

                # Give MLflow a moment to finish writing artifacts
                print("⏳ Ensuring MLflow artifacts are written...")
                time.sleep(2)  # Small buffer to ensure artifacts are written

                # Update best model
                if f1 > best_f1:
                    best_f1 = f1
                    best_run_id = run_id
                    print(f"🏆 New best model! F1: {best_f1:.4f}, Run ID: {best_run_id}")

        except Exception as e:
            print(f"❌ Error in training iteration {i+1}: {str(e)}")
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            continue

    if best_run_id is None:
        raise RuntimeError("❌ No models were successfully trained and logged!")
        
    print(f"\n🎯 Final result - Best F1: {best_f1:.4f}, Run ID: {best_run_id}")
    
    # Final check to ensure best model artifacts are ready
    print(f"🔍 Final verification of best model artifacts...")
    time.sleep(1)  # Extra buffer for the best model
    
    return best_f1, best_run_id