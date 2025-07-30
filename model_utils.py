import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from fairlearn.datasets import fetch_adult
from sklearn.preprocessing import LabelEncoder

def get_data():
    X, y = fetch_adult(as_frame=True, return_X_y=True)
    y = LabelEncoder().fit_transform(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_preprocessor(X):
    num_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_features = [col for col in X.columns if col not in num_features]
    return ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

def build_pipeline(preprocessor):
    return Pipeline([
        ('prep', preprocessor),
        ('model', xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False))
    ])

def evaluate_pipeline(pipeline, X, y):
    score = cross_val_score(pipeline, X, y, cv=5, scoring='f1').mean()
    print(f"Initial Cross-Validated F1 Score: {score:.4f}")
    return score