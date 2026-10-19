from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import CATEGORICAL, NUMERIC, RANDOM_STATE

def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])

def build_candidates():
    return {
        "logistic_regression": Pipeline([("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([("preprocess", build_preprocessor()), ("model", RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
    }
