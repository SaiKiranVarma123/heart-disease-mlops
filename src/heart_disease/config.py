from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_DATA = ROOT / "data" / "processed" / "heart_disease_clean.csv"
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
FIGURE_DIR = ROOT / "reports" / "figures"
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
FEATURES = COLUMNS[:-1]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
RANDOM_STATE = 42
