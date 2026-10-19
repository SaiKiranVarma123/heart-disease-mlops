import pandas as pd
from .config import COLUMNS, FEATURES

def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS, na_values="?")
    return clean_data(df)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df[COLUMNS].copy()
    for column in COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["target"] = (out["target"] > 0).astype(int)
    if out[FEATURES].dropna(how="all").empty:
        raise ValueError("No usable feature rows found")
    return out.drop_duplicates().reset_index(drop=True)
