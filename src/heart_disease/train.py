import argparse, json
from datetime import datetime, timezone
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from .config import CLEAN_DATA, FEATURES, FIGURE_DIR, METADATA_PATH, MODEL_PATH, RANDOM_STATE, RAW_DATA
from .data import load_raw
from .model import build_candidates

def evaluate_cv(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy":"accuracy", "precision":"precision", "recall":"recall", "roc_auc":"roc_auc"}
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {name: float(scores[f"test_{name}"].mean()) for name in scoring}

def main(enable_mlflow=True):
    df = pd.read_csv(CLEAN_DATA) if CLEAN_DATA.exists() else load_raw(RAW_DATA)
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)
    tracking = None
    if enable_mlflow:
        import mlflow
        mlflow.set_experiment("heart-disease-classification")
        tracking = mlflow
    results = {}
    for name, model in build_candidates().items():
        context = tracking.start_run(run_name=name) if tracking else None
        if context: context.__enter__()
        try:
            cv_metrics = evaluate_cv(model, X_train, y_train)
            model.fit(X_train, y_train)
            probability = model.predict_proba(X_test)[:, 1]
            prediction = (probability >= .5).astype(int)
            test_metrics = {"test_accuracy":accuracy_score(y_test,prediction), "test_precision":precision_score(y_test,prediction,zero_division=0), "test_recall":recall_score(y_test,prediction,zero_division=0), "test_roc_auc":roc_auc_score(y_test,probability)}
            metrics = {**{f"cv_{k}":v for k,v in cv_metrics.items()}, **{k:float(v) for k,v in test_metrics.items()}}
            results[name] = (metrics, model)
            if tracking:
                tracking.log_params({"model":name, "random_state":RANDOM_STATE, "cv_folds":5})
                tracking.log_metrics(metrics)
                tracking.sklearn.log_model(model, artifact_path="model")
        finally:
            if context: context.__exit__(None,None,None)
    best_name, (best_metrics, best_model) = max(results.items(), key=lambda item:item[1][0]["cv_roc_auc"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True); joblib.dump(best_model, MODEL_PATH)
    metadata = {"selected_model":best_name, "selection_metric":"mean 5-fold CV ROC-AUC", "metrics":best_metrics, "trained_at_utc":datetime.now(timezone.utc).isoformat(), "features":FEATURES, "threshold":0.5}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True); ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test); plt.tight_layout(); plt.savefig(FIGURE_DIR/"confusion_matrix.png",dpi=160); plt.close()
    print(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--no-mlflow",action="store_true"); args=parser.parse_args(); main(not args.no_mlflow)
