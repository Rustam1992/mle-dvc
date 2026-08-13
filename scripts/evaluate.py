# scripts/evaluate.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

def evaluate_model():

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    model = joblib.load("models/fitted_model.pkl")

    data = pd.read_csv("data/initial_data.csv")

    X = data.drop(columns=[params["target_col"]])
    y = data[params["target_col"]]

    cv_strategy = StratifiedKFold(
        n_splits=params["n_splits"]
    )

    cv_res = cross_validate(
        model,
        X,
        y,
        cv=cv_strategy,
        scoring=params["metrics"],
        n_jobs=params["n_jobs"]
    )

    cv_res = {
        "fit_time": cv_res["fit_time"].mean(),
        "score_time": cv_res["score_time"].mean(),
        "test_f1": cv_res["test_f1"].mean(),
        "test_roc_auc": cv_res["test_roc_auc"].mean()
    }

    os.makedirs("cv_results", exist_ok=True)

    with open("cv_results/cv_res.json", "w") as fd:
        json.dump(cv_res, fd, indent=4)

if __name__ == '__main__':
    evaluate_model()