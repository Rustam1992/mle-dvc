# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import yaml
import os
import joblib

# обучение модели
def fit_model():

    # Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    # загрузите результат предыдущего шага: initial_data.csv
    data = pd.read_csv("data/initial_data.csv")

    # определяем target
    target_col = params["target_col"]
    index_col = params["index_col"]

    # разделяем признаки и целевую переменную
    X = data.drop(columns=[target_col, index_col])
    y = data[target_col]

    # категориальные признаки
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # числовые признаки
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop=params["one_hot_drop"],
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    # модель
    model = LogisticRegression(
        C=params["C"],
        penalty=params["penalty"]
    )

    # pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # обучение
    pipeline.fit(X, y)

    # создаём директорию для модели
    os.makedirs("models", exist_ok=True)

    # сохраняем обученную модель
    joblib.dump(
        pipeline,
        "models/fitted_model.pkl"
    )


if __name__ == "__main__":
    fit_model()