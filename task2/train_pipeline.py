import pickle

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from datetime import datetime
import numpy as np

import optuna
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)


def update_data():
    df = pd.read_csv("../datasets/diamonds/diamonds.csv")

    update = pd.read_csv("synth_diamonds.csv")

    return pd.concat([df, update], axis=0).reset_index(drop=True)


def optimize(trial: optuna.Trial,
             numeric_features,
             x_train, y_train, x_val, y_val,
             cat_features) -> float:
    eta = trial.suggest_float(name="eta", low=0, high=1)
    gamma = trial.suggest_float(name="gamma", low=0, high=1000)
    max_depth = trial.suggest_int(name="max_depth", low=1, high=50)
    min_child_weight = trial.suggest_float(name="min_child_weight", low=1, high=100)
    lambda_ = trial.suggest_float(name="lambda", low=0, high=100)

    params = {
        "eta": eta,
        "gamma": gamma,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "lambda": lambda_
    }

    p2 = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), cat_features)
        ], remainder="passthrough")),
        ("impute", SimpleImputer(strategy="mean")),
        ("clf", XGBRegressor(**params))
    ])

    p2.fit(x_train, y_train)
    predicted = p2.predict(x_val)

    return metrics.mean_absolute_percentage_error(y_val, predicted)


def model_train_tune(x_train, y_train, x_val, y_val, numeric_features, cat_features):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize(trial, numeric_features, x_train, y_train, x_val, y_val, cat_features),
        n_trials=1000,
        n_jobs=-1
    )

    trials = sorted(study.best_trials, key=lambda t: t.values)

    best_trial = trials[0]
    best_hps = best_trial.params

    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), cat_features)
        ], remainder="passthrough")),
        ("impute", SimpleImputer(strategy="mean")),
        ("clf", XGBRegressor(**best_hps))
    ])


def data_prep(df):

    numeric_features = df.select_dtypes(include="number").columns.values
    cat_features = df.select_dtypes(exclude="number").columns.values

    # remove negative values for all numeric features
    for col in numeric_features:
        for index, row in df.iterrows():
            if row[col] < 0:
                df = df.drop(index)

    # clean up outliers
    for index, row in df.iterrows():
        if row["table"] > 90 or row["depth"] < 45 or row["z"] < 2:
            df.drop(index)

    targets = df["price"]
    del df["price"]

    if "price" in numeric_features:
        numeric_features = numeric_features[numeric_features != "price"]

    x_train, x_test, y_train, y_test = train_test_split(df, targets, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

    return x_train, y_train, x_val, y_val, numeric_features, cat_features, x_test, y_test


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    mean_absolute_percentage_error = metrics.mean_absolute_percentage_error(y_true, y_pred)

    result = ''
    result += 'explained_variance: ' + str(round(explained_variance, 4)) + '\n'
    result += 'mean_squared_log_error: ' + str(round(mean_squared_log_error, 4)) + '\n'
    result += 'r2: ' + str(round(r2, 4)) + '\n'
    result += 'MAE: ' + str(round(mean_absolute_error, 4)) + '\n'
    result += 'MAPE: ' + str(round(mean_absolute_percentage_error, 4)) + '\n'
    result += 'MSE: ' + str(round(mse, 4)) + '\n'
    result += 'RMSE: ' + str(round(np.sqrt(mse), 4))

    return result


def main():
    df = update_data()

    x_train, y_train, x_val, y_val, numeric_features, cat_features, x_test, y_test = data_prep(df)

    model = model_train_tune(x_train, y_train, x_val, y_val, numeric_features, cat_features)
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)

    # append evaluation metrics on a separate file
    with open("evaluation_metrics.txt", "a") as f:
        f.write(
            f"Time: {datetime.now()}\n"
            f"Results: {regression_results(y_test, predicted)}\n\n"
        )
        
    with open("latest_pipeline.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
