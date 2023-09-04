import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import mlflow

import argparse
import glob
import os

def get_cvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided path: {path}")
    
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(args):
    df = get_cvs_df(path=args.training_data)

    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    
    return data


def train(data, reg_rate):
    LogisticRegression(C=1/reg_rate, solver="liblinear").\
        fit(data["train"]["X"], data["train"]["y"])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_data", dest="training_data", type=str)
    parser.add_argument("--test_size", dest="test_size", type=float, default=0.2)
    parser.add_argument("--random_state", dest="random_state", type=int, default=123)
    parser.add_argument("--reg_rate", dest="reg_rate", type=float, default=0.1)

    args = parser.parse_args()

    return args

def main(args):
    mlflow.autolog()

    data = split_data(args=args)
    train(data=data, reg_rate=args.reg_rate)


if __name__ == "__main__":

    print("\n\n")
    print("*"*60)

    args = parse_args()
    main(args=args)

    print("*"*60)
    print("\n\n")