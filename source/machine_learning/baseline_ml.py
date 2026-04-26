import numpy as np
import pandas as pd
import sys
import os
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_networks.data_preprocessing_reg import DataPreprocessing
from neural_networks.data_preprocessing_cls import TitanicPreprocessing

def get_regression_baselines():
    print("--- BASELINE: REGRESJA (AMES HOUSING) ---")
    preprocessor = DataPreprocessing(train_path='../../data/housing/train.csv')
    X_train_T, X_val_T, y_train_T, y_val_T = preprocessor.get_processed_data(test_size=0.2)
    X_train, X_val = X_train_T.T, X_val_T.T
    y_train, y_val = y_train_T.T.ravel(), y_val_T.T.ravel()

    models = {
        'k-NN': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        train_rmse = preprocessor.inverse_transform_target(np.sqrt(mean_squared_error(y_train, pred_train)))
        val_rmse = preprocessor.inverse_transform_target(np.sqrt(mean_squared_error(y_val, pred_val)))

        print(
            f"{name}: Uczący: ${train_rmse:,.2f} | Testowy: ${val_rmse:,.2f} | Przeuczenie: ${val_rmse - train_rmse:,.2f}")


def get_classification_baselines():
    print("\n--- BASELINE: KLASYFIKACJA (TITANIC) ---")
    preprocessor = TitanicPreprocessing(data_path='../../data/titanic/train.csv')
    X_train_T, X_val_T, y_train_T, y_val_T = preprocessor.get_processed_data(test_size=0.2)
    X_train, X_val = X_train_T.T, X_val_T.T
    y_train, y_val = y_train_T.T.ravel(), y_val_T.T.ravel()

    models = {
        'k-NN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVC': SVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        train_acc = accuracy_score(y_train, pred_train) * 100
        val_acc = accuracy_score(y_val, pred_val) * 100

        print(f"{name}: Uczący: {train_acc:.2f}% | Testowy: {val_acc:.2f}% | Różnica: {train_acc - val_acc:.2f} p.p.")


if __name__ == "__main__":
    get_regression_baselines()
    get_classification_baselines()