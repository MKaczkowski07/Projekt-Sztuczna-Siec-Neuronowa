import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_networks.data_preprocessing_reg import DataPreprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


def test_ml_parameter(model_name, model_class, param_name, values_to_test, train_path, **default_params):
    print(f"\n{'=' * 70}")
    print(f"Badanie metody: {model_name} | Parametr: {param_name}")
    print(f"{'=' * 70}")

    results = []

    preprocessor = DataPreprocessing(train_path=train_path)
    X_train_T, X_val_T, y_train_T, y_val_T = preprocessor.get_processed_data(test_size=0.2)

    X_train, X_val = X_train_T.T, X_val_T.T
    y_train, y_val = y_train_T.T.ravel(), y_val_T.T.ravel()

    for val in values_to_test:
        print(f"Testowanie {param_name} = {val}...", end=" ")

        current_params = default_params.copy()
        current_params[param_name] = val

        if model_name not in ['k-NN', 'SVR']:
            current_params['random_state'] = 42

        model = model_class(**current_params)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_loss = mean_squared_error(y_train, y_train_pred)
        val_loss = mean_squared_error(y_val, y_val_pred)

        train_rmse = preprocessor.inverse_transform_target(np.sqrt(train_loss))
        val_rmse = preprocessor.inverse_transform_target(np.sqrt(val_loss))

        print("Zakończono.")
        results.append({
            "Wartość": val,
            "Błąd Uczący (USD)": round(train_rmse, 2),
            "Błąd Testowy (USD)": round(val_rmse, 2),
            "Przeuczenie (Różnica)": round(val_rmse - train_rmse, 2)
        })

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False) + "\n")


def main():
    train_path = '../../data/housing/train.csv'

    print("Rozpoczynam badanie gotowych modeli ML dla Regresji (Ames Housing)...")
    print("Ze względu na gotowe algorytmy, uczymy tylko raz (są deterministyczne).")


    # 1. Metoda: Random Forest
    test_ml_parameter('Random Forest', RandomForestRegressor, 'n_estimators', [10, 50, 100, 200], train_path)
    # 2. Metoda: Drzewo Decyzyjne
    test_ml_parameter('Decision Tree', DecisionTreeRegressor, 'max_depth', [5, 10, 20, None], train_path)

    # 3. Metoda: k-Najbliższych Sąsiadów (k-NN)
    test_ml_parameter('k-NN', KNeighborsRegressor, 'n_neighbors', [3, 5, 10, 15], train_path)

    # 4. Metoda: Support Vector Regressor (SVR)
    test_ml_parameter('SVR', SVR, 'kernel', ['linear', 'poly', 'rbf', 'sigmoid'], train_path)


if __name__ == "__main__":
    main()