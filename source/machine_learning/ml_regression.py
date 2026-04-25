import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_networks.data_preprocessing_reg import DataPreprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def run_grid_search(model_name, estimator, param_grid, X, y, preprocessor):
    print(f"\n{'=' * 80}")
    print(f"Optymalizacja siatkowa (GridSearchCV) dla: {model_name}")
    print(f"{'=' * 80}")

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        return_train_score=True,
        n_jobs=-1
    )

    print("Rozpoczęto przeszukiwanie siatki. To może chwilę potrwać...")
    grid_search.fit(X, y)
    print("Zakończono optymalizację.\n")

    results = pd.DataFrame(grid_search.cv_results_)

    results = results.sort_values(by='rank_test_score').head(5)

    output_data = []
    for index, row in results.iterrows():
        train_rmse_scaled = abs(row['mean_train_score'])
        val_rmse_scaled = abs(row['mean_test_score'])

        train_rmse_usd = preprocessor.inverse_transform_target(train_rmse_scaled)
        val_rmse_usd = preprocessor.inverse_transform_target(val_rmse_scaled)

        overfitting_usd = val_rmse_usd - train_rmse_usd

        row_dict = {}
        for param in param_grid.keys():
            row_dict[param] = row[f'param_{param}']

        row_dict['Uczący (USD)'] = round(train_rmse_usd, 2)
        row_dict['Walidacja (USD)'] = round(val_rmse_usd, 2)
        row_dict['Przeuczenie (USD)'] = round(overfitting_usd, 2)
        row_dict['Miejsce'] = row['rank_test_score']

        output_data.append(row_dict)

    df_output = pd.DataFrame(output_data)
    print(df_output.to_string(index=False) + "\n")


def main():
    train_path = '../../data/housing/train.csv'

    print("Pobieranie i przetwarzanie danych...")
    preprocessor = DataPreprocessing(train_path=train_path)

    X_train_T, X_val_T, y_train_T, y_val_T = preprocessor.get_processed_data(test_size=0.2)

    X_train = X_train_T.T
    y_train = y_train_T.T.ravel()

    print("\nRozpoczynam badanie modeli ML z wielowymiarową optymalizacją...")

    # 1. Metoda: k-Najbliższych Sąsiadów (k-NN)
    knn_grid = {
        'n_neighbors': [3, 5, 10, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1: Manhattan, 2: Euclidean
    }
    run_grid_search('k-NN', KNeighborsRegressor(), knn_grid, X_train, y_train, preprocessor)

    # 2. Metoda: Drzewo Decyzyjne
    dt_grid = {
        'max_depth': [3, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['squared_error', 'absolute_error']
    }
    run_grid_search('Decision Tree', DecisionTreeRegressor(random_state=42), dt_grid, X_train, y_train, preprocessor)

    # 3. Metoda: Random Forest
    rf_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    run_grid_search('Random Forest', RandomForestRegressor(random_state=42), rf_grid, X_train, y_train, preprocessor)

    # 4. Metoda: Support Vector Regressor (SVR)
    svr_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto']
    }
    run_grid_search('SVR', SVR(), svr_grid, X_train, y_train, preprocessor)


if __name__ == "__main__":
    main()