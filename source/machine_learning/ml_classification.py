import numpy as np
import pandas as pd
import sys
import os
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_networks.data_preprocessing_cls import TitanicPreprocessing


def tune_and_test_model(model_class, param_grid, train_path):
    print(f"\n{'=' * 85}")
    print(f"Optymalizacja hiperparametrów: {model_class.__name__}")
    print(f"{'=' * 85}")

    preprocessor = TitanicPreprocessing(data_path=train_path)
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = preprocessor.get_processed_data(test_size=0.2)

    X_train, X_test = X_train_nn.T, X_test_nn.T
    y_train, y_test = y_train_nn.T.ravel(), y_test_nn.T.ravel()

    base_model = model_class()

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-3,
        return_train_score = True
    )

    print("Przeszukiwanie siatki. Trenowanie i weryfikacja wszystkich kombinacji...\n")

    grid_search.fit(X_train, y_train)

    results_df = pd.DataFrame(grid_search.cv_results_)
    params_cols = [col for col in results_df.columns if col.startswith('param_')]
    cols_to_show = params_cols + ['mean_train_score', 'mean_test_score', 'rank_test_score']

    top_results = results_df[cols_to_show].sort_values(by='rank_test_score').head(5).copy()

    top_results['mean_train_score'] = round(top_results['mean_train_score'] * 100, 2)
    top_results['mean_test_score'] = round(top_results['mean_test_score'] * 100, 2)

    top_results['Przeuczenie (%)'] = round(top_results['mean_train_score'] - top_results['mean_test_score'], 2)

    rename_dict = {
        'mean_train_score': 'Uczący (%)',
        'mean_test_score': 'Walidacja (%)',
        'rank_test_score': 'Miejsce'
    }
    for col in params_cols:
        rename_dict[col] = col.replace('param_', '')

    top_results.rename(columns=rename_dict, inplace=True)

    kolumny = list(top_results.columns)
    kolumny.remove('Miejsce')
    kolumny.append('Miejsce')
    top_results = top_results[kolumny]

    print(f"--- 5 NAJLEPSZYCH KOMBINACJI ({model_class.__name__}) ---")
    print(top_results.to_string(index=False))
    print("-" * 85 + "\n")


def main():
    train_path = '../../data/titanic/train.csv'
    print("Rozpoczynam zautomatyzowane strojenie algorytmów...")

    # 1. Siatka dla KNN
    knn_grid = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    tune_and_test_model(KNeighborsClassifier, knn_grid, train_path)

    # 2. Siatka dla Drzewa Decyzyjnego
    tree_grid = {
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    tune_and_test_model(DecisionTreeClassifier, tree_grid, train_path)

    # 3. Siatka dla Lasu Losowego
    rf_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    tune_and_test_model(RandomForestClassifier, rf_grid, train_path)

    # 4. Siatka dla SVC
    svc_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    tune_and_test_model(SVC, svc_grid, train_path)


if __name__ == "__main__":
    main()