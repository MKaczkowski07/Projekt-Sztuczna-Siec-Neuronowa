import numpy as np
import pandas as pd
import sys
import os
import warnings

# Ignorujemy ostrzeżenia bibliotek, żeby nie śmiecić w konsoli
warnings.filterwarnings('ignore')

# Sprytny trick: mówimy Pythonowi, żeby widział folder "neural_networks"
# Dzięki temu możemy zaimportować Waszą klasę do czyszczenia danych!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_networks.data_preprocessing_cls import TitanicPreprocessing

# Magia scikit-learn: importujemy 4 gotowe algorytmy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def test_ml_model(model_class, param_name, values_to_test, train_path, repeats=3, **fixed_params):
    print(f"\n{'=' * 65}")
    print(f"Badanie algorytmu: {model_class.__name__} | Parametr: {param_name.upper()}")
    print(f"{'=' * 65}\n")

    results = []

    for val in values_to_test:
        print(f"| Testowanie parametru {param_name}: {val} |")
        train_accuracies = []
        val_accuracies = []

        for _ in range(repeats):
            # 1. Wczytanie i czyszczenie danych Waszą metodą
            preprocessor = TitanicPreprocessing(data_path=train_path)
            X_train_nn, X_val_nn, y_train_nn, y_val_nn = preprocessor.get_processed_data(test_size=0.2)

            # 2. Dopasowanie wymiarów macierzy pod scikit-learn (Transpozycja i spłaszczenie)
            X_train, X_val = X_train_nn.T, X_val_nn.T
            y_train, y_val = y_train_nn.T.ravel(), y_val_nn.T.ravel()

            # 3. Tworzenie obiektu modelu z podmienionym parametrem
            current_params = fixed_params.copy()
            current_params[param_name] = val
            model = model_class(**current_params)

            # 4. Całe uczenie maszynowe dzieje się w tej jednej linijce!
            model.fit(X_train, y_train)

            # 5. Ewaluacja
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_acc = accuracy_score(y_train, y_train_pred) * 100
            val_acc = accuracy_score(y_val, y_val_pred) * 100

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        # Średnia z powtórzeń
        avg_train_acc = np.mean(train_accuracies)
        avg_val_acc = np.mean(val_accuracies)

        results.append({
            "Wartość": str(val),  # str() bo przy SVM testujemy tekst (np. 'linear')
            "Dokładność Uczący (%)": round(avg_train_acc, 2),
            "Dokładność Testowy (%)": round(avg_val_acc, 2),
            "Przeuczenie (Różnica)": round(avg_train_acc - avg_val_acc, 2)
        })

    # Wyświetlenie gotowej tabeli
    print(f"PODSUMOWANIE: {model_class.__name__} ({param_name.upper()})")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

def main():
    # Ścieżka względem nowego folderu machine_learning
    train_path = '../../data/titanic/train.csv'

    print("Rozpoczynam testowanie 4 algorytmów Uczenia Maszynowego dla Titanica...")

    # 1. K-Najbliższych Sąsiadów (Badamy liczbę sąsiadów)
    test_ml_model(KNeighborsClassifier, 'n_neighbors', [3, 5, 10, 20], train_path)

    # 2. Drzewa Decyzyjne (Badamy maksymalną głębokość drzewa)
    # None = brak limitu (drzewo rośnie aż nauczy się wszystkiego na pamięć)
    test_ml_model(DecisionTreeClassifier, 'max_depth', [3, 5, 10, None], train_path)

    # 3. Lasy Losowe (Badamy liczbę drzew w lesie)
    test_ml_model(RandomForestClassifier, 'n_estimators', [10, 50, 100, 200], train_path)

    # 4. Maszyny Wektorów Nośnych (Badamy rodzaj jądra krzywej)
    test_ml_model(SVC, 'kernel', ['linear', 'poly', 'rbf', 'sigmoid'], train_path)

if __name__ == "__main__":
    main()