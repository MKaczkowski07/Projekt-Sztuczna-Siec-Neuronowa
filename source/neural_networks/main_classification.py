import numpy as np
import pandas as pd
from data_preprocessing_cls import TitanicPreprocessing
from neural_network import NeuralNetwork


def test_parameter(param_name, values_to_test, train_path, repeats=3, **defaults):
    print(f"\n{'=' * 60}")
    print(f"Rozpoczynam zautomatyzowane badanie parametru: {param_name.upper()}")
    print(f"{'=' * 60}\n")

    results = []

    for val in values_to_test:
        print(f"| Testowanie {param_name}: {val} |")
        train_accuracies = []
        val_accuracies = []

        for i in range(repeats):
            current_params = defaults.copy()
            current_params[param_name] = val

            preprocessor = TitanicPreprocessing(data_path=train_path)
            X_train, X_val, y_train, y_val = preprocessor.get_processed_data(test_size=current_params['test_size'])
            input_size = X_train.shape[0]

            nn = NeuralNetwork(input_size=input_size,
                               hidden_size=current_params['hidden_size'],
                               learning_rate=current_params['learning_rate'],
                               task_type='classification')

            nn.train(X_train, y_train, epochs=current_params['epochs'], print_cost=False)

            y_train_pred_prob = nn.predict(X_train)
            y_val_pred_prob = nn.predict(X_val)

            y_train_pred_class = (y_train_pred_prob > 0.5).astype(int)
            y_val_pred_class = (y_val_pred_prob > 0.5).astype(int)

            train_acc = np.mean(y_train_pred_class == y_train) * 100
            val_acc = np.mean(y_val_pred_class == y_val) * 100

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        avg_train_acc = np.mean(train_accuracies)
        avg_val_acc = np.mean(val_accuracies)

        print(f"Zakończono. Dokładność (Uczący): {avg_train_acc:.2f}% | Dokładność (Testowy): {avg_val_acc:.2f}%\n")

        results.append({
            "Wartość": val,
            "Dokładność Uczący (%)": round(avg_train_acc, 2),
            "Dokładność Testowy (%)": round(avg_val_acc, 2),
            "Przeuczenie (Różnica %)": round(avg_train_acc - avg_val_acc, 2)
        })

    print(f"PODSUMOWANIE WYNIKÓW: {param_name.upper()}")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))


def main():
    train_path = '../../data/titanic/train.csv'

    defaults = {
        'test_size': 0.2,
        'hidden_size': 16,
        'learning_rate': 0.1,
        'epochs': 1000
    }

    print("Wczytywanie danych i inicjalizacja badań dla problemu KLASYFIKACJI...")
    print("Każdy trening zostanie powtórzony 3 razy.")

    test_parameter('hidden_size', [4, 8, 16, 32], train_path, repeats=3, **defaults)
    test_parameter('test_size', [0.1, 0.2, 0.3, 0.4], train_path, repeats=3, **defaults)

    test_parameter('learning_rate', [0.5, 0.1, 0.01, 0.001], train_path, repeats=3, **defaults)
    test_parameter('epochs', [100, 500, 1000, 2000], train_path, repeats=3, **defaults)


if __name__ == "__main__":
    main()