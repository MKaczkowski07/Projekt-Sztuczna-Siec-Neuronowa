import numpy as np
import pandas as pd
from data_preprocessing_reg import DataPreprocessing
from neural_network import NeuralNetwork


def test_parameter(param_name, values_to_test, train_path, repeats=3, **defaults):
    print(f"\n{'=' * 60}")
    print(f"Rozpoczynam zautomatyzowane badanie parametru: {param_name.upper()}")
    print(f"{'=' * 60}\n")

    results = []

    for val in values_to_test:
        print(f"| Testowanie {param_name}: {val} |")
        train_errors = []
        val_errors = []

        for i in range(repeats):
            current_params = defaults.copy()
            current_params[param_name] = val

            preprocessor = DataPreprocessing(train_path=train_path)
            X_train, X_val, y_train, y_val = preprocessor.get_processed_data(test_size=current_params['test_size'])
            input_size = X_train.shape[0]

            nn = NeuralNetwork(input_size=input_size,
                               hidden_size=current_params['hidden_size'],
                               learning_rate=current_params['learning_rate'])

            nn.train(X_train, y_train, epochs=current_params['epochs'], print_cost=False)

            y_train_pred = nn.predict(X_train)
            y_val_pred = nn.predict(X_val)

            train_loss = nn.compute_loss(y_train, y_train_pred)
            val_loss = nn.compute_loss(y_val, y_val_pred)

            train_rmse = preprocessor.inverse_transform_target(np.sqrt(train_loss))
            val_rmse = preprocessor.inverse_transform_target(np.sqrt(val_loss))

            train_errors.append(train_rmse)
            val_errors.append(val_rmse)

        avg_train_rmse = np.mean(train_errors)
        avg_val_rmse = np.mean(val_errors)

        print(
            f"Zakończono. Średni błąd (Uczący): ${avg_train_rmse:,.2f} | Średni błąd (Testowy): ${avg_val_rmse:,.2f}\n")

        results.append({
            "Wartość": val,
            "Błąd Uczący (USD)": round(avg_train_rmse, 2),
            "Błąd Testowy (USD)": round(avg_val_rmse, 2),
            "Różnica (Przeuczenie)": round(avg_val_rmse - avg_train_rmse, 2)
        })

    print(f"PODSUMOWANIE WYNIKÓW: {param_name.upper()}")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))


def main():
    train_path = '../../data/housing/train.csv'

    defaults = {
        'test_size': 0.2,
        'hidden_size': 32,
        'learning_rate': 0.01,
        'epochs': 1000
    }

    print("Wczytywanie konfiguracji testowej...")
    print("Każdy trening zostanie powtórzony 3 razy.")

    test_parameter('hidden_size', [16, 32, 64, 128], train_path, repeats=3, **defaults)
    test_parameter('test_size', [0.1, 0.2, 0.3, 0.4], train_path, repeats=3, **defaults)
    test_parameter('learning_rate', [0.1, 0.01, 0.001, 0.0001], train_path, repeats=3, **defaults)
    test_parameter('epochs', [100, 500, 1000, 2000], train_path, repeats=3, **defaults)


if __name__ == "__main__":
    main()