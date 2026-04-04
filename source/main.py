import numpy as np
import pandas as pd
from data_preprocessing import DataPreprocessing
from neural_network import NeuralNetwork


def main():
    print("Wczytywanie i przygotowanie danych...")
    preprocessor = DataPreprocessing(train_path='../data/train.csv')
    X_train, X_val, y_train, y_val = preprocessor.get_processed_data(test_size=0.2)

    input_size = X_train.shape[0]
    learning_rate = 0.01
    epochs = 1000

    hidden_sizes_to_test = [16, 32, 64, 128]

    repeats = 3

    results = []

    print("\nRozpoczynam zautomatyzowane badanie wpływu liczby neuronów...\n")

    for hidden_size in hidden_sizes_to_test:
        print(f"| Testowanie liczby neuronów: {hidden_size} |")
        train_errors = []
        val_errors = []

        for i in range(repeats):
            nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, learning_rate=learning_rate)

            nn.train(X_train, y_train, epochs=epochs, print_cost=False)

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
            "Liczba neuronów": hidden_size,
            "Błąd Uczący (USD)": round(avg_train_rmse, 2),
            "Błąd Testowy (USD)": round(avg_val_rmse, 2),
            "Różnica (Przeuczenie)": round(avg_val_rmse - avg_train_rmse, 2)
        })

    print("PODSUMOWANIE WYNIKÓW")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()