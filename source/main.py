import numpy as np
from data_preprocessing import DataPreprocessing
from neural_network import NeuralNetwork


def main():
    print("Rozpoczynam przetwarzanie danych...")

    preprocessor = DataPreprocessing(train_path='../data/train.csv')
    X_train, X_val, y_train, y_val = preprocessor.get_processed_data(test_size=0.2)

    input_size = X_train.shape[0]
    hidden_size = 64
    learning_rate = 0.01
    epochs = 1000

    print(f"Dane wejściowe: {input_size} cech, {X_train.shape[1]} obserwacji uczących.")
    print("Inicjalizacja sieci neuronowej...")
    nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, learning_rate=learning_rate)

    print("Rozpoczynam uczenie...")
    nn.train(X_train, y_train, epochs=epochs, print_cost=True)

    print("\nEwaluacja końcowa:")
    y_train_pred = nn.predict(X_train)
    y_val_pred = nn.predict(X_val)

    train_loss = nn.compute_loss(y_train, y_train_pred)
    val_loss = nn.compute_loss(y_val, y_val_pred)

    train_loss_dollars = preprocessor.inverse_transform_target(np.sqrt(train_loss))
    val_loss_dollars = preprocessor.inverse_transform_target(np.sqrt(val_loss))

    print(f"Końcowy błąd MSE (skalowany) - Uczący: {train_loss:.6f}, Testowy: {val_loss:.6f}")
    print(f"Średni błąd w dolarach (RMSE) - Uczący: ~${train_loss_dollars:.2f}, Testowy: ~${val_loss_dollars:.2f}")


if __name__ == "__main__":
    main()