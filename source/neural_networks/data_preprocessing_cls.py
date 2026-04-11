import pandas as pd
import numpy as np


class TitanicPreprocessing:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.target_col = 'Survived'  # Zmienna, którą będziemy przewidywać (0 - zmarł, 1 - przeżył)

    def load_and_clean_data(self):
        """Wczytuje dane i usuwa brudy."""
        self.data = pd.read_csv(self.data_path)

        # 1. Usunięcie pustych kolumn
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        self.data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # 2. Uzupełnienie brakujących danych
        # Wiek uzupełniamy medianą
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
        # Port zaokrętowania najczęstszą wartością (modą)
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
        # Uzupełniamy opłatę za bilet
        self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())

        # 3. Zamiana tekstu na liczby
        # Płeć
        self.data['Sex'] = self.data['Sex'].map({'female': 1, 'male': 0})

        # Port zaokrętowania (C, Q, S)
        self.data = pd.get_dummies(self.data, columns=['Embarked'], drop_first=True)

        self.data = self.data.astype(float)

    def get_processed_data(self, test_size: float = 0.2):
        self.load_and_clean_data()

        # Oddzielenie cech (X) od zmiennej docelowej (y)
        X = self.data.drop(columns=[self.target_col]).values
        y = self.data[self.target_col].values.reshape(-1, 1)

        # 4. Standaryzacja cech
        mean_val = np.mean(X, axis=0)
        std_val = np.std(X, axis=0)
        std_val[std_val == 0] = 1  # Unikamy dzielenia przez zero
        X = (X - mean_val) / std_val

        # 5. Losowy podział na zbiór uczący i testowy
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        val_size = int(len(X) * test_size)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        return X_train.T, X_val.T, y_train.T, y_val.T