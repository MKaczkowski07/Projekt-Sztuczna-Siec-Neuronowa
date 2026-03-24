import pandas as pd
import numpy as np

class DataPreprocessing:
    def __init__(self, train_path: str, target_col: str = 'SalePrice', drop_cols: list | None = None):
        self.train_path = train_path
        self.target_col = target_col
        self.drop_cols = drop_cols if drop_cols is not None else ['Id']
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.train_path)
        existing_drop_cols = [col for col in self.drop_cols if col in self.data.columns]
        if existing_drop_cols:
            self.data.drop(columns=existing_drop_cols, inplace=True)

    def handle_missing_values(self):
        num_cols = self.data.select_dtypes(include=['number']).columns
        cat_cols = self.data.select_dtypes(exclude=['number']).columns

        for col in num_cols:
            mean_val = self.data[col].mean()
            self.data[col] = self.data[col].fillna(mean_val)

        for col in cat_cols:
            self.data[col] = self.data[col].fillna('None')

    def encode_categorical(self):
        self.data = pd.get_dummies(self.data, drop_first=True)

    def scale_features(self):
        features = [col for col in self.data.columns if col != self.target_col]

        for col in features:
            mean_val = self.data[col].mean()
            std_val = self.data[col].std()

            if std_val != 0:
                self.data[col] = (self.data[col] - mean_val) / std_val

    def get_processed_data(self, test_size: float = 0.2):
        """
        """
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical()
        self.scale_features()

        if self.target_col not in self.data.columns:
            raise ValueError(f"Kolumna docelowa '{self.target_col}' nie istnieje w danych.")

        x_data = self.data.drop(columns=[self.target_col]).values.astype(np.float32)
        y_data = self.data[self.target_col].values.astype(np.float32).reshape(-1, 1)

        np.random.seed(42)
        indices = np.random.permutation(len(x_data))
        val_size = int(len(x_data) * test_size)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        x_train, x_val = x_data[train_idx], x_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        return x_train, x_val, y_train, y_val