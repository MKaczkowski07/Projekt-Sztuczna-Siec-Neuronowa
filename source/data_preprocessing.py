import pandas as pd
import numpy as np

class DataPreprocessing:
    def __init__(self, train_path: str, target_col: str = 'SalePrice', drop_cols: list | None = None):
        self.train_path = train_path
        self.target_col = target_col
        self.drop_cols = drop_cols if drop_cols is not None else ['Id']
        self.data = None
        self.y_max = None

    def load_data(self):
        self.data = pd.read_csv(self.train_path)
        existing_drop_cols = [col for col in self.drop_cols if col in self.data.columns]
        if existing_drop_cols:
            self.data.drop(columns=existing_drop_cols, inplace=True)

    def encode_categorical(self):
        self.data = pd.get_dummies(self.data, drop_first=True)

    def inverse_transform_target(self, y_scaled):
        if self.y_max is None:
            raise ValueError("Dane nie zostały jeszcze przeskalowane.")
        return y_scaled * self.y_max

    def get_processed_data(self, test_size: float = 0.2):
        self.load_data()
        self.encode_categorical()

        if self.target_col not in self.data.columns:
            raise ValueError(f"Kolumna docelowa '{self.target_col}' nie istnieje w danych.")

        np.random.seed(42)
        indices = np.random.permutation(len(self.data))
        val_size = int(len(self.data) * test_size)

        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_df = self.data.iloc[train_idx].copy()
        val_df = self.data.iloc[val_idx].copy()

        num_cols = train_df.select_dtypes(include=['number']).columns
        for col in num_cols:
            mean_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(mean_val)
            val_df[col] = val_df[col].fillna(mean_val)

        features = [col for col in train_df.columns if col != self.target_col]
        for col in features:
            mean_val = train_df[col].mean()
            std_val = train_df[col].std()
            if std_val != 0:
                train_df[col] = (train_df[col] - mean_val) / std_val
                val_df[col] = (val_df[col] - mean_val) / std_val

        self.y_max = train_df[self.target_col].max()
        train_df[self.target_col] = train_df[self.target_col] / self.y_max
        val_df[self.target_col] = val_df[self.target_col] / self.y_max

        x_train = train_df.drop(columns=[self.target_col]).values.astype(np.float32)
        y_train = train_df[self.target_col].values.astype(np.float32).reshape(-1, 1)

        x_val = val_df.drop(columns=[self.target_col]).values.astype(np.float32)
        y_val = val_df[self.target_col].values.astype(np.float32).reshape(-1, 1)

        return x_train.T, x_val.T, y_train.T, y_val.T