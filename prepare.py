import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]


class BikeDataset(Dataset):
    """Holds features and targets as float32 tensors, one row at a time."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def make_splits(seed: int = 2026):
    df = make_rentals(seed=seed)
    train_df, rest_df = train_test_split(df, test_size=0.4, random_state=seed)
    val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL,
            ),
        ]
    )


class Splits:
    def __init__(self, train_df, val_df, test_df):
        preprocessor = make_preprocessor()

        X_train = preprocessor.fit_transform(train_df[NUMERIC + CATEGORICAL])
        y_train = train_df[TARGET].to_numpy()

        X_val = preprocessor.transform(val_df[NUMERIC + CATEGORICAL])
        y_val = val_df[TARGET].to_numpy()

        X_test = preprocessor.transform(test_df[NUMERIC + CATEGORICAL])
        y_test = test_df[TARGET].to_numpy()

        self.train = BikeDataset(X_train, y_train)
        self.val = BikeDataset(X_val, y_val)
        self.test = BikeDataset(X_test, y_test)
        self.n_features = X_train.shape[1]
        self.preprocessor = preprocessor


def make_loaders(splits: Splits, seed: int = 2026, batch_size: int = 64):
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        splits.train, batch_size=batch_size, shuffle=True, generator=gen
    )
    val_loader = DataLoader(splits.val, batch_size=256, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader
