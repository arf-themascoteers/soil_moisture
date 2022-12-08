import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class SM_Dataset(Dataset):
    def __init__(self, is_train=True):
        sm = pd.read_csv("sm.csv").to_numpy()
        X = sm[:, 3:]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        y = sm[:, 1:2]

        scaler = MinMaxScaler()
        y = scaler.fit_transform(y)
        y = y.squeeze()

        self.X, X_test, self.y, y_test = train_test_split(X, y, random_state=1)

        if not is_train:
            self.X = X_test
            self.y = y_test

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]