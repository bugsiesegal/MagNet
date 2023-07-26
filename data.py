import glob

import lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import Normalizer, StandardScaler


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data = []
        self.materials = {}

    def prepare_data(self):
        self.data = []
        for material in glob.glob("data" + "/*"):
            print(material)
            for file in glob.glob(material + "/*"):
                print(file)
                if file.endswith("Data_Curr.csv"):
                    current = pd.read_csv(file, header=None)
                elif file.endswith("Data_Time.csv"):
                    time = pd.read_csv(file, header=None)
                elif file.endswith("Data_Volt.csv"):
                    volt = pd.read_csv(file, header=None)
                elif file.endswith("Parameters.csv"):
                    parameters = np.tile(pd.read_csv(file, header=None).fillna(0).values, (10000, 1, 1)).swapaxes(0, 1)

            material_data = np.array([current.values, time.values, volt.values])
            material_data = material_data.swapaxes(0, 1)
            material_data = material_data.swapaxes(1, 2)
            material_data = np.concatenate([material_data, parameters], axis=2)

            self.data.append(material_data)
            self.materials[material] = torch.tensor(material_data).to(torch.float32)

        self.data = np.concatenate(self.data, axis=0)

        scaler = StandardScaler()
        # Since we need to normalize along the 3rd axis, we reshape the data to a 2D array.
        # And then inverse the reshape back to a 3D array.
        for i in range(len(self.data)):
            self.data[i] = scaler.fit_transform(self.data[i].reshape(-1, self.data[i].shape[-1])).reshape(
                self.data[i].shape)

    def setup(self, stage=None):
        self.train_data = self.data[:int(0.8 * len(self.data))]
        self.val_data = self.data[int(0.8 * len(self.data)):int(0.9 * len(self.data))]
        self.test_data = self.data[int(0.9 * len(self.data)):]

        self.train_data = torch.tensor(self.train_data).to(torch.float32)
        self.val_data = torch.tensor(self.val_data).to(torch.float32)
        self.test_data = torch.tensor(self.test_data).to(torch.float32)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
