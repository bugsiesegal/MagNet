import lightning as pl
import torch
from JAAEC import AmazingAutoEncoder
from lightning.pytorch.callbacks import ModelPruning
from lightning.pytorch.loggers import WandbLogger

import data


def train(embedding_size, batch_size, num_workers):
    datamodule = data.DataModule("data", batch_size=batch_size, num_workers=num_workers)

    autoencoder = AmazingAutoEncoder((-1, 10_000, 8), (1, embedding_size), learning_rate=1e-5,
                                     num_layers=2, num_heads=2)

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(logger=WandbLogger(project="JAAEC_MagNet"),
                         precision='bf16', benchmark=True, enable_checkpointing=True,
                         max_time="00:06:00:00")

    trainer.fit(autoencoder, datamodule=datamodule)


if __name__ == "__main__":
    train(16, 1, 16)
