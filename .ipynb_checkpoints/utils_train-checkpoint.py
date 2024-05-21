from __future__ import annotations

from typing import Any, Callable

import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from utils_data import collate_fn


class BaseLightning(pl.LightningModule):
    """Lightning wrapper defining dataset and training functions."""

    def __init__(
        self,
        dataset,
        model,
        num_workers: int = 0,
        batch_size: int = 32,
        loss_fn: Callable | None = None,
        additional_losses: dict | None = None,
        lr: float = 0.005,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
        monitor: str | None = None,
        callbacks: list[pl.callbacks.callback.Callback] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "dataset", "optimizer", "scheduler"])
        self.dataset = dataset
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.loss_fn = nn.functional.mse_loss if loss_fn is None else loss_fn

        self.additional_losses = additional_losses
        if additional_losses is None and self.loss_fn is nn.functional.mse_loss:
            # default to including MAE if we are also calculating MSE
            self.additional_losses = {"mae": nn.functional.l1_loss}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor = monitor

        self.callbacks = callbacks
        if optimizer is not None:
            self.lr=optimizer.state_dict()['param_groups'][0]['lr']
        else:    
            self.lr = lr


    def forward(self, *args, **kwargs):
        """Pass data through the model."""
        return self.model(*args, **kwargs)

    def get_losses(self, batch, prefix=""):
        """Get the losses."""
        y_pred = self.model(batch)
        loss = self.loss_fn(y_pred, batch.target)
        losses = {f"{prefix}_loss": loss}
        if self.additional_losses:
            for loss_name, loss_fn in self.additional_losses.items():
                losses[f"{prefix}_{loss_name}"] = loss_fn(y_pred, batch.target)
        return losses

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        losses = self.get_losses(batch, prefix="train")
        self.log_dict(losses, on_epoch=True, on_step=False, batch_size=self.batch_size)
        return losses["train_loss"]

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        losses = self.get_losses(batch, prefix="test")
        self.log_dict(losses, batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        losses = self.get_losses(batch, prefix="val")
        self.log_dict(losses, on_epoch=True, on_step=False, batch_size=self.batch_size)

    def configure_optimizers(self):
        """Configure optimizers."""
        opt = self.optimizer
        if self.optimizer is None:
            opt = optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = self.scheduler
        if self.scheduler is None:
            scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

        if self.monitor is not None:
            return {"optimizer": opt, "scheduler": scheduler, "monitor": self.monitor}
        return [opt], [scheduler]

    def setup(self, stage: str = None):
        """Construct datasets."""
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader."""
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the testing dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def configure_callbacks(self):
        """Configure callbacks for early stopping."""
        if self.callbacks is None:
            learning_rate = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
            early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=30)
            checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
            return [learning_rate, early_stop, checkpoint]
        return self.callbacks