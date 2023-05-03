"""
Multi-layer perceptron classifier for the LLHR task.
"""

from typing import Any, Dict, Literal

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy


class MLP_classifier(pl.LightningModule):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 10,
                 output_dim: int = 1,
                 output_method: Literal["classifier",
                                        "llhr"] = "classifier") -> None:
        super().__init__()

        # 0. Class attributes and save hyperparameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_method = output_method
        self.save_hyperparameters()

        # 1. Defining the model architecture.
        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_dim,
                      out_features=self.hidden_dim),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_dim,
                      out_features=self.hidden_dim),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_dim,
                      out_features=self.output_dim)
        )

        # 2. Deciding on regression on llhr or classification.
        if self.output_method == "classifier":
            self.model.append(nn.Sigmoid())

        # 3. Define accuracy metric.
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        # 4. Losses and accuracies.
        self.tl = []
        self.ta = []

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self,
                      batch: Any,
                      batch_idx: int) -> Dict[str, torch.Tensor]:
        # 1. Divide batch into input and target.
        x, y = batch
        # 2. Get predictions.
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        # 3. Calculate loss.
        if self.output_method == "classifier":
            loss = F.binary_cross_entropy(y_hat, y)
        elif self.output_method == "rolr":
            pass
        # 4. Log loss.
        self.log(name='train_loss',
                 value=loss)
        self.tl.append(loss.detach().numpy())
        # 5. Log accuracy.
        self.log(name='train_acc',
                 value=self.train_acc(y_hat, y))
        self.ta.append(self.train_acc(y_hat, y).detach().numpy())
        return {'loss': loss}

    def validation_step(self,
                        batch: Any,
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        # 1. Divide batch into input and target.
        x, y = batch
        # 2. Get predictions.
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        # 3. Calculate loss.
        if self.output_method == "classifier":
            loss = F.binary_cross_entropy(y_hat, y)
        elif self.output_method == "rolr":
            pass
        # 4. Log loss.
        self.log('val_loss',
                 value=loss)
        # 5. Log accuracy.
        self.log('val_acc',
                 value=self.val_acc(y_hat, y))
        return {'val_loss': loss}

    def test_step(self,
                  batch: Any,
                  batch_idx: int) -> Dict[str, torch.Tensor]:
        # 1. Divide batch into input and target.
        x, y = batch
        # 2. Get predictions.
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        # 3. Calculate loss.
        if self.output_method == "classifier":
            loss = F.binary_cross_entropy(y_hat, y)
        elif self.output_method == "rolr":
            pass
        # 4. Log loss.
        self.log('test_loss',
                 value=loss,
                 prog_bar=True)
        # 5. Log accuracy.
        self.log('test_acc',
                 value=self.test_acc(y_hat, y))
        return {'test_loss': loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return Adam(self.parameters(),
                    lr=1e-3)
