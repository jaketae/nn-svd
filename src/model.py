import lightning.pytorch as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import Accuracy

from .low_rank import LowRankLinear, apply_low_rank


class MNISTModel(pl.LightningModule):
    def __init__(self, hidden_size: int = 64, lr=2e-4):
        super().__init__()
        self.lr = lr
        num_classes = 10
        self.l1 = nn.Linear(28 * 28, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

        self.val_accuracy = Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.test_accuracy = Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.dropout(F.relu(self.l1(x)))
        x = F.dropout(F.relu(self.l2(x)))
        x = self.l3(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def base_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return x, y, logits, loss

    def training_step(self, batch, batch_idx):
        _, _, logits, loss = self.base_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, logits, loss = self.base_step(batch, batch_idx)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy)

    def test_step(self, batch, batch_idx):
        x, y, logits, loss = self.base_step(batch, batch_idx)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.val_accuracy)

    @classmethod
    def from_pretrained(cls, rank: int, checkpoint_path: str):
        model = cls.load_from_checkpoint(checkpoint_path)
        apply_low_rank(model, "l1", rank, LowRankLinear)
        apply_low_rank(model, "l2", rank, LowRankLinear)
        apply_low_rank(model, "l3", rank, LowRankLinear)
        return model


class MNISTLowRankModel(MNISTModel):
    def __init__(
        self,
        rank: int,
        symmetric: bool = False,
        hidden_size: int = 64,
        lr: float = 2e-4,
    ):
        super().__init__(hidden_size=hidden_size, lr=lr)
        self.rank = rank
        self.l1 = LowRankLinear(28 * 28, hidden_size, self.rank)
        self.l2 = LowRankLinear(
            hidden_size, hidden_size, self.rank, symmetric=symmetric
        )
        self.l3 = nn.Linear(hidden_size, 10)
