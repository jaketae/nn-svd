import argparse
import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import MNISTDataModule
from .model import MNISTLowRankModel, MNISTModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    args = parser.parse_args()
    return args


def read_metrics(logger):
    logger_path = logger.log_dir
    metrics_path = os.path.join(logger_path, "metrics.csv")
    metrics = pd.read_csv(metrics_path)
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    sns.relplot(data=metrics, kind="line")
    plt.savefig(os.path.join(logger_path, "plot.png"))


def main(args):
    pl.seed_everything(args.seed)
    logger_path = "./src/logs"
    if args.rank == -1:
        # full rank from scratch
        model = MNISTModel(hidden_size=args.hidden_size, lr=args.lr)
        logger_path = os.path.join(logger_path, "full_rank")
    elif args.checkpoint_path is None:
        # low rank from scratch
        model = MNISTLowRankModel(
            hidden_size=args.hidden_size,
            lr=args.lr,
            rank=args.rank,
        )
        logger_path = os.path.join(logger_path, f"{args.rank}")
    else:
        # svd full rank to low rank
        model = MNISTModel.from_pretrained(
            checkpoint_path=args.checkpoint_path, rank=args.rank
        )
        logger_path = os.path.join(logger_path, f"{args.rank}_svd")

    datamodule = MNISTDataModule(batch_size=args.batch_size)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs if args.checkpoint_path is None else 0,
        logger=pl.loggers.CSVLogger(logger_path),
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    read_metrics(trainer.logger)


if __name__ == "__main__":
    args = get_args()
    main(args)
