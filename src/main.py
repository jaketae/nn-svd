import argparse
import os

import lightning.pytorch as pl

from .data import MNISTDataModule
from .model import MNISTLowRankModel, MNISTModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--no-symmetric", dest="symmetric", action="store_false")
    parser.set_defaults(symmetric=False)
    parser.add_argument("--orthonormal", action="store_true")
    parser.add_argument("--no-orthonormal", dest="orthonormal", action="store_false")
    parser.set_defaults(orthonormal=False)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    args = parser.parse_args()
    return args


def main(args) -> None:
    pl.seed_everything(args.seed)
    logger_path = "logs"
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
            symmetric=args.symmetric,
            orthonormal=args.orthonormal,
            alpha=args.alpha,
        )
        suffix = ""
        if args.symmetric:
            suffix += "_symmetric"
        if args.orthonormal:
            suffix += f"_orthonormal_{args.alpha}"
        logger_path = os.path.join(logger_path, f"{args.rank}" + suffix)
    else:
        # svd full rank to low rank
        model = MNISTModel.from_pretrained(
            checkpoint_path=args.checkpoint_path, rank=args.rank
        )
        logger_path = os.path.join(logger_path, f"{args.rank}_svd")

    datamodule = MNISTDataModule(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs if args.checkpoint_path is None else 0,
        logger=pl.loggers.CSVLogger(logger_path),
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    args = get_args()
    main(args)
