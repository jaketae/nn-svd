## NN SVD

This repository contains code for NN SVD. This work was completed as part of MATH 232: Advanced Linear Algebra.

## Abstract

> Neural networks have become the standard architecture in machine learning. A typical deep learning solution involves an over-parameterized neural network, which is trained on large amounts of data. However, the recent exponential growth trend in model size highlights the need for effective compression methods that reduce the model footprint. In this work, we propose NN-SVD, a simple yet effective method that employs singular value decomposition to neural network layer weights to compress the model. We show that \ourmethod is theoretically robust under mild Lipschitz assumptions and demonstrates strong empirical performance on a small fully-connected model trained on the MNIST dataset, as well as on GPT-2, a standard language model based on the transformer architecture.


## Quickstart

This project was developed and tested on an Ubuntu 20.04.5 LTS server with NVIDIA RTX 3090 GPUs on CUDA 11.7, using Python 3.8.17.

1. Clone this repository.

```
$ git clone https://github.com/jaketae/nn-svd.git
```

2. Create a Python virtual environment and install package requirements. Depending on the local platform, you may have to adjust package versions (e.g., PyTorch) as appropriate.

```
$ cd nn-svd
$ python -m venv venv
$ pip install -U pip wheel # update pip
$ pip install -r requirements.txt
```

3. Train and test the model.

```
$ CUDA_VISIBLE_DEVICES=0 python -m src.main
```

Experiment logs and checkpoints are saved in `logs`.

## Training

To train a model, run [`src/main.py`](src/main.py). The full list of supported arguments are shown below.

```
$ python -m src.main --help
usage: main.py [-h] [--seed SEED] [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
               [--hidden_size HIDDEN_SIZE] [--lr LR] [--rank RANK] [--symmetric] [--no-symmetric] [--orthonormal]
               [--no-orthonormal] [--alpha ALPHA] [--checkpoint_path CHECKPOINT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED
  --max_epochs MAX_EPOCHS
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --hidden_size HIDDEN_SIZE
  --lr LR
  --rank RANK
  --symmetric
  --no-symmetric
  --orthonormal
  --no-orthonormal
  --alpha ALPHA
  --checkpoint_path CHECKPOINT_PATH
```

For instance, to train a model with a rank constraint of 8 and symmetric regularization, run

```
$ CUDA_VISIBLE_DEVICES=0 python -m src.main --symmetric --rank 8
```

The script will report the hyperparameters and training log with the final test results under [`logs`](./logs/).

## NN-SVD

To perform NN-SVD on a trained checkpoint, run [`src/main.py`](src/main.py) with the `--checkpoint_path` flag.

For instance, to perform NN-SVD with rank parameter 16, run

```
$ CUDA_VISIBLE_DEVICES=0 python -m src.main --rank 16 --checkpoint_path src/logs/full_rank/lightning_logs/version_0/checkpoints/epoch\=9-step\=1080.ckpt
```

This command will load the checkpoint at the specified location, perform NN-SVD, then run evaluation on the compressed model (without any training). The final result will be saved under [`logs`](./logs/).

## License

Released under the [MIT License](LICENSE).
