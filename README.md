## NN SVD

This repository contains code for NN SVD. This work was completed as part of MATH 232: Advanced Linear Algebra.

## Abstract

> TODO

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

Experiment logs and checkpoints are saved in `src/logs`.

## License

Released under the [MIT License](LICENSE).
