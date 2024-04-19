import abc

import dask.array as da
import torch
from torch import nn
from torch.nn import functional as F


def torch2dask(torch_tensor: torch.Tensor):
    return da.from_array(
        torch_tensor.detach().cpu().numpy(),
        chunks=torch_tensor.shape,
    )


def rgetattr(obj, attr_string: str):
    if "." not in attr_string:
        return getattr(obj, attr_string)
    result = obj
    for attr in attr_string.split("."):
        if not ("[" in attr and "]" in attr):
            result = getattr(result, attr)
        else:
            attr_name, indices_str = attr.split("[")
            indices = [int(index.strip("]")) for index in indices_str.split("][")]
            for index in indices:
                result = getattr(result, attr_name)[index]
    return result


def rsetattr(obj, attr_string: str, value) -> None:
    if "." not in attr_string:
        setattr(obj, attr_string, value)
        return
    attrs = attr_string.split(".")
    last_attr = attrs[-1]
    parent_obj = rgetattr(obj, ".".join(attrs[:-1]))
    if not ("[" in last_attr and "]" in last_attr):
        setattr(parent_obj, last_attr, value)
        return
    attr_name, indices_str = last_attr.split("[")
    indices = [int(index.strip("]")) for index in indices_str.split("][")]
    for index in indices[:-1]:
        parent_obj = getattr(parent_obj, attr_name)[index]
    setattr(getattr(parent_obj, attr_name)[indices[-1]], indices[-1], value)


def validate_linear_input(
    in_features: int, out_features: int, rank: int, symmetric: True
):
    if rank >= min(in_features, out_features):
        raise ValueError(
            "`rank` is greater or equal to both `in_features` and `out_features`"
        )
    if symmetric and in_features != out_features:
        raise ValueError("Cannot have `symmetric` when `in_features != out_features`")
    num_full_rank_parameters = in_features * out_features
    num_low_rank_parameters = rank * (in_features + out_features)
    if symmetric:
        num_low_rank_parameters //= 2
    if num_low_rank_parameters >= num_full_rank_parameters:
        raise ValueError("Low-rank setup has more parameters than full-rank setup.")


def singular_values(matrix) -> torch.Tensor:
    dask_array = torch2dask(matrix)
    _, sigma, _ = da.linalg.svd(dask_array)
    return torch.from_numpy(da.compute(sigma))


def factorize(
    matrix: torch.Tensor,
    rank: int,
    device: torch.device,
):
    dask_array = torch2dask(matrix)
    u, sigma, v_t = da.linalg.svd(dask_array)
    if rank > len(sigma):
        raise ValueError("`rank` is larger than the actual rank of `matrix`")
    sqrt_sigma = da.diag(da.sqrt(sigma[:rank]))
    u, v_t = da.compute(u[:, :rank] @ sqrt_sigma, sqrt_sigma @ v_t[:rank])
    return torch.from_numpy(u).to(device), torch.from_numpy(v_t).to(device)


def apply_low_rank(model, path: str, rank: int, cls) -> None:
    module = rgetattr(model, path)
    weight = module.weight
    if isinstance(module, nn.Linear):
        weight = weight.T
    factor1, factor2 = factorize(weight, rank, model.device)
    low_rank_module = cls(*weight.shape, rank).to(model.device)
    rsetattr(model, path, low_rank_module)
    if cls is LowRankLinear and module.bias is not None:
        low_rank_module.bias = module.bias.clone().detach()
    low_rank_module.factor1 = factor1
    low_rank_module.factor2 = factor2


class LowRankMixin(abc.ABC):
    @property
    @abc.abstractmethod
    def factor1(self):
        pass

    @property
    @abc.abstractmethod
    def factor2(self):
        pass


def compute_orthonormal_penalty(weight):
    symmetric1 = weight @ weight.T - torch.eye(weight.size(0), device=weight.device)
    symmetric2 = weight.T @ weight - torch.eye(weight.size(1), device=weight.device)
    return torch.square(torch.norm(symmetric1)) + torch.square(torch.norm(symmetric2))


class LowRankLinear(nn.Module, LowRankMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        symmetric: bool = False,
        bias: bool = True,
    ):
        validate_linear_input(in_features, out_features, rank, symmetric)
        super().__init__()
        self.symmetric = symmetric
        if not self.symmetric:
            self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    @property
    def bias(self):
        return self.linear2.bias

    @property
    def factor1(self):
        return self.linear1.weight

    @property
    def factor2(self):
        return self.linear2.weight

    @bias.setter
    def bias(self, bias):
        self.linear2.bias = nn.Parameter(bias)

    def set_bias(self, bias):
        self.linear2.bias = nn.Parameter(bias)

    @factor1.setter
    def factor1(self, factor):
        if self.symmetric:
            raise ValueError("Cannot set factor with `symmetric = True`")
        self.linear1.weight = nn.Parameter(factor.T)

    @factor2.setter
    def factor2(self, factor):
        self.linear2.weight = nn.Parameter(factor.T)

    def orthonormal_penalty(self):
        penalty = compute_orthonormal_penalty(self.linear2.weight)
        if self.symmetric:
            penalty = penalty * 2
        else:
            penalty = penalty + compute_orthonormal_penalty(self.linear1.weight)
        return penalty

    def forward(self, x):
        if not self.symmetric:
            return self.linear2(self.linear1(x))
        return self.linear2(F.linear(x, self.linear2.weight.T))
