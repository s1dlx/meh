from typing import Tuple, List

import torch
from torch import Tensor

__all__ = [
    "weighted_sum",
    "weighted_subtraction",
    "tensor_sum",
    "add_difference",
    "sum_twice",
    "triple_sum",
    "transmogrify_distribution",
    "similarity_add_difference",
    "ties_add_difference",
]


EPSILON = 1e-10  # Define a small constant EPSILON to prevent division by zero


def weighted_sum(a: Tensor, b: Tensor, alpha: float, **kwargs) -> Tensor:
    return (1 - alpha) * a + alpha * b


def weighted_subtraction(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    # Adjust beta if both alpha and beta are 1.0 to avoid division by zero
    if alpha == 1.0 and beta == 1.0:
        beta -= EPSILON

    return (a - alpha * beta * b) / (1 - alpha * beta)


def tensor_sum(a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs) -> Tensor:
    if alpha + beta <= 1:
        tt = a.clone()
        talphas = int(a.shape[0] * beta)
        talphae = int(a.shape[0] * (alpha + beta))
        tt[talphas:talphae] = b[talphas:talphae].clone()
    else:
        talphas = int(a.shape[0] * (alpha + beta - 1))
        talphae = int(a.shape[0] * beta)
        tt = b.clone()
        tt[talphas:talphae] = a[talphas:talphae].clone()
    return tt


def add_difference(a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs) -> Tensor:
    return a + alpha * (b - c)


def sum_twice(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    return (1 - beta) * ((1 - alpha) * a + alpha * b) + beta * c


def triple_sum(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    return (1 - alpha - beta) * a + alpha * b + beta * c


def transmogrify_distribution(a: Tensor, b: Tensor, **kwargs) -> Tensor:
    a_values = torch.msort(torch.flatten(a))
    b_indices = torch.argsort(torch.flatten(b), stable=True)
    redistributed_a_values = torch.gather(a_values, 0, torch.argsort(b_indices))
    return redistributed_a_values.reshape(a.shape)


def similarity_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = ((a * b / threshold**2) + 1) / 2
    similarity = torch.nan_to_num(similarity * beta, nan=beta)

    ab_diff = a + alpha * (b - c)
    ab_sum = (1 - alpha / 2) * a + (alpha / 2) * b
    return (1 - similarity) * ab_diff + similarity * ab_sum


def ties_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    signs: List[Tuple[Tensor, Tensor]] = []
    for m in [a, b]:
        topk = topk_filter(m - c, beta)
        signs.append((topk, torch.sign(topk)))

    sings_only = [m_signs for _, m_signs in signs]
    total_signs = torch.sign(torch.sum(torch.stack(sings_only, dim=0), dim=0))
    res = c
    for m_signs, m_delta in signs:
        m_filter = (m_signs == total_signs).float()
        res += alpha * m_filter * m_delta
    return res


def topk_filter(a: Tensor, k: float):
    top_k = max(int(k * torch.numel(a)), 1)
    a_value, a_index = torch.kthvalue(torch.abs(a.flatten()), top_k)
    res = a / (1 + torch.exp(32 - 32 * torch.abs(a) / a_value))
    return torch.nan_to_num(res)
