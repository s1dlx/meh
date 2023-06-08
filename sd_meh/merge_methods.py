import torch
from torch import Tensor

__all__ = [
    "weighted_sum",
    "weighted_subtraction",
    "tensor_sum",
    "add_difference",
    "sum_twice",
    "triple_sum",
    "euclidean_add_difference",
    "multiply_difference",
    "top_k_tensor_sum",
    "similarity_add_difference",
    "distribution_crossover",
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


def euclidean_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    distance = (a - c) ** 2 + alpha * (b - c) ** 2
    try:
        distance = torch.sqrt(distance)
    except RuntimeError:
        distance = torch.sqrt(distance.float()).half()
    distance = torch.copysign(distance, (a - c) + alpha * (b - c))

    a_norm = torch.linalg.norm(a - c)
    b_norm = torch.linalg.norm(b - c)
    target_norm = (1 - alpha / 2) * a_norm + (alpha / 2) * b_norm
    return c + distance / torch.linalg.norm(distance) * target_norm


def multiply_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    difference = torch.abs((a - c) * (b - c))
    try:
        difference = torch.sqrt(difference)
    except RuntimeError:
        difference = torch.sqrt(difference.float()).half()
    difference = torch.copysign(torch.sqrt(difference), a + b - 2 * c)
    return c + alpha * difference


def top_k_tensor_sum(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    a_flat = torch.flatten(a)
    a_dist = torch.msort(a_flat)
    b_indices = torch.argsort(torch.flatten(b), stable=True)

    a_numel = torch.numel(a)
    if alpha + beta <= 1:
        start_top_k = beta * a_numel
        end_top_k = (alpha + beta) * a_numel
        invert_mask = False
    else:
        start_top_k = (alpha + beta - 1) * a_numel
        end_top_k = beta * a_numel
        invert_mask = True

    start_top_k, _ = torch.kthvalue(torch.abs(a_dist).float(), max(1, int(start_top_k)))
    end_top_k, _ = torch.kthvalue(torch.abs(a_dist).float(), max(1, int(end_top_k)))

    redist_indices = torch.argsort(b_indices)
    indices_mask = (start_top_k <= torch.abs(a_dist)) & (torch.abs(a_dist) < end_top_k)
    if invert_mask:
        indices_mask = ~indices_mask
    indices_mask = torch.gather(indices_mask.float(), 0, redist_indices)

    a_redist = torch.gather(a_dist, 0, redist_indices)
    a_redist = (1 - indices_mask) * a_flat + indices_mask * a_redist
    return a_redist.reshape_as(a)


def similarity_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = ((a * b / threshold**2) + 1) / 2
    similarity = torch.nan_to_num(similarity * beta, nan=beta)

    ab_diff = a + alpha * (b - c)
    ab_sum = (1 - alpha / 2) * a + (alpha / 2) * b
    return (1 - similarity) * ab_diff + similarity * ab_sum


def distribution_crossover(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
):
    if a.shape == ():
        return alpha * a + (1 - alpha) * b

    c_indices = torch.argsort(torch.flatten(c))
    a_dist = torch.gather(torch.flatten(a), 0, c_indices)
    b_dist = torch.gather(torch.flatten(b), 0, c_indices)

    a_dft = torch.fft.rfft(a_dist.float())
    b_dft = torch.fft.rfft(b_dist.float())

    dft_filter = torch.arange(0, torch.numel(a_dft), device=a_dft.device).float()
    dft_filter /= torch.numel(a_dft)
    if beta > EPSILON:
        dft_filter = (dft_filter - alpha) / beta + 1 / 2
        dft_filter = torch.clamp(dft_filter, 0.0, 1.0)
    else:
        dft_filter = (dft_filter >= alpha).float()

    x_dft = (1 - dft_filter) * a_dft + dft_filter * b_dft
    x_dist = torch.fft.irfft(x_dft, a_dist.shape[0])
    x_values = torch.gather(x_dist, 0, torch.argsort(c_indices))
    return x_values.reshape_as(a)
