import math
from typing import Tuple

import scipy
import cupy as cp
import cupyx.scipy.ndimage
from cupyx.scipy.ndimage._filters import median_filter as filter
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import torch
from torch import Tensor

__all__ = [
    "weighted_sum",
    "add_difference",
    "euclidean_add_difference",
    "multiply_difference",
    "ties_add_difference",
    "similarity_add_difference",
    "weighted_subtraction",
    "triple_sum",
    "sum_twice",
    "distribution_crossover",
    "tensor_sum",
    "top_k_tensor_sum",
    "rotate"
]

EPSILON = 1e-10  # Define a small constant EPSILON to prevent division by zero
CPU_FALLBACK = 0


def weighted_sum(a: Tensor, b: Tensor, alpha: float, sum_mode: str, diff_mode: str, **kwargs) -> Tensor:
    if sum_mode == 'cos_a':
        k = cosine_similarity_a(a, b, alpha, sim, sims)
        return a * (1 - k) + b * k
    elif sum_mode == 'cos_b':
        k = cosine_similarity_b(a, b, alpha, sim, sims)
        return a * (1 - k) + b * k
    else:
        return (1 - alpha) * a + alpha * b


def add_difference(a: Tensor, b: Tensor, c: Tensor, alpha: float, sum_mode: str, diff_mode: str, **kwargs) -> Tensor:
    if diff_mode == 'smooth+train':
        return a + alpha * smooth(train_difference(a, b, c))
    elif diff_mode == 'smooth':
        return a + alpha * smooth(b - c)
    elif diff_mode == 'train':
        return a + alpha * train_difference(a, b, c)
    else:
        return a + alpha * (b - c)


def euclidean_add_difference(
        a: Tensor, b: Tensor, c: Tensor, alpha: float, sum_mode: str, diff_mode: str, **kwargs
) -> Tensor:
    if diff_mode == 'smooth+train':
        a_diff = smooth(train_difference(b, a, c)).float()
        b_diff = smooth(train_difference(a, b, c)).float()
    elif diff_mode == 'smooth':
        a_diff = smooth(a - c).float()
        b_diff = smooth(b - c).float()
    elif diff_mode == 'train':
        a_diff = train_difference(b, a, c).float()
        b_diff = train_difference(a, b, c).float()
    else:
        a_diff = (a-c).float()
        b_diff = (b-c).float()
    a_diff = torch.nan_to_num(a_diff / torch.linalg.norm(a_diff))
    b_diff = torch.nan_to_num(b_diff / torch.linalg.norm(b_diff))

    distance = (1 - alpha) * a_diff ** 2 + alpha * b_diff ** 2
    distance = torch.sqrt(distance)
    sum_diff = weighted_sum(a.float(), b.float(), alpha, sum_mode, diff_mode) - c.float()
    distance = torch.copysign(distance, sum_diff)

    target_norm = torch.linalg.norm(sum_diff)
    if diff_mode in ['smooth+train', 'smooth']:
        return c + smooth(distance / torch.linalg.norm(distance) * target_norm)
    else:
        return c + distance / torch.linalg.norm(distance) * target_norm


def multiply_difference(
        a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs
) -> Tensor:
    if diff_mode == 'smooth+train':
        a_diff = smooth(train_difference(b, a, c)).float()
        b_diff = smooth(train_difference(a, b, c)).float()
    elif diff_mode == 'smooth':
        a_diff = smooth(a - c).float()
        b_diff = smooth(b - c).float()
    elif diff_mode == 'train':
        a_diff = train_difference(b, a, c).float()
        b_diff = train_difference(a, b, c).float()
    else:
        a_diff = (a-c).float()
        b_diff = (b-c).float()
    diff_a = torch.pow(torch.abs(a_diff), (1 - alpha))
    diff_b = torch.pow(torch.abs(b_diff), alpha)
    difference = torch.copysign(diff_a * diff_b, weighted_sum(a, b, beta, sum_mode, diff_mode) - c)
    if diff_mode in ['smooth+train', 'smooth']:
        return c + smooth(difference.to(c.dtype))
    else:
        return c + difference.to(c.dtype)


def ties_add_difference(
        a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs
) -> Tensor:
    deltas = []
    signs = []
    for m in [a, b]:
        deltas.append(filter_top_k(m - c, beta))
        signs.append(torch.sign(deltas[-1]))

    signs = torch.stack(signs, dim=0)
    final_sign = torch.sign(torch.sum(signs, dim=0))
    delta_filters = (signs == final_sign).float()

    res = torch.zeros_like(c, device=c.device)
    for delta_filter, delta in zip(delta_filters, deltas):
        res += delta_filter * delta

    param_count = torch.sum(delta_filters, dim=0)
    if diff_mode in ['smooth+train', 'smooth']:
        return c + alpha * smooth(torch.nan_to_num(res / param_count))
    else:
        return c + alpha * torch.nan_to_num(res / param_count)


def similarity_add_difference(
        a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs
) -> Tensor:
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = ((a * b / threshold ** 2) + 1) / 2
    similarity = torch.nan_to_num(similarity * beta, nan=beta)

    ab_diff = add_difference(a, b, c, alpha, sum_mode, diff_mode)
    ab_sum = weighted_sum(a, b, alpha / 2, sum_mode, diff_mode)
    return (1 - similarity) * ab_diff + similarity * ab_sum


def weighted_subtraction(
        a: Tensor, b: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs
) -> Tensor:
    # Adjust beta if both alpha and beta are 1.0 to avoid division by zero
    if alpha == 1.0 and beta == 1.0:
        beta -= EPSILON
    if diff_mode in ['smooth+train', 'smooth']:
        return smooth(a - alpha * beta * b) / (1 - alpha * beta)
    else:
        return (a - alpha * beta * b) / (1 - alpha * beta)


def sum_twice(
        a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs
) -> Tensor:
    return weighted_sum(weighted_sum(a, b, alpha, sum_mode, diff_mode), c, beta, 'normal', diff_mode)


def triple_sum(
        a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs
) -> Tensor:
    if sum_mode == 'cos_a':
        alpha = cosine_similarity_a(a, b, alpha, sim, sims)
    elif sum_mode == 'cos_b':
        alpha = cosine_similarity_b(a, b, alpha, sim, sims)
    return (1 - alpha - beta) * a + alpha * b + beta * c


def distribution_crossover(
        a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs
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


def tensor_sum(a: Tensor, b: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs) -> Tensor:
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


def top_k_tensor_sum(
        a: Tensor, b: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs
) -> Tensor:
    a_flat = torch.flatten(a)
    a_dist = torch.msort(a_flat)
    b_indices = torch.argsort(torch.flatten(b), stable=True)
    redist_indices = torch.argsort(b_indices)

    start_i, end_i, region_is_inverted = ratio_to_region(alpha, beta, torch.numel(a))
    start_top_k = kth_abs_value(a_dist, start_i)
    end_top_k = kth_abs_value(a_dist, end_i)

    indices_mask = (start_top_k < torch.abs(a_dist)) & (torch.abs(a_dist) <= end_top_k)
    if region_is_inverted:
        indices_mask = ~indices_mask
    indices_mask = torch.gather(indices_mask.float(), 0, redist_indices)

    a_redist = torch.gather(a_dist, 0, redist_indices)
    a_redist = (1 - indices_mask) * a_flat + indices_mask * a_redist
    return a_redist.reshape_as(a)


def rotate(a: Tensor, b: Tensor, alpha: float, beta: float, sum_mode: str, diff_mode: str, **kwargs):
    if alpha == 0 and beta == 0:
        return a

    is_conv = len(a.shape) == 4 and a.shape[-1] != 1
    if len(a.shape) == 0 or is_conv or torch.allclose(a.half(), b.half()):
        return weighted_sum(a, b, beta, sum_mode, diff_mode)

    if len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a_neurons = a.reshape(*shape_2d).double()
    b_neurons = b.reshape(*shape_2d).double()

    a_centroid = a_neurons.mean(0)
    b_centroid = b_neurons.mean(0)
    new_centroid = weighted_sum(a_centroid, b_centroid, alpha, sum_mode, diff_mode)
    if len(a.shape) == 1 or len(a.shape) == 2 and a.shape[0] == 1:
        return new_centroid.reshape_as(a)

    a_neurons -= a_centroid
    b_neurons -= b_centroid

    svd_driver = "gesvd" if a.is_cuda else None
    u, _, v_t = torch.linalg.svd(a_neurons.T @ b_neurons, driver=svd_driver)

    alpha_is_float = alpha != round(alpha)
    if alpha_is_float:
        # cancel reflection. without this, eigenvalues often have a complex component
        #   and then we can't obtain a valid dtype for the merge
        u[:, -1] /= torch.det(u) * torch.det(v_t)

    transform = rotation = u @ v_t
    if not torch.isfinite(u).all():
        raise ValueError(
            textwrap.dedent(
                f"""determinant error: {torch.det(rotation)}.
            This can happen when merging on the CPU with the "rotate" method.
            Consider merging on a cuda device, or try setting alpha to 1 for the problematic blocks.
            See this related discussion for more info: https://github.com/s1dlx/meh/pull/50#discussion_r1429469484"""
            )
        )

    if alpha_is_float:
        transform = fractional_matrix_power(transform, alpha)
    elif alpha == 0:
        transform = torch.eye(
            len(transform),
            dtype=transform.dtype,
            device=transform.device,
        )
    elif alpha != 1:
        transform = torch.linalg.matrix_power(transform, round(alpha))

    if beta != 0:
        # interpolate the relationship between the neurons
        a_neurons = weighted_sum(a_neurons, b_neurons @ rotation.T, beta, sum_mode, diff_mode)

    a_neurons @= transform
    a_neurons += new_centroid
    return a_neurons.reshape_as(a).to(a.dtype)


# utils

def kth_abs_value(a: Tensor, k: int) -> Tensor:
    if k <= 0:
        return torch.tensor(-1, device=a.device)
    else:
        return torch.kthvalue(torch.abs(a.float()), k)[0]


def ratio_to_region(width: float, offset: float, n: int) -> Tuple[int, int, bool]:
    if width < 0:
        offset += width
        width = -width
    width = min(width, 1)

    if offset < 0:
        offset = 1 + offset - int(offset)
    offset = math.fmod(offset, 1.0)

    if width + offset <= 1:
        inverted = False
        start = offset * n
        end = (width + offset) * n
    else:
        inverted = True
        start = (width + offset - 1) * n
        end = offset * n

    return round(start), round(end), inverted


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * torch.numel(a)), 1)
    k_value, _ = torch.kthvalue(torch.abs(a.flatten()).float(), k)
    top_k_filter = (torch.abs(a) >= k_value).float()
    return a * top_k_filter


def cosine_similarity_a(a: Tensor, b: Tensor, alpha: float, sim, sims) -> float:
    a_norm = torch.nn.functional.normalize(a.to(torch.float32), p=2, dim=0)
    b_norm = torch.nn.functional.normalize(b.to(torch.float32), p=2, dim=0)

    simab = sim(a_norm, b_norm)
    dot_product = torch.dot(a_norm.view(-1), b_norm.view(-1))
    magnitude_similarity = dot_product / (torch.norm(a) * torch.norm(b))
    combined_similarity = (simab + magnitude_similarity) / 2.0

    k = (combined_similarity - sims.min()) / (sims.max() - sims.min())
    k = k - alpha
    return 1 - k.clip(min=0.0, max=1.0)


def cosine_similarity_b(a: Tensor, b: Tensor, alpha: float, sim, sims) -> float:
    simab = sim(a.to(torch.float32), b.to(torch.float32))
    dot_product = torch.dot(a.view(-1).to(torch.float32), b.view(-1).to(torch.float32))
    magnitude_similarity = dot_product / (
            torch.norm(a.to(torch.float32)) * torch.norm(b.to(torch.float32))
    )
    combined_similarity = (simab + magnitude_similarity) / 2.0
    k = (combined_similarity - sims.min()) / (sims.max() - sims.min())
    k = k - alpha
    return 1 - k.clip(min=0.0, max=1.0)


def smooth(a: Tensor) -> Tensor:
    global CPU_FALLBACK
    if CPU_FALLBACK == 1:
        # Apply median filter to the weight differences
        filtered_diff = scipy.ndimage.median_filter(
            a.to(torch.float32).cpu().numpy(), size=3
        )
        # Apply Gaussian filter to the filtered differences
        filtered_diff = scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)

        # Add the filtered differences to the original weights
        return torch.tensor(filtered_diff)
    try:
        # CuPy hacks to avoid copying tensors to gpu.
        tx1 = a.to(torch.float32).cuda()
        dx = to_dlpack(tx1)
        cx = cp.from_dlpack(dx)
        # Apply median filter to the weight differences
        filtered_diff = cupyx.scipy.ndimage.median_filter(cx, size=3)

        # Apply Gaussian filter to the filtered differences
        filtered_diff = cupyx.scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)

        return from_dlpack(
            filtered_diff.toDlpack()
        ).cpu()  # Let's pretend this was on the cpu the whole time
    except Exception as err:
        print(
            f"CuPy not installed or CuPy dependencies were not installed properly. Error: {err}"
        )
        print(
            "Falling back to CPU based filtering. Expect a Significant Increase in merge times!"
        )
        CPU_FALLBACK = 1
        # Apply median filter to the weight differences
        filtered_diff = scipy.ndimage.median_filter(
            a.to(torch.float32).cpu().numpy(), size=3
        )
        # Apply Gaussian filter to the filtered differences
        filtered_diff = scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)
        return torch.tensor(filtered_diff)


def train_difference(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    diff_AB = a.float() - b.float()

    distance_A0 = torch.abs(b.float() - c.float())
    distance_A1 = torch.abs(b.float() - a.float())

    sum_distances = distance_A0 + distance_A1

    scale = torch.where(
        sum_distances != 0, distance_A1 / sum_distances, torch.tensor(0.0).float()
    )
    sign_scale = torch.sign(b.float() - c.float())
    scale = sign_scale * torch.abs(scale)

    new_diff = scale * torch.abs(diff_AB)
    return 1.8 * new_diff


def fractional_matrix_power(matrix: Tensor, power: float):
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    eigenvalues.pow_(power)
    result = eigenvectors @ torch.diag(eigenvalues) @ torch.linalg.inv(eigenvectors)
    return result.real.to(dtype=matrix.dtype)
