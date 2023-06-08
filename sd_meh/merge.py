import gc
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch
from tqdm import tqdm

from sd_meh import merge_methods
from sd_meh.model import SDModel
from sd_meh.rebasin import (
    apply_permutation,
    sdunet_permutation_spec,
    step_weights_and_bases,
    update_model_a,
    weight_matching,
)

MAX_TOKENS = 77
NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

KEY_POSITION_IDS = ".".join(
    [
        "cond_stage_model",
        "transformer",
        "text_model",
        "embeddings",
        "position_ids",
    ]
)


NAI_KEYS = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def fix_clip(model: Dict) -> Dict:
    if KEY_POSITION_IDS in model.keys():
        model[KEY_POSITION_IDS] = torch.tensor(
            [list(range(MAX_TOKENS))], dtype=torch.int64
        )

    return model


def fix_key(model: Dict, key: str) -> Dict:
    for nk in NAI_KEYS:
        if key.startswith(nk):
            model[key.replace(nk, NAI_KEYS[nk])] = model[key]
            del model[key]

    return model


# https://github.com/j4ded/sdweb-merge-block-weighted-gui/blob/master/scripts/mbw/merge_block_weighted.py#L115
def fix_model(model: Dict) -> Dict:
    for k in model.keys():
        model = fix_key(model, k)
    return fix_clip(model)


def load_sd_model(model: os.PathLike | str, device: str = "cpu") -> Dict:
    if isinstance(model, str):
        model = Path(model)

    return SDModel(model, device).load_model()


def prune_sd_model(model: Dict) -> Dict:
    keys = list(model.keys())
    for k in keys:
        if (
            not k.startswith("model.diffusion_model.")
            and not k.startswith("first_stage_model.")
            and not k.startswith("cond_stage_model.")
        ):
            del model[k]
    return model


def restore_sd_model(original_model: Dict, merged_model: Dict) -> Dict:
    for k in original_model:
        if k not in merged_model:
            merged_model[k] = original_model[k]
    return merged_model


def log_vram(txt=""):
    alloc = torch.cuda.memory_allocated(0)
    print(f"{txt}: {alloc*1e-9:5.3f}")


def load_thetas(
    models: Dict[str, os.PathLike | str],
    prune: bool,
    device: str,
    precision: int,
) -> Dict:
    log_vram("before loading models")
    if prune:
        thetas = {k: prune_sd_model(load_sd_model(m, "cpu")) for k, m in models.items()}
    else:
        thetas = {k: load_sd_model(m, device) for k, m in models.items()}

    if device == "cuda":
        for model_key, model in thetas.items():
            for key, block in model.items():
                if precision == 16:
                    thetas[model_key].update({key: block.to(device).half()})
                else:
                    thetas[model_key].update({key: block.to(device)})

    log_vram("models loaded")
    return thetas


def merge_models(
    models: Dict[str, os.PathLike | str],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
    re_basin: bool = False,
    iterations: int = 1,
    device: str = "cpu",
    prune: bool = False,
) -> Dict:
    thetas = load_thetas(models, prune, device, precision)

    if re_basin:
        merged = rebasin_merge(
            thetas,
            weights,
            bases,
            merge_mode,
            precision=precision,
            weights_clip=False,
            iterations=iterations,
            device=device,
        )
        # clip only after the last re-basin iteration
        if weights_clip:
            merged = clip_weights(thetas, merged)
    else:
        merged = simple_merge(
            thetas,
            weights,
            bases,
            merge_mode,
            precision=precision,
            weights_clip=weights_clip,
        )

    return un_prune_model(merged, thetas, models, device, prune, precision)


def un_prune_model(
    merged: Dict,
    thetas: Dict,
    models: Dict,
    device: str,
    prune: bool,
    precision: int,
) -> Dict:
    if prune:
        del thetas
        gc.collect()
        log_vram("remove thetas")
        original_a = load_sd_model(models["model_a"], device)
        for key in tqdm(original_a.keys(), desc="un-prune model a"):
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in merged.keys():
                merged.update({key: original_a[key]})
                if precision == 16:
                    merged.update({key: merged[key].half()})
        del original_a
        gc.collect()
        log_vram("remove original_a")
        original_b = load_sd_model(models["model_b"], device)
        for key in tqdm(original_b.keys(), desc="un-prune model b"):
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in merged.keys():
                merged.update({key: original_b[key]})
                if precision == 16:
                    merged.update({key: merged[key].half()})
        del original_b

    return fix_model(merged)


def simple_merge(
    thetas: Dict[str, Dict],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
) -> Dict:
    for key in tqdm(thetas["model_a"].keys(), desc="stage 1"):
        with merge_key_context(
            key, thetas, weights, bases, merge_mode, precision, weights_clip
        ) as result:
            if result is not None:
                thetas["model_a"].update({key: result.detach().clone()})

    log_vram("after stage 1")

    for key in tqdm(thetas["model_b"].keys(), desc="stage 2"):
        if KEY_POSITION_IDS in key:
            continue
        if "model" in key and key not in thetas["model_a"].keys():
            thetas["model_a"].update({key: thetas["model_b"][key]})
            if precision == 16:
                thetas["model_a"].update({key: thetas["model_a"][key].half()})

    log_vram("after stage 2")

    return fix_model(thetas["model_a"])


def rebasin_merge(
    thetas: Dict[str, os.PathLike | str],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
    iterations: int = 1,
    device="cpu",
):
    # WARNING: not sure how this does when 3 models are involved...

    model_a = thetas["model_a"].clone()
    perm_spec = sdunet_permutation_spec()

    print("permuting")
    for it in range(iterations):
        # print(it)
        log_vram(f"{it} iteration start")
        new_weights, new_bases = step_weights_and_bases(weights, bases, it, iterations)
        log_vram("weights & bases, before simple merge")

        # normal block merge we already know and love
        thetas["model_a"] = simple_merge(
            thetas, new_weights, new_bases, merge_mode, precision, weights_clip
        )

        log_vram("simple merge done")

        # find permutations
        perm_1, y = weight_matching(
            perm_spec,
            model_a,
            thetas["model_a"],
            max_iter=it,
            init_perm=None,
            usefp16=precision == 16,
            device=device,
        )

        log_vram("weight matching #1 done")

        thetas["model_a"] = apply_permutation(perm_spec, perm_1, thetas["model_a"])

        log_vram("apply perm 1 done")

        perm_2, z = weight_matching(
            perm_spec,
            thetas["model_b"],
            thetas["model_a"],
            max_iter=it,
            init_perm=None,
            usefp16=precision == 16,
            device=device,
        )

        log_vram("weight matching #2 done")

        new_alpha = torch.nn.functional.normalize(
            torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0
        ).tolist()[0]
        thetas["model_a"] = update_model_a(
            perm_spec, perm_2, thetas["model_a"], new_alpha
        )

        log_vram("model a updated")

    return thetas["model_a"]


def merge_key(
    key: str,
    thetas: Dict,
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
) -> Optional[Tuple[str, Dict]]:
    if KEY_POSITION_IDS in key:
        return

    for theta in thetas.values():
        if key not in theta.keys():
            return

    if "model" in key:
        current_bases = bases

        if "model.diffusion_model." in key:
            weight_index = -1

            re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
            re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
            re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12

            if "time_embed" in key:
                weight_index = 0  # before input blocks
            elif ".out." in key:
                weight_index = NUM_TOTAL_BLOCKS - 1  # after output blocks
            elif m := re_inp.search(key):
                weight_index = int(m.groups()[0])
            elif re_mid.search(key):
                weight_index = NUM_INPUT_BLOCKS
            elif m := re_out.search(key):
                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + int(m.groups()[0])

            if weight_index >= NUM_TOTAL_BLOCKS:
                raise ValueError(f"illegal block index {key}")

            if weight_index >= 0:
                current_bases = {k: w[weight_index] for k, w in weights.items()}

        try:
            merge_method = getattr(merge_methods, merge_mode)
        except AttributeError as e:
            raise ValueError(f"{merge_mode} not implemented, aborting merge!") from e

        merge_args = get_merge_method_args(current_bases, thetas, key)
        merged_key = merge_method(**merge_args)

        if weights_clip:
            t0 = thetas["model_a"][key]
            t1 = thetas["model_b"][key]
            threshold = torch.maximum(torch.abs(t0), torch.abs(t1))
            merged_key = torch.minimum(torch.maximum(merged_key, -threshold), threshold)

        if precision == 16:
            merged_key = merged_key.half()

        return merged_key


def clip_weights(thetas, merged):
    for k, t0 in thetas["model_a"].items():
        t1 = thetas["model_b"][k]
        th = torch.maximum(torch.abs(t0), torch.abs(t1))
        merged.update({k: torch.minimum(torch.maximum(merged[k], -th), th)})
    return merged


@contextmanager
def merge_key_context(*args, **kwargs):
    result = merge_key(*args, **kwargs)
    try:
        yield result
    finally:
        if result is not None:
            del result


def get_merge_method_args(current_bases: Dict, thetas: Dict, key: str) -> Dict:
    merge_method_args = {
        "a": thetas["model_a"][key],
        "b": thetas["model_b"][key],
        **current_bases,
    }

    if "model_c" in thetas:
        merge_method_args["c"] = thetas["model_c"][key]

    return merge_method_args


def save_model(model, output_file, file_format) -> None:
    print(f"saving {output_file}")
    if file_format == "safetensors":
        safetensors.torch.save_file(
            model if type(model) == dict else model.to_dict(),
            f"{output_file}.safetensors",
            metadata={"format": "pt"},
        )
    else:
        torch.save({"state_dict": model}, f"{output_file}.ckpt")
