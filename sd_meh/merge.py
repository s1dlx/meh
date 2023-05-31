import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch
from tqdm import tqdm

from sd_meh import merge_methods
from sd_meh.model import SDModel

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
    if KEY_POSITION_IDS in model:
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
    for k in model:
        model = fix_key(model, k)
    return fix_clip(model)


def load_sd_model(model: os.PathLike | str, device: str = "cpu") -> Dict:
    if isinstance(model, str):
        model = Path(model)

    return SDModel(model, device).load_model()


def merge_models(
    models: Dict[str, os.PathLike | str],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
) -> Dict:
    thetas = {k: load_sd_model(m) for k, m in models.items()}

    for key in tqdm(thetas["model_a"].keys(), desc="stage 1"):
        if result := merge_key(
            key,
            thetas,
            weights,
            bases,
            merge_mode,
            precision,
            weights_clip,
        ):
            thetas["model_a"][key] = result[1]

    for key in tqdm(thetas["model_b"].keys(), desc="stage 2"):
        if KEY_POSITION_IDS in key:
            continue
        if "model" in key and key not in thetas["model_a"]:
            thetas["model_a"].update({key: thetas["model_b"][key]})
            if precision == 16:
                thetas["model_a"][key] = thetas["model_a"][key].half()

    return fix_model(thetas["model_a"])


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
        if key not in theta:
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
        except AttributeError:
            raise ValueError(f"{merge_mode} not implemented, aborting merge!")

        merge_args = get_merge_method_args(current_bases, thetas, key)
        merged_key = merge_method(**merge_args)

        if weights_clip:
            t0 = thetas["model_a"][key]
            t1 = thetas["model_b"][key]
            threshold = torch.maximum(torch.abs(t0), torch.abs(t1))
            merged_key = torch.minimum(torch.maximum(merged_key, -threshold), threshold)

        if precision == 16:
            merged_key = merged_key.half()

        return key, merged_key


def get_merge_method_args(current_bases: Dict, thetas: Dict, key: str) -> Dict:
    merge_method_args = {
        "a": thetas["model_a"][key],
        "b": thetas["model_b"][key],
        **current_bases,
    }

    if "model_c" in thetas:
        merge_method_args.update(
            {
                "c": thetas["model_c"][key],
            }
        )

    return merge_method_args


def save_model(model, output_file, file_format) -> None:
    if file_format == "safetensors":
        safetensors.torch.save_file(
            model, f"{output_file}.safetensors", metadata={"format": "pt"}
        )
    else:
        torch.save({"state_dict": model}, f"{output_file}.ckpt")
