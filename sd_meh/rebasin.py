# https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion

from typing import Dict, Tuple


SPECIAL_KEYS = [
    "first_stage_model.decoder.norm_out.weight",
    "first_stage_model.decoder.norm_out.bias",
    "first_stage_model.encoder.norm_out.weight",
    "first_stage_model.encoder.norm_out.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
]


def step_weights_and_bases(
    weights: Dict, bases: Dict, it: int = 0, iterations: int = 1
) -> Tuple[Dict, Dict]:
    new_weights = {
        gl: {
            k: 1 - (1 - (1 + it) * v / iterations) / (1 - it * v / iterations)
            if it > 0
            else v / iterations
            for k, v in w.items()
        }
        for gl, w in weights.items()
    }

    new_bases = {
        k: 1 - (1 - (1 + it) * v / iterations) / (1 - it * v / iterations)
        if it > 0
        else v / iterations
        for k, v in bases.items()
    }

    return new_weights, new_bases


def flatten_params(model):
    return model["state_dict"]
