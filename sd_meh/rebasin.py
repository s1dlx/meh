import torch

from sd_meh.merge import simple_merge
from sd_meh.weights_matching import (
    apply_permutation,
    sdunet_permutation_spec,
    weight_matching,
)

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


def rebasin_merge(
    thetas: Dict[str, os.PathLike | str],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
    iterations: int = 1,
):
    # WARNING: not sure how this does when 3 models are involved...

    model_a = thetas["theta_0"].copy()
    perm_spec = sdunet_permutation_spec()
    for it in range(iterations):
        new_weights, new_bases = step_weights_and_bases(weigths, bases, it, iterations)

        # normal block merge we already know and love
        thetas["theta_0"] = simple_merge(
            thetas, new_weights, new_bases, merge_mode, precision, weights_clip
        )

        # find permutations
        print("permuting")
        perm_1, y = weight_matching(
            perm_spec, flatten_params(model_a), thetas["theta_0"], usefp16=True
        )
        thetas["theta_0"] = apply_permutation(perm_spec, perm_1, thetas["theta_0"])
        perm_2, z = weight_matching(
            perm_spec, flatten_params(model_b), thetas["theta_0"], usefp16=True
        )
        theta_3 = apply_permutation(perm_spec, perm_2, thetas["theta_0"])

        # TODO: how to turn this to block-merge?
        new_alpha = torch.nn.functional.normalize(
            torch.sigmoid(toch.Tensor([y, z])), p=1, dim=0
        ).tolist()[0]
        for key in SPECIAL_KEYS:
            thetas["theta_0"][key] = (1 - new_alpha) * (
                thetas["theta_0"][key]
            ) + new_alpha * theta_3[key]

    return thetas["theta_0"]
