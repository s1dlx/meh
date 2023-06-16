import inspect

from sd_meh import merge_methods
from sd_meh.merge import NUM_TOTAL_BLOCKS
from sd_meh.presets import BLOCK_WEIGHTS_PRESETS

MERGE_METHODS = dict(inspect.getmembers(merge_methods, inspect.isfunction))
BETA_METHODS = [
    name
    for name, fn in MERGE_METHODS.items()
    if "beta" in inspect.getfullargspec(fn)[0]
]


def compute_weights(weights, base):
    if not weights:
        return [base] * NUM_TOTAL_BLOCKS
    if "," in weights:
        w_alpha = list(map(float, weights.split(",")))
        if len(w_alpha) == NUM_TOTAL_BLOCKS:
            return w_alpha


def assemble_weights_and_bases(preset, weights, base, greek_letter):
    if preset:
        b, *w = BLOCK_WEIGHTS_PRESETS[preset]
    bases = {greek_letter: b}
    weights = {greek_letter: compute_weights(w, b)}

    return weights, bases


def interpolate_presets(
    weights, bases, weights_b, bases_b, greek_letter, presets_lambda
):
    for i, e in enumerate(weights[greek_letter]):
        weights[greek_letter][i] = (
            1 - presets_lambda
        ) * e + presets_lambda * weigths_b[greek_letter][i]
        bases[greek_letter] = (1 - presets_lambda) * bases[
            greek_letter
        ] + presets_lambda * bases_b[greek_letter]

    return weiths, bases


def weigths_and_bases(
    merge_mode,
    weigths_alpha,
    base_alpha,
    block_weights_preset_alpha,
    weigths_beta,
    base_beta,
    block_weights_preset_beta,
    block_weights_preset_alpha_b,
    block_weights_preset_beta_b,
    presets_alpha_lambda,
    presets_beta_lambda,
):
    weights, bases = assemble_weights_and_bases(
        block_weights_preset_alpha,
        weights_alpha,
        base_alpha,
        "alpha",
    )

    if block_weights_preset_alpha_b:
        weights_b, bases_b = assemble_weights_and_bases(
            block_weights_preset_alpha_b,
            None,
            None,
            "alpha",
        )
        weights, bases = interpolate_presets(
            weights,
            bases,
            weights_b,
            bases_b,
            "alpha",
            presets_alpha_lambda,
        )

    if merge_mode in BETA_METHODS:
        weights_beta, bases_beta = assemble_weights_and_bases(
            block_weights_preset_beta,
            weights_beta,
            base_beta,
            "beta",
        )

        if block_weights_preset_beta_b:
            weights_b, bases_b = assemble_weights_and_bases(
                block_weights_preset_beta_b,
                None,
                None,
                "beta",
            )
            weights, bases = interpolate_presets(
                weights,
                bases,
                weights_b,
                bases_b,
                "beta",
                presets_beta_lambda,
            )

        weights |= weights_beta
        bases |= bases_beta

    return weights, bases
