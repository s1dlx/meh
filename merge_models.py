import inspect

import click

from sd_meh import merge_methods
from sd_meh.merge import NUM_TOTAL_BLOCKS, merge_models, save_model
from sd_meh.presets import BLOCK_WEIGHTS_PRESETS

merge_methods = dict(inspect.getmembers(merge_methods, inspect.isfunction))
beta_methods = [
    name
    for name, fn in merge_methods.items()
    if "beta" in inspect.getfullargspec(fn)[0]
]


def compute_weights(weights, base):
    if not weights:
        return [base] * NUM_TOTAL_BLOCKS
    if "," in weights:
        w_alpha = list(map(float, weights.split(",")))
        if len(w_alpha) == NUM_TOTAL_BLOCKS:
            return w_alpha


@click.command()
@click.option("-a", "--model_a", "model_a", type=str)
@click.option("-b", "--model_b", "model_b", type=str)
@click.option("-c", "--model_c", "model_c", default=None, type=str)
@click.option(
    "-m",
    "--merging_method",
    "merge_mode",
    type=click.Choice(list(merge_methods.keys()), case_sensitive=False),
)
@click.option("-wc", "--weights_clip", "weights_clip", is_flag=True)
@click.option("-p", "--precision", "precision", type=int, default=16)
@click.option("-o", "--output_path", "output_path", type=str, default="model_out")
@click.option(
    "-f",
    "--output_format",
    "output_format",
    type=click.Choice(["safetensors", "ckpt"], case_sensitive=False),
)
@click.option("-wa", "--weights_alpha", "weights_alpha", type=str, default=None)
@click.option("-ba", "--base_alpha", "base_alpha", type=float, default=0.0)
@click.option("-wb", "--weights_beta", "weights_beta", type=str, default=None)
@click.option("-bb", "--base_beta", "base_beta", type=float, default=0.0)
@click.option("-rb", "--re_basin", "re_basin", is_flag=True)
@click.option(
    "-rbi", "--re_basin_iterations", "re_basin_iterations", type=int, default=1
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.Choice(
        ["cpu", "cuda"],
        case_sensitive=False,
    ),
    default="cpu",
)
@click.option("-pr", "--prune", "prune", is_flag=True)
@click.option(
    "-bwpa",
    "--block_weights_preset_alpha",
    "block_weights_preset_alpha",
    type=click.Choice(list(BLOCK_WEIGHTS_PRESETS.keys()), case_sensitive=False),
    default=None,
)
@click.option(
    "-bwpb",
    "--block_weights_preset_beta",
    "block_weights_preset_beta",
    type=click.Choice(list(BLOCK_WEIGHTS_PRESETS.keys()), case_sensitive=False),
    default=None,
)
def main(
    model_a,
    model_b,
    model_c,
    merge_mode,
    weights_clip,
    precision,
    output_path,
    output_format,
    weights_alpha,
    base_alpha,
    weights_beta,
    base_beta,
    re_basin,
    re_basin_iterations,
    device,
    prune,
    block_weights_preset_alpha,
    block_weights_preset_beta,
):
    models = {"model_a": model_a, "model_b": model_b}
    if model_c:
        models["model_c"] = model_c

    if block_weights_preset_alpha:
        base_alpha, *weights_alpha = BLOCK_WEIGHTS_PRESETS[block_weights_preset_alpha]
    bases = {"alpha": base_alpha}
    weights = {"alpha": compute_weights(weights_alpha, base_alpha)}

    if merge_mode in beta_methods:
        if block_weights_preset_beta:
            base_beta, *weights_beta = BLOCK_WEIGHTS_PRESETS[block_weights_preset_beta]
        weights["beta"] = compute_weights(weights_beta, base_beta)
        bases["beta"] = base_beta

    merged = merge_models(
        models,
        weights,
        bases,
        merge_mode,
        precision,
        weights_clip,
        re_basin,
        re_basin_iterations,
        device,
        prune,
    )

    save_model(merged, output_path, output_format)


if __name__ == "__main__":
    main()
