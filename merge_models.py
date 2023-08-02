import logging

import click

from sd_meh.merge import merge_models, save_model
from sd_meh.presets import BLOCK_WEIGHTS_PRESETS
from sd_meh.utils import MERGE_METHODS, weights_and_bases


@click.command()
@click.option("-a", "--model_a", "model_a", type=str)
@click.option("-b", "--model_b", "model_b", type=str)
@click.option("-c", "--model_c", "model_c", default=None, type=str)
@click.option(
    "-m",
    "--merging_method",
    "merge_mode",
    type=click.Choice(list(MERGE_METHODS.keys()), case_sensitive=False),
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
@click.option(
    "-wd",
    "--work_device",
    "work_device",
    type=click.Choice(
        ["cpu", "cuda"],
        case_sensitive=False,
    ),
    default=None,
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
@click.option(
    "-j",
    "--threads",
    "threads",
    type=int,
    default=1,
)
@click.option(
    "-bwpab",
    "--block_weights_preset_alpha_b",
    "block_weights_preset_alpha_b",
    type=click.Choice(list(BLOCK_WEIGHTS_PRESETS.keys()), case_sensitive=False),
    default=None,
)
@click.option(
    "-bwpbb",
    "--block_weights_preset_beta_b",
    "block_weights_preset_beta_b",
    type=click.Choice(list(BLOCK_WEIGHTS_PRESETS.keys()), case_sensitive=False),
    default=None,
)
@click.option(
    "-pal",
    "--presets_alpha_lambda",
    "presets_alpha_lambda",
    type=float,
    default=None,
)
@click.option(
    "-pbl",
    "--presets_beta_lambda",
    "presets_beta_lambda",
    type=float,
    default=None,
)
@click.option(
    "-ll",
    "--logging_level",
    "logging_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
)
@click.option("-xl", "--sdxl", "sdxl", is_flag=True)
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
    work_device,
    prune,
    block_weights_preset_alpha,
    block_weights_preset_beta,
    threads,
    block_weights_preset_alpha_b,
    block_weights_preset_beta_b,
    presets_alpha_lambda,
    presets_beta_lambda,
    logging_level,
    sdxl,
):
    if logging_level:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging_level)

    models = {"model_a": model_a, "model_b": model_b}
    if model_c:
        models["model_c"] = model_c

    weights, bases = weights_and_bases(
        merge_mode,
        weights_alpha,
        base_alpha,
        block_weights_preset_alpha,
        weights_beta,
        base_beta,
        block_weights_preset_beta,
        block_weights_preset_alpha_b,
        block_weights_preset_beta_b,
        presets_alpha_lambda,
        presets_beta_lambda,
        sdxl,
    )

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
        work_device,
        prune,
        threads,
    )

    save_model(merged, output_path, output_format)


if __name__ == "__main__":
    main()
