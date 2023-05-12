import click

from sd_meh.merge import NUM_TOTAL_BLOCKS, merge_models, save_model


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
    type=click.Choice(
        [
            "weighted_sum",
            "add_difference",
            "weighted_subtraction",
            "sum_twice",
            "triple_sum",
            "tensor_sum",
        ],
        case_sensitive=False,
    ),
)
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
def main(
    model_a,
    model_b,
    model_c,
    merge_mode,
    precision,
    output_path,
    output_format,
    weights_alpha,
    base_alpha,
    weights_beta,
    base_beta,
):
    models = {"model_a": model_a, "model_b": model_b}
    if model_c:
        models["model_c"] = model_c

    bases = {"alpha": base_alpha}
    weights = {"alpha": compute_weights(weights_alpha, base_alpha)}

    if merge_mode in ["weighted_subtraction", "tensor_sum", "sum_twice", "triple_sum"]:
        weights["beta"] = compute_weights(weights_beta, base_beta)
        bases["beta"] = base_beta

    merged = merge_models(
        models, weights, bases, merge_mode, precision
    )
    save_model(merged, output_path, output_format)


if __name__ == "__main__":
    main()
