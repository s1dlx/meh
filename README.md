# sd-meh

[![PyPI version](https://badge.fury.io/py/sd-meh.svg)](https://badge.fury.io/py/sd-meh)

The merging execution helper (meh) is a python module for stable diffusion models merging.
This repository will never contain code for a webui extension.
This is because the aim is to have a GUI agnostic merging engine that can be reused in multiple extensions. 

You can install the module as

```
pip install sd-meh
```

and then use it in your extension as

```python
from sd_meh.merge import merge_models

merged_model = merge_models(models, weights, bases, merge_mode, precision)
```

You can have a look at the provided `merge_models.py` for an example on how to use the function.


```
Usage: merge_models.py [OPTIONS]

Options:
  -a, --model_a TEXT
  -b, --model_b TEXT
  -c, --model_c TEXT
  -m, --merging_method [weighted_sum|add_difference|weighted_subtraction|sum_twice|triple_sum|tensor_sum|similarity_add_difference|top_k_tensor_sum|multiply_difference|distribution_crossover|euclidean_add_difference|ties_add_difference]
  -wc, --weights_clip
  -p, --precision INTEGER
  -o, --output_path TEXT
  -f, --output_format [safetensors|ckpt]
  -wa, --weights_alpha TEXT
  -ba, --base_alpha FLOAT
  -wb, --weights_beta TEXT
  -bb, --base_beta FLOAT
  -rb, --re_basin
  -rbi, --re_basin_iterations INTEGER
  -d, --device [cpu|cuda]
  -pr, --prune
  --help                          Show this message and exit.
```

## Features

- gpu merging
- prune model before merging (and un-prune at the end)
- weights matching aka re-basin
- weights clipping
- registered pypi package
- block merge
- merging methods: `weighted_sum`, `add_difference`, `weighted_subtraction`, `sum_twice`, `triple_sum`, `tensor_sum`, `similarity_add_difference`, `top_k_tensor_sum`, `distribution_crossover`, `multiply_difference`, `euclidean_add_difference`, `ties_add_difference`
- `fp16` and `fp32`
