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
  -m, --merging_method [weighted_sum|add_difference|weighted_subtraction|sum_twice|triple_sum|tensor_sum]
  -wc, --weights_clip
  -p, --precision INTEGER
  -o, --output_path TEXT
  -f, --output_format [safetensors|ckpt]
  -wa, --weights_alpha TEXT
  -ba, --base_alpha FLOAT
  -wb, --weights_beta TEXT
  -bb, --base_beta FLOAT
  --help                          Show this message and exit.
```

## Features

- weights clipping
- registered pypi package
- block merge
- merging methods: `weighted_sum`, `add_difference`, `weighted_subtraction`, `sum_twice`, `triple_sum`, `tensor_sum`
- `fp16` and `fp32`
