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

You can have a look at the provided `merge_models.py` cli for an example on how to use the function. Run `python3 merge_models.py --help` for a list of the available arguments.

## Changelog

### 0.7.0
- add `-bwpa, --block_weights_preset_alpha` and `-bwpb, --block_weights_preset_beta` to use pre-defined merging weights. Have a look at the [wiki](https://github.com/s1dlx/meh/wiki/Presets) for all the presets.
