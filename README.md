# sd-meh

[![PyPI version](https://badge.fury.io/py/sd-meh.svg)](https://badge.fury.io/py/sd-meh)

[![](https://dcbadge.vercel.app/api/server/EZJuBfNVHh)](https://discord.gg/EZJuBfNVHh)


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

[Join](https://discord.gg/EZJuBfNVHh) our discord server for discussion and features/bugfix requests

## Changelog

### 0.9.1 ... 0.9.3
- bugfixes
- support for pix2pix and inpainting models

### 0.8.0
- add `-bwpab, --block_weights_preset_alpha_b"` and `-pal, --presets_alpha_lambda` for presets interpolation (same for `beta`)
- add `-ll, --logging_level`, default to `INFO`

### 0.7.0
- add `-bwpa, --block_weights_preset_alpha` and `-bwpb, --block_weights_preset_beta` to use pre-defined merging weights. Have a look at the [wiki](https://github.com/s1dlx/meh/wiki/Presets) for all the presets
- add `-wd, --work_device`
- add `-pr, --prune`
- add `-j, --threads`


## DEV

PRs are welcome for both new features and bug fixes. 

Please make sure you format the code with black (you can `make format`) before submitting a PR.

### You want to add a `feature`

- open a `feat:` PR merging to `dev` branch, not `main`
- *do not* update version numbers
- ask for a review

### You want to make a bug `fix`

- open a `fix:` PR mergin to `main`
- update version number in `pyproject.toml` and `sd_meh/__init__.py`
- ask for a review
