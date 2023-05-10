# sd-meh

```
Usage: merge_models.py [OPTIONS]

Options:
  -a, --model_a TEXT
  -b, --model_b TEXT
  -c, --model_c TEXT
  -m, --merging_method [weighted_sum|add_difference|weighted_subtraction|sum_twice|triple_sum|tensor_sum]
  -p, --precision INTEGER
  -s, --skip_position_ids INTEGER
  -o, --output_path TEXT
  -f, --output_format [safetensors|ckpt]
  -wa, --weights_alpha TEXT
  -ba, --base_alpha FLOAT
  -wb, --weights_beta TEXT
  -bb, --base_beta FLOAT
  --help                          Show this message and exit.
  ```