import torch
import json
import safetensors.torch
from scipy.sparse.linalg import svds
import time

CLAMP_QUANTILE = 0.99

def convert_key(key):
    model_key = ".".join(key.split(".")[:-2])
    return model_key

def convert_lora_to_weights(lora_state_dict):
    state_dict = {}
    with open("keydict.json", 'r') as kd:
        key_dict = json.load(kd)
    print("Converting LoRA to weight space")
    for key in lora_state_dict.keys():
        if "lora_down" in key:
            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[: key.index("lora_down")] + "alpha"
            model_key = key_dict[convert_key(key)]
            down_weight = lora_state_dict[key].to(dtype=torch.float32)
            up_weight = lora_state_dict[up_key].to(dtype=torch.float32)
            multiplier = lora_state_dict.get(alpha_key, down_weight.size()[0]) / down_weight.size()[0]

            if len(down_weight.size()) == 2:
                # linear
                state_dict[model_key] = (up_weight @ down_weight) * multiplier

            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                state_dict[model_key] = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * multiplier

            else:
                # conv2d 3x3
                state_dict[model_key] = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3) * multiplier

    return state_dict

def convert_weights_to_lora(state_dict):
    # extract from merged weights
    lora_state_dict = {}
    with open("keydict.json", 'r') as kd:
        key_dict = json.load(kd)
    with torch.no_grad():
        for key, weight in state_dict.items():
            conv2d = (len(weight.size()) == 4)
            kernel_size = None if not conv2d else weight.size()[2:4]
            conv2d_3x3 = conv2d and kernel_size != (1, 1)
            out_dim, in_dim = weight.size()[0:2]

            if conv2d:
                if conv2d_3x3:
                    weight = weight.flatten(start_dim=1)
                else:
                    weight = weight.squeeze()
            new_rank = 128
            new_conv_rank = 128
            module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
            module_new_rank = min(module_new_rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

            U, S, Vh = torch.linalg.svd(weight)

            U = U[:, :module_new_rank]
            S = S[:module_new_rank]
            U = U @ torch.diag(S)

            Vh = Vh[:module_new_rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            if conv2d:
                U = U.reshape(out_dim, module_new_rank, 1, 1)
                Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

            up_weight = U
            down_weight = Vh

            lora_state_dict[key_dict[key] + '.lora_up.weight'] = up_weight.to("cpu").to(dtype=torch.float16).contiguous()
            lora_state_dict[key_dict[key] + '.lora_down.weight'] = down_weight.to("cpu").to(dtype=torch.float16).contiguous()
            lora_state_dict[key_dict[key] + '.alpha'] = torch.tensor(module_new_rank)

    return lora_state_dict

# testing routine
# state_dict = convert_lora_to_weights(safetensors.torch.load_file(".safetensors"))
# lora_state_dict = convert_weights_to_lora(state_dict)
#
# safetensors.torch.save_file(lora_state_dict, ".safetensors", None)
