import torch
import json

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
