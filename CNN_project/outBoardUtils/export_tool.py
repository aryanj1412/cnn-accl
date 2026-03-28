# export_tool.py
# outputs a json file which contains location of weights, type of layer, etc

import torch
import torch.nn as nn
import json
from dummy_model import TinyResNet


def compute_output_dim(in_dim, kernel, stride, padding):
    return (in_dim + 2 * padding - kernel) // stride + 1


def export_model(weight_path,
                 input_shape=(3, 224, 224),
                 bin_path="model.bin",
                 json_path="model.json"):

    model = TinyResNet()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    layers_json = []

    c, h, w = input_shape

    weight_id = 3
    bias_id = 4
    input_id = 1
    output_id = 2

    with open(bin_path, "wb") as bin_file:

        for name, module in model.named_modules():

            if isinstance(module, nn.Conv2d):

                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_h, kernel_w = module.kernel_size
                stride = module.stride[0]
                padding = module.padding[0]

                out_h = compute_output_dim(h, kernel_h, stride, padding)
                out_w = compute_output_dim(w, kernel_w, stride, padding)

                weight = module.weight.detach().numpy().astype("float32")
                bias = module.bias.detach().numpy().astype("float32")

                bin_file.write(weight.tobytes())
                bin_file.write(bias.tobytes())

                layer_entry = {
                    "type": "cnn",
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": h,
                    "in_width": w,
                    "out_height": out_h,
                    "out_width": out_w,
                    "activation": 1,  # ReLU assumed
                    "kernel_height": kernel_h,
                    "kernel_width": kernel_w,
                    "stride": stride,
                    "padding": padding,
                    "resnet": 1 if "layer" in name else 0,
                    "input_address_id": input_id,
                    "output_address_id": output_id,
                    "weight_address_id": weight_id,
                    "bias_address_id": bias_id,
                    "flags": 0
                }

                layers_json.append(layer_entry)

                # Update feature map state
                c = out_channels
                h = out_h
                w = out_w

                weight_id += 2
                bias_id += 2
                input_id += 1
                output_id += 1

    with open(json_path, "w") as f:
        json.dump(layers_json, f, indent=4)

    print("Export complete.")
    print(f"Generated: {bin_path}")
    print(f"Generated: {json_path}")


if __name__ == "__main__":
    export_model("weights.pt")
