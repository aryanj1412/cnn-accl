# Reads model/model weights, outputs location per layer

import torch
import json
import argparse
from collections import OrderedDict
from pathlib import Path

FIXED_POINT_FRACTIONAL_BITS = 16
DTYPE_BYTES = 4

def float_to_fixed(tensor, frac_bits):
    scale = 1 << frac_bits # left shift operator frac_bits times
    tensor_fp = torch.round(tensor*scale)
    tensor_fp = torch.clamp(tensor_fp, min=-(2**31), max=(2**31-1))

    return tensor_fp.to(torch.int32)

def extract_weights(model_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] loading model: {model_path}')
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    state_dict = model.state_dict()

    weights_bin_path = output_dir
    weights_map_path = output_dir

    weights_map = OrderedDict()
    current_offset = 0

    with open(weights_bin_path, 'wb') as fbin:
        for name, tensor in state_dict.items():
            tensor = tensor.detach().cpu()

            tensor_fp = float_to_fixed(tensor, FIXED_POINT_FRACTIONAL_BITS)

            raw = tensor_fp.numpy().tobytes()
            size_bytes = len(raw)

            fbin.write(raw)

            weights_map[name] = {
                "offset": current_offset,
                "size": size_bytes,
                "shape": list(tensor.shape),
                "dtype": "int32",
                "frac_bits": FIXED_POINT_FRACTIONAL_BITS
            }

            print(f'offset={current_offset}, size={size_bytes}')

            current_offset += size_bytes

    with open(weights_map_path, "w") as fmap:
        json.dump(weights_map, fmap, indent=4)
    
    print("\n[DONE]")
    print(f" weights.bin        : {weights_bin_path}")
    print(f" weights.map.json   : {weights_map_path}")
    print(f" total bytes        : {current_offset}")


parser = argparse.ArgumentParser(
    description="Extract PyTorch weights into fixed-point binary"
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to .pt model file"
)

parser.add_argument(
    "--out",
    type=str,
    default="out_weights",
    help="Output directory"
)

args = parser.parse_args()

extract_weights(args.model, args.out)