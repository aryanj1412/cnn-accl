# CNNProjectModelReader.py
# Loads a .pt model file, converts all weights/biases to int16 fixed-point,
# writes weights.bin and weights_map.json to the output directory.
#
# Supports both full model objects (torch.save(model)) and
# state dicts (torch.save(model.state_dict())).

import torch
import json
import argparse
from collections import OrderedDict
from pathlib import Path

# Q1.15 format: 1 sign bit, 15 fractional bits
# Range: [-1.0, 1.0) in steps of 2^-15
FRACTIONAL_BITS = 15
INT16_MIN = -(2 ** 15)      # -32768
INT16_MAX =  (2 ** 15) - 1  #  32767


def float_to_fixed16(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a float tensor to Q1.15 fixed-point int16."""
    scale = 1 << FRACTIONAL_BITS
    tensor_fp = torch.round(tensor * scale)
    tensor_fp = torch.clamp(tensor_fp, min=INT16_MIN, max=INT16_MAX)
    return tensor_fp.to(torch.int16)


def load_state_dict(model_path: str) -> OrderedDict:
    """
    Load a .pt file. Handles both:
      - Full model objects  (torch.save(model))
      - State dicts         (torch.save(model.state_dict()))
    Returns an OrderedDict of {name: tensor}.
    """
    obj = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(obj, OrderedDict) or isinstance(obj, dict):
        print("[INFO] Detected state dict format.")
        return obj
    elif hasattr(obj, "state_dict"):
        print("[INFO] Detected full model object format.")
        obj.eval()
        return obj.state_dict()
    else:
        raise ValueError(
            f"Unrecognised .pt file contents: {type(obj)}. "
            "Expected a model object or a state dict."
        )


def extract_weights(model_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    weights_bin_path  = out / "weights.bin"
    weights_map_path  = out / "weights_map.json"

    print(f"[INFO] Loading model: {model_path}")
    state_dict = load_state_dict(model_path)

    weights_map = OrderedDict()
    current_offset = 0

    with open(weights_bin_path, "wb") as fbin:
        for name, tensor in state_dict.items():
            tensor = tensor.detach().cpu()

            tensor_fp = float_to_fixed16(tensor)
            raw = tensor_fp.numpy().tobytes()
            size_bytes = len(raw)

            fbin.write(raw)

            weights_map[name] = {
                "offset":    current_offset,
                "size":      size_bytes,
                "shape":     list(tensor.shape),
                "dtype":     "int16",
                "frac_bits": FRACTIONAL_BITS,
            }

            print(f"  {name:40s}  offset={current_offset:>10d}  size={size_bytes:>8d}  shape={list(tensor.shape)}")
            current_offset += size_bytes

    with open(weights_map_path, "w") as fmap:
        json.dump(weights_map, fmap, indent=4)

    print("\n[DONE]")
    print(f"  weights.bin      : {weights_bin_path}")
    print(f"  weights_map.json : {weights_map_path}")
    print(f"  total bytes      : {current_offset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract PyTorch weights into int16 fixed-point binary (Q1.15)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .pt model file (full model or state dict).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out_weights",
        help="Output directory for weights.bin and weights_map.json.",
    )
    args = parser.parse_args()
    extract_weights(args.model, args.out)
