# export_tool.py
# Reads a trained .pt model, generates:
#   - instructions.json  : one opcode entry per layer (human-readable field names)
#   - data_table.json    : maps 4-bit address IDs -> DDR byte offsets (consumed by PS)
#
# Supported layer types (matches opcode spec):
#   0 (0b0000) : Conv2d        (CNN)
#   1 (0b0001) : MaxPool2d
#   2 (0b0010) : AvgPool2d  /  AdaptiveAvgPool2d
#   3 (0b0011) : Linear        (Dense)
#
# Data-table layout:
#   ID 0 : reserved (input  feature map, filled by PS at runtime)
#   ID 1 : reserved (output feature map, filled by PS at runtime)
#   IDs 2..15 : weight/bias buffers, allocated sequentially by this compiler
#
# Limitation: ResNet (skip-connection) flag is set heuristically by checking
# whether the layer name contains "layer". A more robust solution would require
# tracing the model graph (e.g. torch.fx).

import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path
from collections import OrderedDict


# ── Opcode field helpers ──────────────────────────────────────────────────────

LAYER_TYPE = {
    "cnn":     0b0000,
    "maxpool": 0b0001,
    "avgpool": 0b0010,
    "dense":   0b0011,
}

ACTIVATION_RELU = 0b00   # Only ReLU supported in current hardware revision

DATA_TABLE_INPUT_ID  = 0  # Reserved: input  feature map
DATA_TABLE_OUTPUT_ID = 1  # Reserved: output feature map
DATA_TABLE_FIRST_WEIGHT_ID = 2  # Weight/bias IDs start here
DATA_TABLE_MAX_ID = 15    # 4-bit ID space: 0–15


# ── Dimension helpers ─────────────────────────────────────────────────────────

def conv_out_dim(in_dim: int, kernel: int, stride: int, padding: int) -> int:
    return (in_dim + 2 * padding - kernel) // stride + 1


def pool_out_dim(in_dim: int, kernel: int, stride: int, padding: int) -> int:
    return (in_dim + 2 * padding - kernel) // stride + 1


# ── Model loading (mirrors CNNProjectModelReader approach) ────────────────────

def load_model(model_path: str) -> nn.Module:
    """
    Load a .pt file. Handles full model objects only for export_tool,
    since we need the module hierarchy (named_modules) not just weights.
    If a bare state dict is passed, raises a clear error.
    """
    obj = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(obj, (dict, OrderedDict)):
        raise ValueError(
            "export_tool.py requires a full model object (torch.save(model)), "
            "not a state dict. Re-save with torch.save(model) or use "
            "CNNProjectModelReader.py for weight extraction only."
        )
    if not isinstance(obj, nn.Module):
        raise ValueError(f"Unrecognised .pt contents: {type(obj)}")

    obj.eval()
    return obj


# ── Address ID allocator ──────────────────────────────────────────────────────

class AddressAllocator:
    """
    Allocates 4-bit data-table IDs for weight and bias tensors.
    IDs 0 and 1 are reserved for input/output feature maps.
    Raises if the 16-entry table is exhausted.
    """
    def __init__(self):
        self._next_id = DATA_TABLE_FIRST_WEIGHT_ID
        self._table: dict[int, dict] = {
            DATA_TABLE_INPUT_ID:  {"role": "input_feature_map",  "offset": None},
            DATA_TABLE_OUTPUT_ID: {"role": "output_feature_map", "offset": None},
        }

    def allocate(self, role: str) -> int:
        if self._next_id > DATA_TABLE_MAX_ID:
            raise RuntimeError(
                f"Data-table exhausted (max {DATA_TABLE_MAX_ID + 1} entries). "
                "Consider splitting the model or extending the opcode spec."
            )
        id_ = self._next_id
        self._table[id_] = {"role": role, "offset": None}  # PS fills offset at load time
        self._next_id += 1
        return id_

    def table(self) -> dict:
        return self._table


# ── Main export logic ─────────────────────────────────────────────────────────

def export_model(
    model_path: str,
    input_shape: tuple = (3, 32, 32),
    json_path: str = "instructions.json",
    data_table_path: str = "data_table.json",
) -> None:

    model = load_model(model_path)
    allocator = AddressAllocator()
    layers = []

    c, h, w = input_shape  # track feature map dimensions as we walk layers

    for name, module in model.named_modules():

        # ── Conv2d ────────────────────────────────────────────────────────────
        if isinstance(module, nn.Conv2d):
            in_c  = module.in_channels
            out_c = module.out_channels
            k_h, k_w = module.kernel_size
            stride  = module.stride[0]
            padding = module.padding[0]

            out_h = conv_out_dim(h, k_h, stride, padding)
            out_w = conv_out_dim(w, k_w, stride, padding)

            weight_id = allocator.allocate(f"{name}.weight")
            bias_id   = allocator.allocate(f"{name}.bias") if module.bias is not None else 0

            layers.append({
                "name":              name,
                "type":              LAYER_TYPE["cnn"],
                "type_label":        "cnn",
                "in_channels":       in_c,
                "out_channels":      out_c,
                "in_height":         h,
                "in_width":          w,
                "out_height":        out_h,
                "out_width":         out_w,
                "activation":        ACTIVATION_RELU,
                "kernel_height":     k_h,
                "kernel_width":      k_w,
                "stride":            0 if stride == 1 else 1,   # opcode: 0=stride1, 1=stride2
                "padding":           0 if padding == 0 else 1,  # opcode: 0=no pad, 1=pad1
                "resnet":            1 if "layer" in name else 0,  # heuristic — see file header
                "input_address_id":  DATA_TABLE_INPUT_ID,
                "output_address_id": DATA_TABLE_OUTPUT_ID,
                "weight_address_id": weight_id,
                "bias_address_id":   bias_id,
                "flags":             0,
            })

            c, h, w = out_c, out_h, out_w

        # ── MaxPool2d ─────────────────────────────────────────────────────────
        elif isinstance(module, nn.MaxPool2d):
            k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            s = module.stride      if isinstance(module.stride,      int) else module.stride[0]
            p = module.padding     if isinstance(module.padding,      int) else module.padding[0]

            out_h = pool_out_dim(h, k, s, p)
            out_w = pool_out_dim(w, k, s, p)

            layers.append({
                "name":              name,
                "type":              LAYER_TYPE["maxpool"],
                "type_label":        "maxpool",
                "in_channels":       c,
                "out_channels":      c,
                "in_height":         h,
                "in_width":          w,
                "out_height":        out_h,
                "out_width":         out_w,
                "activation":        ACTIVATION_RELU,
                "kernel_height":     k,
                "kernel_width":      k,
                "stride":            0 if s == 1 else 1,
                "padding":           0 if p == 0 else 1,
                "resnet":            0,
                "input_address_id":  DATA_TABLE_INPUT_ID,
                "output_address_id": DATA_TABLE_OUTPUT_ID,
                "weight_address_id": 0,  # no weights
                "bias_address_id":   0,  # no bias
                "flags":             0,
            })

            h, w = out_h, out_w

        # ── AvgPool2d / AdaptiveAvgPool2d ─────────────────────────────────────
        elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            if isinstance(module, nn.AdaptiveAvgPool2d):
                # output_size gives us out_h, out_w directly
                out_size = module.output_size
                out_h = out_size[0] if hasattr(out_size, "__len__") else out_size
                out_w = out_size[1] if hasattr(out_size, "__len__") else out_size
                # Derive equivalent kernel to describe the operation
                k_h = h - out_h + 1
                k_w = w - out_w + 1
                s, p = 1, 0
            else:
                k_h = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                k_w = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[1]
                s   = module.stride      if isinstance(module.stride, int)      else module.stride[0]
                p   = module.padding     if isinstance(module.padding, int)     else module.padding[0]
                out_h = pool_out_dim(h, k_h, s, p)
                out_w = pool_out_dim(w, k_w, s, p)

            layers.append({
                "name":              name,
                "type":              LAYER_TYPE["avgpool"],
                "type_label":        "avgpool",
                "in_channels":       c,
                "out_channels":      c,
                "in_height":         h,
                "in_width":          w,
                "out_height":        out_h,
                "out_width":         out_w,
                "activation":        ACTIVATION_RELU,
                "kernel_height":     k_h,
                "kernel_width":      k_w,
                "stride":            0 if s == 1 else 1,
                "padding":           0 if p == 0 else 1,
                "resnet":            0,
                "input_address_id":  DATA_TABLE_INPUT_ID,
                "output_address_id": DATA_TABLE_OUTPUT_ID,
                "weight_address_id": 0,
                "bias_address_id":   0,
                "flags":             0,
            })

            h, w = out_h, out_w

        # ── Linear (Dense) ────────────────────────────────────────────────────
        elif isinstance(module, nn.Linear):
            in_features  = module.in_features
            out_features = module.out_features

            weight_id = allocator.allocate(f"{name}.weight")
            bias_id   = allocator.allocate(f"{name}.bias") if module.bias is not None else 0

            # Per opcode spec: in_channels = out_channels = 1 for dense
            # in_height = in_features, in_width = 1
            layers.append({
                "name":              name,
                "type":              LAYER_TYPE["dense"],
                "type_label":        "dense",
                "in_channels":       1,
                "out_channels":      1,
                "in_height":         in_features,
                "in_width":          1,
                "out_height":        out_features,
                "out_width":         1,
                "activation":        ACTIVATION_RELU,
                "kernel_height":     1,
                "kernel_width":      1,
                "stride":            0,
                "padding":           0,
                "resnet":            0,
                "input_address_id":  DATA_TABLE_INPUT_ID,
                "output_address_id": DATA_TABLE_OUTPUT_ID,
                "weight_address_id": weight_id,
                "bias_address_id":   bias_id,
                "flags":             0,
            })

            # After a linear layer spatial dims are no longer meaningful
            c, h, w = 1, out_features, 1

    # ── Write outputs ─────────────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(layers, f, indent=4)

    with open(data_table_path, "w") as f:
        json.dump(allocator.table(), f, indent=4)

    print("[DONE]")
    print(f"  instructions.json : {json_path}   ({len(layers)} layers)")
    print(f"  data_table.json   : {data_table_path}  ({len(allocator.table())} entries)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a PyTorch model to FPGA instruction + data-table JSON files."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .pt full model file (torch.save(model)).",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=[3, 32, 32],
        metavar=("C", "H", "W"),
        help="Input tensor shape as C H W (default: 3 32 32 for CIFAR-10).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="instructions.json",
        help="Output path for layer instruction JSON.",
    )
    parser.add_argument(
        "--out-table",
        type=str,
        default="data_table.json",
        help="Output path for data-table JSON.",
    )
    args = parser.parse_args()

    export_model(
        model_path=args.model,
        input_shape=tuple(args.input_shape),
        json_path=args.out_json,
        data_table_path=args.out_table,
    )
