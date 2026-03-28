# image_to_fixed.py
# Converts an input image to Q1.15 fixed-point format for FPGA inference.
#
# Pipeline:
#   Raw pixel (uint8, 0-255)
#   -> Normalise to [-1, 1]  (float32)
#   -> Scale by 2^15         (float32)
#   -> Round and clamp       (int16, Q1.15)
#   -> Write to .bin         (raw bytes, little-endian)
#
# Memory layout in .bin file:
#   Stored as [C][H][W] — channel-first (matches PyTorch convention)
#   Each value is 2 bytes, signed 16-bit little-endian
#
# Usage:
#   python image_to_fixed.py --image cat.png --out image.bin
#   python image_to_fixed.py --image cat.png --out image.bin --visualise

import argparse
import numpy as np
from pathlib import Path
from PIL import Image


FRACTIONAL_BITS = 15
SCALE           = 1 << FRACTIONAL_BITS   # 32768
INT16_MIN       = -(2 ** 15)             # -32768
INT16_MAX       =  (2 ** 15) - 1         #  32767

# These match QATmodel.py's transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
# formula: normalised = (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1.0
NORM_MEAN = 0.5
NORM_STD  = 0.5


def load_image(image_path: str, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load an image as a float32 numpy array in [C, H, W] format, values in [0, 1].
    Optionally resize to (H, W).
    """
    img = Image.open(image_path).convert("RGB")

    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # PIL takes (W, H)

    arr = np.array(img, dtype=np.float32) / 255.0   # H x W x C, range [0, 1]
    arr = arr.transpose(2, 0, 1)                     # C x H x W
    return arr


def normalise(arr: np.ndarray, mean: float = NORM_MEAN, std: float = NORM_STD) -> np.ndarray:
    """
    Apply per-channel normalisation matching QATmodel.py:
        out = (in - mean) / std
    With mean=0.5, std=0.5 this maps [0,1] -> [-1, 1].
    """
    return (arr - mean) / std


def to_fixed_q115(arr: np.ndarray) -> np.ndarray:
    """
    Convert a float32 array in [-1, 1] to Q1.15 int16.
        stored_value = round(float_value * 32768)
    Values outside [-1, 1) are clamped — a warning is printed if this occurs.
    """
    scaled  = arr * SCALE
    rounded = np.round(scaled)
    
    n_clipped = np.sum((rounded < INT16_MIN) | (rounded > INT16_MAX))
    if n_clipped > 0:
        pct = 100.0 * n_clipped / arr.size
        print(f"[WARNING] {n_clipped} values ({pct:.2f}%) saturated during Q1.15 conversion.")
        print(f"          Min value: {arr.min():.4f}, Max value: {arr.max():.4f}")
        print(f"          These were clamped to [{INT16_MIN}, {INT16_MAX}].")

    clamped = np.clip(rounded, INT16_MIN, INT16_MAX)
    return clamped.astype(np.int16)


def write_bin(arr_fixed: np.ndarray, out_path: str) -> None:
    """
    Write a C x H x W int16 array to a flat binary file (little-endian).
    The PS code reads this directly into DDR.
    """
    # Ensure C x H x W order and contiguous memory layout
    arr_chw = np.ascontiguousarray(arr_fixed)
    arr_chw.tofile(out_path)

    c, h, w     = arr_chw.shape
    total_bytes = arr_chw.size * 2   # 2 bytes per int16
    print(f"[INFO] Written: {out_path}")
    print(f"       Shape  : C={c}, H={h}, W={w}")
    print(f"       Bytes  : {total_bytes}  ({c}×{h}×{w}×2)")


def convert_image(
    image_path: str,
    out_path:   str,
    target_size: tuple[int, int] | None = None,
    visualise:   bool = False,
) -> np.ndarray:
    """
    Full pipeline: load -> normalise -> quantise -> write.
    Returns the int16 C x H x W array (useful for testing).
    """
    arr_float  = load_image(image_path, target_size)
    arr_norm   = normalise(arr_float)
    arr_fixed  = to_fixed_q115(arr_norm)

    write_bin(arr_fixed, out_path)

    if visualise:
        _visualise(arr_float, arr_norm, arr_fixed)

    return arr_fixed


def _visualise(arr_float: np.ndarray, arr_norm: np.ndarray, arr_fixed: np.ndarray) -> None:
    """Print a small numeric summary — no matplotlib dependency required."""
    print("\n[VISUALISE] First 5 values of channel 0, row 0:")
    print(f"  Raw float  (0-1)  : {arr_float[0, 0, :5]}")
    print(f"  Normalised (-1,1) : {arr_norm [0, 0, :5]}")
    print(f"  Q1.15 int16       : {arr_fixed[0, 0, :5]}")
    print(f"\n  To recover float from int16: value / {SCALE}  (= value / 2^{FRACTIONAL_BITS})")

    # Verify round-trip error
    recovered   = arr_fixed[0, 0, :5].astype(np.float32) / SCALE
    original    = arr_norm[0, 0, :5]
    max_error   = np.max(np.abs(recovered - original))
    print(f"\n  Max round-trip error (first 5 pixels): {max_error:.8f}")
    print(f"  Theoretical max Q1.15 error          : {1/SCALE:.8f}  (= 1/2^{FRACTIONAL_BITS})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an image to Q1.15 fixed-point binary for FPGA inference."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (PNG, JPG, etc.)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="image.bin",
        help="Output .bin file path (default: image.bin)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Resize image to H W before conversion (e.g. --size 32 32 for CIFAR-10)",
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Print a numeric summary of the conversion for verification",
    )
    args = parser.parse_args()

    convert_image(
        image_path  = args.image,
        out_path    = args.out,
        target_size = tuple(args.size) if args.size else None,
        visualise   = args.visualise,
    )
