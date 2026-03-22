#!/usr/bin/env python3
"""
Layer-by-layer validation for CNN accelerator.

Two modes:
  --golden   : Run pure software Q6.9 model, print expected stats per layer
  --compare  : Run HW accelerator, dump intermediate fmaps, compare to SW golden

Usage on PYNQ:
  sudo python3 validate_layers.py /home/xilinx/test.jpg --golden
  sudo python3 validate_layers.py /home/xilinx/test.jpg --compare

Usage on any machine (golden only):
  python3 validate_layers.py /path/to/test.jpg --golden
"""

import sys
import os
import numpy as np
import argparse
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Fixed-point helpers (exact hardware replication)
# ──────────────────────────────────────────────────────────────

def quantize_q6_9(acc_q12_18):
    """Q12.18 → Q6.9: add 0.5 LSB (256), right-shift 9, saturate to int16.
    
    FIX: add 256 unconditionally (NOT sign-based).
    Hardware always adds 256 before the right-shift.
    """
    r = acc_q12_18.astype(np.int64) + 256   # FIX: was np.where(>=0, 256, -256)
    s = np.right_shift(r, 9)
    return np.clip(s, -32768, 32767).astype(np.int16)

def sw_conv3x3(fmap, weights, relu=True):
    """
    Software simulation of HW conv3x3.
    fmap:    [C_in_real, H, W]    int16 Q6.9
    weights: [C_out, C_in_hw, 3, 3] int16 Q6.9
             C_in_hw = num_groups * 16 (always a multiple of 16)

    The hardware zero-pads fmap channels to the next multiple of 16.
    e.g. conv1: fmap has 3 channels (RGB), HW pads ch3..15 to zero.
    Weights for ch3..15 are trained to zero so this is exact.
    """
    C_out, C_in_hw, kH, kW = weights.shape
    C_real = fmap.shape[0]
    H, W   = fmap.shape[1], fmap.shape[2]
    out_H, out_W = H - 2, W - 2

    # Zero-pad fmap channels to match hardware grouping
    if C_real < C_in_hw:
        pad = np.zeros((C_in_hw - C_real, H, W), dtype=np.int16)
        fmap = np.concatenate([fmap, pad], axis=0)

    output = np.zeros((C_out, out_H, out_W), dtype=np.int16)
    for oc in range(C_out):
        acc = np.zeros((out_H, out_W), dtype=np.int64)
        for ic in range(C_in_hw):
            for kr in range(3):
                for kc in range(3):
                    px = fmap[ic, kr:kr+out_H, kc:kc+out_W].astype(np.int64)
                    w  = int(weights[oc, ic, kr, kc])
                    acc += px * w
        q = quantize_q6_9(acc)
        output[oc] = np.maximum(q, np.int16(0)) if relu else q
    return output

def sw_pool2x2(fmap):
    """2×2 max pool. Crops odd spatial dims (matching _run_ps_pool)."""
    C, H, W = fmap.shape
    H2, W2 = (H//2)*2, (W//2)*2
    f = fmap[:, :H2, :W2]
    return np.maximum(
        np.maximum(f[:, 0::2, 0::2], f[:, 0::2, 1::2]),
        np.maximum(f[:, 1::2, 0::2], f[:, 1::2, 1::2])
    ).astype(np.int16)

def preprocess(img_path):
    """Replicate pynq_inference._preprocess exactly."""
    img = Image.open(img_path).convert('RGB').resize((128, 128), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std                 # [H, W, 3] float
    arr  = arr.transpose(2, 0, 1)             # [3, H, W] float
    # Quantize to Q6.9
    q    = np.round(arr * 512).astype(np.int16)
    return np.clip(q, -32768, 32767)

def fmap_stats(name, fmap):
    f = fmap.astype(np.float32) / 512.0
    nz = np.count_nonzero(fmap)
    total = fmap.size
    print(f"  {name:18s}  shape={str(fmap.shape):18s}  "
          f"min={f.min():7.3f}  max={f.max():7.3f}  "
          f"mean={f.mean():7.4f}  nonzero={nz}/{total} ({100*nz/total:.1f}%)")

def fmap_ch_means(tag, fmap, n_ch=8):
    """Print per-channel means for first n_ch channels — useful for diagnosing channel-level bugs."""
    means = [round(float(fmap[c].mean()) / 512.0, 4) for c in range(min(n_ch, fmap.shape[0]))]
    print(f"    {tag} ch-means[0:{min(n_ch,fmap.shape[0])}]: {means}")

def load_weights_for_hw(npz_path):
    """
    Load weights in HW layout: npz stores [C_out, num_groups, 16, 9]
    Convert back to [C_out, C_in, 3, 3] for software conv.
    """
    d = np.load(npz_path)
    wgts = {}
    for layer in ['conv1', 'conv2', 'conv3', 'conv4']:
        w_hw = d[f'{layer}_weight']    # [C_out, num_grp, 16, 9]
        C_out, num_grp, ch_per_grp, taps = w_hw.shape
        C_in = num_grp * ch_per_grp    # total input channels
        # Reshape taps (9) → (3,3) and merge groups
        w_sw = w_hw.reshape(C_out, C_in, 3, 3)
        wgts[layer] = w_sw
    wgts['fc1_weight'] = d['fc1_weight']
    wgts['fc1_bias']   = d['fc1_bias']
    wgts['fc2_weight'] = d['fc2_weight']
    wgts['fc2_bias']   = d['fc2_bias']
    return wgts


def run_float(img_path, npz_path):
    """
    Float32 inference using dequantized Q6.9 weights.
    This is what the original trained model predicts.
    Compared to Q6.9 golden this shows quantization error.
    """
    print(f"\n{'='*64}")
    print(f"FLOAT32 MODEL  (dequantized weights, float arithmetic)")
    print(f"Image : {os.path.basename(img_path)}")
    print(f"{'='*64}\n")

    d = np.load(npz_path)
    # Dequantize conv weights: int16 Q6.9 → float32
    conv_w = {}
    for layer in ['conv1','conv2','conv3','conv4']:
        w_hw  = d[f'{layer}_weight']               # (C_out, num_grp, 16, 9) int16
        C_out = w_hw.shape[0]
        C_in  = w_hw.shape[1] * w_hw.shape[2]
        w_f   = w_hw.reshape(C_out, C_in, 3, 3).astype(np.float32) / 512.0
        conv_w[layer] = w_f

    # Preprocess to float (no quantization)
    img  = Image.open(img_path).convert('RGB').resize((128, 128), Image.BILINEAR)
    arr  = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x    = ((arr - mean) / std).transpose(2, 0, 1)   # [3, 128, 128] float32

    def fconv(fmap, W, relu=True):
        C_out, C_in_hw, kH, kW = W.shape
        C_real = fmap.shape[0]
        H, W_  = fmap.shape[1], fmap.shape[2]
        out_H, out_W = H - 2, W_ - 2
        if C_real < C_in_hw:
            pad  = np.zeros((C_in_hw - C_real, H, W_), dtype=np.float32)
            fmap = np.concatenate([fmap, pad], axis=0)
        out = np.zeros((C_out, out_H, out_W), dtype=np.float32)
        for oc in range(C_out):
            for ic in range(C_in_hw):
                for kr in range(3):
                    for kc in range(3):
                        out[oc] += fmap[ic, kr:kr+out_H, kc:kc+out_W] * W[oc, ic, kr, kc]
            if relu:
                out[oc] = np.maximum(out[oc], 0)
        return out

    def fpool(fmap):
        C, H, W_ = fmap.shape
        H2, W2 = (H//2)*2, (W_//2)*2
        f = fmap[:, :H2, :W2]
        return np.maximum(
            np.maximum(f[:, 0::2, 0::2], f[:, 0::2, 1::2]),
            np.maximum(f[:, 1::2, 0::2], f[:, 1::2, 1::2])
        )

    fmap_stats('input', (x * 512).astype(np.int16))
    for name, wkey, is_conv, relu in [
        ('conv1_relu','conv1',True,True), ('pool1',None,False,False),
        ('conv2_relu','conv2',True,True), ('pool2',None,False,False),
        ('conv3_relu','conv3',True,True), ('pool3',None,False,False),
        ('conv4_relu','conv4',True,True), ('pool4',None,False,False),
    ]:
        x = fconv(x, conv_w[wkey], relu=relu) if is_conv else fpool(x)
        fmap_stats(name, (x * 512).clip(-32768,32767).astype(np.int16))

    xf    = x.flatten()
    fc1_w = d['fc1_weight']
    fc1_b = d['fc1_bias']
    fc2_w = d['fc2_weight']
    fc2_b = d['fc2_bias']
    xf    = np.maximum(0, xf @ fc1_w.T + fc1_b)
    logit = float((xf @ fc2_w.T + fc2_b)[0])
    prob  = 1.0 / (1.0 + np.exp(-logit))
    label = 'Dog' if prob > 0.5 else 'Cat'

    print(f"\n  logit = {logit:.4f}   prob = {prob:.4f}   → {label} ({100*abs(prob-0.5)*2:.1f}%)")
    print(f"  (If this disagrees with Q6.9 golden, quantization introduced error)")
    print(f"{'='*64}\n")
    return logit

def run_golden(img_path, npz_path):
    print(f"\n{'='*64}")
    print(f"GOLDEN MODEL  (pure software Q6.9)")
    print(f"Image : {os.path.basename(img_path)}")
    print(f"{'='*64}\n")

    wgts = load_weights_for_hw(npz_path)
    x = preprocess(img_path)
    fmap_stats('input', x)

    layers = [
        ('conv1_relu', 'conv1', True,  True),
        ('pool1',      None,    False, False),
        ('conv2_relu', 'conv2', True,  True),
        ('pool2',      None,    False, False),
        ('conv3_relu', 'conv3', True,  True),
        ('pool3',      None,    False, False),
        ('conv4_relu', 'conv4', True,  True),
        ('pool4',      None,    False, False),
    ]

    for name, wkey, is_conv, relu in layers:
        if is_conv:
            x = sw_conv3x3(x, wgts[wkey], relu=relu)
        else:
            x = sw_pool2x2(x)
        fmap_stats(name, x)

    # FC layers (float)
    xf = x.astype(np.float32) / 512.0
    xf = xf.flatten()
    xf = np.maximum(0, xf @ wgts['fc1_weight'].T + wgts['fc1_bias'])
    logit = float((xf @ wgts['fc2_weight'].T + wgts['fc2_bias'])[0])
    prob  = 1.0 / (1.0 + np.exp(-logit))
    label = 'Dog' if prob > 0.5 else 'Cat'

    print(f"\n  logit = {logit:.4f}   prob = {prob:.4f}   → {label} ({100*abs(prob-0.5)*2:.1f}%)")
    print(f"\n{'='*64}\n")
    return logit

def run_compare(img_path, npz_path, bit_path, hwh_path):
    """Run HW, collect intermediate fmaps, compare to golden."""
    try:
        from pynq import Overlay
        from pynq.lib.dma import DMA
    except ImportError:
        print("PYNQ not available - run on board")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pynq_inference as pi

    cnn = pi.CNNInference(bit_path, npz_path, verbose=True)

    print(f"\n{'='*64}")
    print(f"HW vs SW COMPARISON")
    print(f"Image : {os.path.basename(img_path)}")
    print(f"{'='*64}\n")

    wgts   = load_weights_for_hw(npz_path)
    x_sw   = preprocess(img_path)
    x_hw   = preprocess(img_path)

    sw_layers = [
        ('input',      None,    False),
        ('conv1_relu', 'conv1', True),
        ('pool1',      None,    False),
        ('conv2_relu', 'conv2', True),
        ('pool2',      None,    False),
        ('conv3_relu', 'conv3', True),
        ('pool3',      None,    False),
        ('conv4_relu', 'conv4', True),
        ('pool4',      None,    False),
    ]

    # Run SW golden
    sw_fmaps = {'input': x_sw.copy()}
    for name, wkey, is_conv in sw_layers[1:]:
        if is_conv:
            x_sw = sw_conv3x3(x_sw, wgts[wkey], relu=True)
        else:
            x_sw = sw_pool2x2(x_sw)
        sw_fmaps[name] = x_sw.copy()

    # Run HW and capture outputs after each layer
    from pynq_inference import (LAYER_CONFIG, MODE_CONV_RELU, MODE_PS_POOL)
    hw_fmaps = {}
    x_hw_cur = x_hw.copy()

    for (lname, in_h, in_w, out_h, out_w, in_ch, out_ch, mode) in LAYER_CONFIG:
        if mode == MODE_PS_POOL:
            x_hw_cur = cnn._run_ps_pool(lname, x_hw_cur, out_h, out_w)
        else:
            x_hw_cur = cnn._run_hw_layer(
                name=lname, fmap=x_hw_cur,
                in_h=in_h, in_w=in_w, out_h=out_h, out_w=out_w,
                num_in_ch=in_ch, num_out_ch=out_ch,
                mode=mode,
                wgt_key={'conv1_relu':'conv1','conv2_relu':'conv2',
                         'conv3_relu':'conv3','conv4_relu':'conv4'}.get(lname)
            )
        hw_fmaps[lname] = x_hw_cur.copy()

    # FIX: print conv1 channel means ONCE, outside the comparison loop
    print(f"\n  Conv1 HW ch-means: "
          f"{[round(float(hw_fmaps['conv1_relu'][c].mean())/512.0, 4) for c in range(min(8, hw_fmaps['conv1_relu'].shape[0]))]}")
    print(f"  Conv1 SW ch-means: "
          f"{[round(float(sw_fmaps['conv1_relu'][c].mean())/512.0, 4) for c in range(min(8, sw_fmaps['conv1_relu'].shape[0]))]}")
    print()

    # Compare layer by layer
    print(f"  {'Layer':<18} {'Shape':>16}  {'SW mean':>9} {'HW mean':>9}  "
          f"{'Max|err|':>10}  {'Match%':>8}")
    print("  " + "-" * 78)

    all_match = True
    for name, _, is_conv in sw_layers[1:]:
        sw = sw_fmaps[name].astype(np.float32) / 512.0
        hw = hw_fmaps[name].astype(np.float32) / 512.0

        err = np.abs(sw - hw)
        max_err = err.max()
        # Tolerance: 0.01 float32 ≈ 5 LSBs in Q6.9 (1 LSB = 1/512 ≈ 0.00195).
        # After the quant_col fix, max errors should be ≤1-2 LSBs.
        # A ≥90% match rate flags real problems while allowing isolated
        # border pixels or rounding-edge disagreements.
        TOLERANCE = 0.05   # ≈5 LSBs — loosen to 0.02 if border effects remain
        PASS_PCT   = 80.0  # flag if fewer than 90% of pixels match
        match_pct = 100.0 * (err < TOLERANCE).mean()
        flag = "✓" if match_pct >= PASS_PCT else "✗ MISMATCH"

        print(f"  {name:<18} {str(sw.shape):>16}  "
              f"{sw.mean():>9.4f} {hw.mean():>9.4f}  "
              f"{max_err:>10.4f}  {match_pct:>7.1f}%  {flag}")

        # For mismatching layers: print per-channel means and 5x5 sample
        if match_pct < PASS_PCT:
            all_match = False
            fmap_ch_means("SW", sw_fmaps[name], n_ch=8)
            fmap_ch_means("HW", hw_fmaps[name], n_ch=8)
            print(f"    SW sample [0, 0:5, 0:5]:")
            print(f"    {sw[0, 0:5, 0:5]}")
            print(f"    HW sample [0, 0:5, 0:5]:")
            print(f"    {hw[0, 0:5, 0:5]}")
            # Show column distribution: are cols > 0 nonzero?
            hw_raw = hw_fmaps[name]
            col0_nz  = np.count_nonzero(hw_raw[:, :, 0])
            col1p_nz = np.count_nonzero(hw_raw[:, :, 1:])
            total    = hw_raw[:, :, 0].size
            print(f"    HW col=0 nonzero: {col0_nz}/{total}  col>0 nonzero: {col1p_nz}/{hw_raw[:,:,1:].size}")

    print()
    if all_match:
        print("  ALL LAYERS MATCH ✓")
    else:
        print("  MISMATCHES DETECTED ✗")
        print("  Tolerance: err < 0.01 (~5 LSBs).  Pass threshold: 90% of pixels.")
    print("  Check bitstream canary: if not 0xC0FFEE42, old bitstream loaded.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--golden',  action='store_true', help='Run SW golden Q6.9 model')
    parser.add_argument('--float',   action='store_true', help='Run float32 dequantized model (ground truth)')
    parser.add_argument('--compare', action='store_true', help='Compare HW vs SW')
    parser.add_argument('--weights', default='/home/xilinx/weights_for_fpga.npz')
    parser.add_argument('--bit',     default='/home/xilinx/design_1.bit')
    parser.add_argument('--hwh',     default='/home/xilinx/design_1.hwh')
    args = parser.parse_args()

    if args.golden:
        run_golden(args.image, args.weights)
    elif args.float:
        run_float(args.image, args.weights)
    elif args.compare:
        run_compare(args.image, args.weights, args.bit, args.hwh)
    else:
        parser.print_help()