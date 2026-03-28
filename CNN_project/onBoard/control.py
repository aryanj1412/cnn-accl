# control.py

import os
# Note: apparently since mmap is a c extension, pylance may not detect its attributes inside venv
import mmap 
import sys # handles python interpreter
import json
import time
import numpy as np
import argparse
import math

# pylance resolved by explicit imports
from hardware_config import (
    FRAC_BITS, FRAC_SCALE, DTYPE_BYTES, INT32_MAX, INT32_MIN,
    LAYER_CNN, LAYER_MAXPOOL, LAYER_AVGPOOL, LAYER_DENSE, LAYER_TYPE_STR,
    DATATABLE_SIZE, DT_INPUT_ID, DT_OUTPUT_ID, DT_WEIGHT_ID, DT_BIAS_ID,
    REG_CTRL, REG_LAYER_TYPE, REG_IN_CHANNELS, REG_OUT_CHANNELS,
    REG_IN_HEIGHT, REG_IN_WIDTH, REG_OUT_HEIGHT, REG_OUT_WIDTH,
    REG_KERNEL_HEIGHT, REG_KERNEL_WIDTH, REG_STRIDE, REG_PADDING, REG_RESNET,
    REG_WEIGHT_XFER_SIZE, REG_BIAS_XFER_SIZE,
    REG_INPUT_XFER_SIZE, REG_OUTPUT_CHAN_XFER_SIZE,
    CTRL_START, CTRL_RESET, CTRL_OUTBUF_COMPLETED,
    CTRL_READY_FOR_NEXT_WEIGHTS, CTRL_ERROR,
    POLL_INTERVAL_S,
)

DTYPE = np.int32

try:
    import pynq
    from pynq import Overlay, allocate
except:
    print("PYNQ not available")

import fcl_total


class OpcodeParser:
    # copy of opcode, for reference
    """
    [3:0]    layer_type
    [11:4]   in_channels
    [19:12]  out_channels
    [29:20]  in_height
    [39:30]  in_width
    [49:40]  out_height
    [59:50]  out_width
    [61:60]  activation
    [64:62]  kernel_height
    [67:65]  kernel_width
    [68]     stride
    [69]     padding
    [70]     resnet
    [74:71]  input_addr_id
    [78:75]  output_addr_id
    [82:79]  weight_addr_id
    [86:83]  bias_addr_id
    [127:87] reserved
    """

    # keep the fields readonly; TODO: implement encapsulation
    def __init__(self, raw: int):
        self.layer_type      = self.raw >> 0 & 0xF
        self.in_channels     = self.raw >> 4  & 0xFF
        self.out_channels    = self.raw >> 12 & 0xFF
        self.in_height       = self.raw >> 20 & 0x3FF
        self.in_width        = self.raw >> 30 & 0x3FF
        self.out_height      = self.raw >> 40 & 0x3FF
        self.out_width       = self.raw >> 50 & 0x3FF
        self.activation      = self.raw >> 60 & 0x3
        self.kernel_height   = self.raw >> 62 & 0x7
        self.kernel_width    = self.raw >> 65 & 0x7

        if((self.raw >> 68) & 0x1):
            self.stride = 2
        else:
            self.stride = 1

        if((self.raw >> 69) & 0x1):
            self.padding = 1
        else:
            self.stride = 0

        if((self.raw >> 70) & 0x1):
            self.resnet = True
        else:
            self.resnet = False

        self.input_addr_id   = self.raw >> 71 & 0xF
        self.output_addr_id  = self.raw >> 75 & 0xF
        self.weight_addr_id  = self.raw >> 79 & 0xF
        self.bias_addr_id    = self.raw >> 83 & 0xF
    
    def layer_type_str(self):
        return LAYER_TYPE_STR.get(self.layer_type, "Unknown")


# File loading functions --------------------------------------------

def load_instructions(path: str) -> list:
    opcodes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                opcodes.append(OpcodeParser(int(line,16)))
    print(f"[INFO] Opcodes loaded from {path}")
    return opcodes

# These values are in hardware_config.py; Pylance is not resolving
DATATABLE_SIZE      = 16
DT_INPUT_ID         = 0
DT_OUTPUT_ID        = 1
DT_WEIGHT_ID        = 2
DT_BIAS_ID          = 3
def load_datatable(path: str) ->list:
    """
    Load datatables.txt 16 DDR base addresses
    Format: one entry per line as '<>: 0xADDRESS # comment'
    Return list of 16 ints

    Validations:
    No TODO enteries remain unfilled
    Weights and bias addresss are non zero
    All indices within valid range
    """
    table = [0] * DATATABLE_SIZE
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split on first colon only — handles 0x addresses safely
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue

            idx_str  = line[:colon_pos].strip()
            rest     = line[colon_pos + 1:].strip()
            addr_str = rest.split("#")[0].strip()   # strip inline comment

            if not idx_str.isdigit():
                continue

            idx = int(idx_str)
            if idx < 0 or idx >= DATATABLE_SIZE:
                raise ValueError(
                    f"datatable.txt: index {idx} out of range (0-{DATATABLE_SIZE - 1})"
                )

            if "TODO" in addr_str:
                raise ValueError(
                    f"datatable.txt entry {idx} has not been set. "
                    f"Fill in all 0xTODO entries before running control.py."
                )

            table[idx] = int(addr_str, 16)

    # Validate critical entries are non-zero
    # Zero means writing to start of DDR, corrupting Linux kernel
    if table[DT_WEIGHT_ID] == 0:
        raise ValueError(
            f"datatable.txt entry {DT_WEIGHT_ID} (weights base) is 0x00000000. "
            f"This would corrupt DDR. Set a valid address."
        )
    if table[DT_BIAS_ID] == 0:
        raise ValueError(
            f"datatable.txt entry {DT_BIAS_ID} (bias base) is 0x00000000. "
            f"This would corrupt DDR. Set a valid address."
        )

    print(f"[INFO] DataTable loaded — "
          f"weights base: 0x{table[DT_WEIGHT_ID]:08X}, "
          f"bias base:    0x{table[DT_BIAS_ID]:08X}")
    return table

def load_model_json(path: str) -> list:
    """
    Loads model.json, returns fill dict (total_weight_bytes and total_bias_bytes)
    """
    with open(path, "r") as f:
        data = json.load(f)
    print(f"model.json loaded: {len(data['layers'])} layers")
    return data

def load_weights_bin(path: str, datatable: list, model_data: dict) -> tuple:
    """
    Loads weights.bin from SSD into DDR using /dev/mem + mmap + readinto

    Note: keep w_map and b_map ALIVE throughout.
          If they go out of scope/close, the system collapses
          TODO: implement this in a more 'safe' way, the system must not collapse
    """
    weights_base = datatable[DT_WEIGHT_ID]
    bias_base = datatable[DT_BIAS_ID]
    try:
        total_weight_bytes  = model_data["total_weight_bytes"]
        total_bias_bytes    = model_data["total_bias_bytes"]
    except:
        raise ValueError(
            "Error loading size of model. Check if dictionary keys in the code are correct"
        )
    
    page_size = mmap.ALLOCATIONGRANULARITY # 4096 in Linux
    if weights_base % page_size != 0:
        raise ValueError(
            f'Weights base address 0x{weights_base:08X} is not page-aligned'
            f'(must be a multiple of {page_size}). Edit datatable.txt'
        )
    if bias_base % page_size != 0:
        raise ValueError(
            f"Bias base address 0x{bias_base:08X} is not page-aligned "
            f"(must be a multiple of {page_size}). Edit datatable.txt."
        )

    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)

    # Map weights region and read directly from file into DDR
    w_map = mmap.mmap(fd, total_weight_bytes,
                      mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                      offset=weights_base)
    with open(path, "rb") as f:
        f.readinto(w_map)       # direct SD card → DDR, no intermediate copy
    w_map.seek(0)
    weights_ddr = np.frombuffer(w_map, dtype=np.int32,
                                count=total_weight_bytes // DTYPE_BYTES)

    # Map bias region and read bias section directly into DDR
    b_map = mmap.mmap(fd, total_bias_bytes,
                      mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                      offset=bias_base)
    with open(path, "rb") as f:
        f.seek(total_weight_bytes)  # skip weights section
        f.readinto(b_map)           # SD card → DDR
    b_map.seek(0)
    biases_ddr = np.frombuffer(b_map, dtype=np.int32,
                               count=total_bias_bytes // DTYPE_BYTES)

    os.close(fd)
    print(f"[INFO] Weights loaded into DDR: 0x{weights_base:08X} "
          f"({total_weight_bytes} bytes)")
    print(f"[INFO] Biases  loaded into DDR: 0x{bias_base:08X} "
          f"({total_bias_bytes} bytes)")

    return weights_ddr, biases_ddr, w_map, b_map

    
def load_image(path:str) -> np.ndarray:
    """
    Conversion to int32 fixed point format must be done in laptop
    """
    img = np.fromfile(path, dtype=np.int32)
    print(f"[INFO] Image Loaded; {len(img)} int32 values ({img.nbytes}) bytes")

# AXI Lite Register Interface
class AcceleratorMMIO:
    """
    The instructions (stored in AXI Lite Register are communicated to hardware via functins of this class
    PYNQ mmio is used to communicate
    """
    def __init__(self):
        self._mmio = self._mmio
    
    def write(self, offset: int, value: int):
        if self._mmio is not None:
            self._mmio.write(offset, int(value))
        else:
            raise AttributeError(
                f"Tried to Write config registers",
                f"self._mmio is None"
            )
            return 0

    def read(self, offset:int) -> int:
        if self._mmio is not None:
            return self._mmio(offset)
        else:
            raise AttributeError(
                f"Tried to Write config registers",
                f"self._mmio is None"
            )
            return 0
    
    def write_config(self, op: OpcodeParser, meta: dict):
        self.write(REG_LAYER_TYPE,            op.layer_type)
        self.write(REG_IN_CHANNELS,           op.in_channels)
        self.write(REG_OUT_CHANNELS,          op.out_channels)
        self.write(REG_IN_HEIGHT,             op.in_height)
        self.write(REG_IN_WIDTH,              op.in_width)
        self.write(REG_OUT_HEIGHT,            op.out_height)
        self.write(REG_OUT_WIDTH,             op.out_width)
        self.write(REG_KERNEL_HEIGHT,         op.kernel_height)
        self.write(REG_KERNEL_WIDTH,          op.kernel_width)
        self.write(REG_STRIDE,                0 if op.stride == 1 else 1)
        self.write(REG_PADDING,               op.padding)
        self.write(REG_RESNET,                1 if op.resnet else 0)
        self.write(REG_WEIGHT_XFER_SIZE,      meta["weight_per_out_channel_bytes"])
        self.write(REG_BIAS_XFER_SIZE,        meta["bias_byte_size"])
        self.write(REG_INPUT_XFER_SIZE,       meta["input_feature_map_bytes"])
        self.write(REG_OUTPUT_CHAN_XFER_SIZE, meta["output_channel_bytes"])

    def assert_start(self):
        self.write(REG_CTRL, CTRL_START)

    def assert_reset(self):
        self.write(REG_CTRL, CTRL_RESET)

    def poll_flag(self, flag_mask: int):
        """
        Poll CTRL register until flag_mask bit is set
        """
        if self._mmio is None:
            raise AttributeError(
                f"Tried to poll flag",
                f"self._mmio is None"
            )
            return
        while True:
            ctrl = self.read(REG_CTRL)
            if ctrl & CTRL_ERROR:
                raise RuntimeError(
                    f"PL asserted ERROR flag while polling",
                    f"mask=0x{flag_mask:02X}. Check main.v error state"
                )
            if ctrl & flag_mask:
                return
            time.sleep(POLL_INTERVAL_S)

    def clear_flag(self, flag_mask: int):
        """
        Clear a flag in CTRL register (read-modify-write)
        """
        if self._mmio is None:
            return
        ctrl = self.read(REG_CTRL)
        self.write(REG_CTRL, ctrl & ~flag_mask)


# Layer Metadata functions
def build_layer_meta(op: OpcodeParser, json_layer: dict) -> dict:
    """
    Compute all transfer sizes and offsets for a layer
    Dense layers's weights sliced directly by fcl_total.py 
    (Not great design, but fcl_total.py is already made and dense layers run on PS)
    """
    is_cnn = (op.layer_type == LAYER_CNN)

    if is_cnn:
        weight_per_out_channel_bytes = op.in_channels*op.kernel_height*op.kernel_width * DTYPE_BYTES
        bias_byte_size = op.out_channels * DTYPE_BYTES
    else:
        weight_per_out_channel_bytes = 0
        bias_byte_size = 0

    input_feature_map_bytes = op.in_channels*op.in_height*op.in_width*DTYPE_BYTES
    output_channel_bytes = op.out_height * op.out_width * DTYPE_BYTES
    
    return {
        "weight_per_out_channel_bytes": weight_per_out_channel_bytes,
        "bias_byte_size":               bias_byte_size,
        "input_feature_map_bytes":      input_feature_map_bytes,
        "output_channel_bytes":         output_channel_bytes,
        "weight_byte_offset":           json_layer.get("weight_byte_offset", 0),
        "bias_byte_offset":             json_layer.get("bias_byte_offset",   0),
        "weight_byte_size":             json_layer.get("weight_byte_size",   0),
    }

def run_pl_layer(op:            OpcodeParser,
                 meta:          dict,
                 mmio:          AcceleratorMMIO,
                 dma0,
                 dma1,
                 weights_raw:   np.ndarray,
                 biases_raw:    np.ndarray,
                 input_data:    np.ndarray):
    """
    Dispatch 1 CNN or Pool layer to the PL accelerator

    DMA0 stream format per output channel (CNN only):
    [weights group0]...[weights last group][bias (1 x int32)]
    Bias is the last word per transfer. PL uses WEIGHT_XFER_SIZE to locate it
    Bias added  + ReLU applied after last partial sum for the output channel
    Pool layers skip DMA0

    Both DMA are waited on before asserting start to gurantee data is present on PL's AXi stream ports before begining
    """

    out_channels  = op.out_channels
    is_cnn        = (op.layer_type == LAYER_CNN)

    w_per_ch      = meta["weight_per_out_channel_bytes"] // DTYPE_BYTES
    w_offset_elem = meta["weight_byte_offset"]   // DTYPE_BYTES
    b_offset_elem = meta["bias_byte_offset"]     // DTYPE_BYTES
    out_ch_elems  = meta["output_channel_bytes"] // DTYPE_BYTES

    # ── Ping-pong row group parameters — must match main.v BRAM_ROWS parameter ─
    BRAM_ROWS   = 8
    kernel_h    = op.kernel_height
    n_rows      = BRAM_ROWS - kernel_h + 1          # useful output rows per group
    n_rowgroups = math.ceil(op.out_height / n_rows)  # row groups per input channel
    n_chgroups  = math.ceil(op.in_channels / 16)     # N_PE=16 channels per group

    # One row group DMA1 transfer = BRAM_ROWS rows x in_width pixels
    row_group_elems = BRAM_ROWS * op.in_width

    print(f"  [PL] {op.layer_type_str()} layer: "
          f"in={op.in_channels}x{op.in_height}x{op.in_width} "
          f"out={op.out_channels}x{op.out_height}x{op.out_width} "
          f"n_rowgroups={n_rowgroups} n_chgroups={n_chgroups}")

    # Extract this layer's weights and biases from DDR-backed arrays (CNN only)
    if is_cnn:
        layer_weights = weights_raw[w_offset_elem : w_offset_elem + w_per_ch * out_channels]
        layer_biases  = biases_raw[b_offset_elem  : b_offset_elem + out_channels]

    # ── Step 1: Write config registers ───────────────────────────────────────
    mmio.write_config(op, meta)

    # ── Step 2: Allocate DMA-safe buffers ─────────────────────────────────────
    # input_buf sized for ONE row group (BRAM_ROWS rows) not the full feature map
    if PYNQ_AVAILABLE:
        input_buf  = allocate(shape=(row_group_elems,), dtype=DTYPE)
        output_buf = allocate(shape=(out_ch_elems,),    dtype=DTYPE)
        if is_cnn:
            weight_buf = allocate(shape=(w_per_ch + 1,), dtype=DTYPE)
    else:
        input_buf  = np.zeros(row_group_elems, dtype=DTYPE)
        output_buf = np.zeros(out_ch_elems,    dtype=DTYPE)
        if is_cnn:
            weight_buf = np.zeros(w_per_ch + 1, dtype=DTYPE)

    # Reshape input_data to (in_channels, in_height, in_width) for clean slicing
    input_3d = input_data.reshape(op.in_channels, op.in_height, op.in_width)

    # ── Step 3: Assert start ──────────────────────────────────────────────────
    # main.v starts immediately and asserts READY_FOR_NEXT_WEIGHTS to request
    # the first DMA1 row group transfer. PS must not fire any DMA before start —
    # unlike the old architecture, the PL now drives the DMA1 timing via flags.
    mmio.assert_start()

    # ── Step 4: Fire first DMA0 (output channel 0 weights + bias) ────────────
    # DMA0 timing is unchanged — one transfer per output channel.
    # First transfer fires here; subsequent ones fire at end of each out_ch loop.
    if is_cnn:
        weight_buf[:w_per_ch] = layer_weights[0 : w_per_ch]
        weight_buf[w_per_ch]  = layer_biases[0]
        if PYNQ_AVAILABLE:
            dma0.sendchannel.transfer(weight_buf)
            dma0.sendchannel.wait()

    # ── Step 5: Main inference loop ───────────────────────────────────────────
    # Loop order matches main.v exactly:
    #   outer: output channel
    #   middle: input channel group (ch_group)
    #   inner: row group
    #
    # For each row group, PL asserts READY_FOR_NEXT_WEIGHTS → PS fires DMA1.
    # After all ch_groups and row_groups for one out_ch, PL asserts
    # OUTBUF_COMPLETED → PS fires DMA1 MM2S to pull the output channel.
    full_output = np.zeros(out_channels * out_ch_elems, dtype=DTYPE)

    for out_ch in range(out_channels):
        for ch_grp in range(n_chgroups):
            for rg in range(n_rowgroups):

                # PL signals it is ready for the next row group DMA1 transfer
                mmio.poll_flag(CTRL_READY_FOR_NEXT_WEIGHTS)
                mmio.clear_flag(CTRL_READY_FOR_NEXT_WEIGHTS)

                # Compute row range for this row group
                # rg=0: rows 0 .. BRAM_ROWS-1  (no overlap prefix)
                # rg>0: rows (rg*n_rows - overlap) .. (rg*n_rows - overlap + BRAM_ROWS - 1)
                #        first (kernel_h-1) rows are the overlap from previous group
                overlap   = kernel_h - 1
                row_start = rg * n_rows - (overlap if rg > 0 else 0)
                row_end   = min(row_start + BRAM_ROWS, op.in_height)
                actual_rows = row_end - row_start

                # Extract channels for this ch_grp, rows for this row_group
                # input_3d shape: (in_channels, in_height, in_width)
                ch_start  = ch_grp * 16
                ch_end    = min(ch_start + 16, op.in_channels)
                row_slice = input_3d[ch_start:ch_end,
                                     row_start:row_end, :]   # (ch, rows, width)
                row_data  = row_slice.flatten().astype(DTYPE)

                input_buf[:len(row_data)] = row_data
                if PYNQ_AVAILABLE:
                    dma1.sendchannel.transfer(input_buf[:len(row_data)])
                    dma1.sendchannel.wait()

        # All ch_groups done for this output channel
        # PL signals output channel is ready
        mmio.poll_flag(CTRL_OUTBUF_COMPLETED)

        if PYNQ_AVAILABLE:
            dma1.recvchannel.transfer(output_buf)
            dma1.recvchannel.wait()

        out_start = out_ch * out_ch_elems
        full_output[out_start : out_start + out_ch_elems] = output_buf[:]
        mmio.clear_flag(CTRL_OUTBUF_COMPLETED)

        # Pre-fire DMA0 for next output channel while current result is being pulled
        if is_cnn and out_ch + 1 < out_channels:
            nxt = out_ch + 1
            weight_buf[:w_per_ch] = layer_weights[nxt * w_per_ch : (nxt + 1) * w_per_ch]
            weight_buf[w_per_ch]  = layer_biases[nxt]
            if PYNQ_AVAILABLE:
                dma0.sendchannel.transfer(weight_buf)
                dma0.sendchannel.wait()

    # ── Step 6: Free buffers ──────────────────────────────────────────────────
    if PYNQ_AVAILABLE:
        input_buf.freebuffer()
        output_buf.freebuffer()
        if is_cnn:
            weight_buf.freebuffer()

    # ── Step 7: Flatten in C,H,W row-major order (PyTorch-compatible) ─────────
    full_output = full_output.reshape(
        out_channels, op.out_height, op.out_width
    ).flatten()

    return full_output

# ─── FCL Dispatcher ──────────────────────────────────────────────────────────

def run_fcl_layers(fcl_ops:      list,
                   json_layers:  list,
                   weights_raw:  np.ndarray,
                   biases_raw:   np.ndarray,
                   input_data:   np.ndarray) -> np.ndarray:
    """
    Batch all consecutive FCL layers and dispatch to fcl_total.py

    IMPORTANT: FCL neuron counts are read from model.json, NOT from opcode
    in_channels/out_channels. Per the opcode spec, dense layers encode
    in_channels = out_channels = 1. Actual sizes (e.g. 8192, 10) are stored
    in model.json under 'in_channels' and 'out_channels' by the compiler.

    Args:
        fcl_ops:     list of OpcodeParser for consecutive FCL layers
        json_layers: corresponding model.json layer dicts
        weights_raw: full weights array (DDR-backed or numpy in simulation)
        biases_raw:  full biases array
        input_data:  1D int32 array from previous PL layer or image

    Returns:
        1D float64 softmax probabilities
    """
    print(f"  [PS] Running {len(fcl_ops)} FCL layer(s) on PS")

    # Build control list from model.json actual neuron counts
    control = [json_layers[0]["in_channels"]]
    for jl in json_layers:
        control.append(jl["out_channels"])

    resnet_flags = [op.resnet for op in fcl_ops]

    all_weights = []
    all_biases  = []

    for op, jl in zip(fcl_ops, json_layers):
        in_size  = jl["in_channels"]
        out_size = jl["out_channels"]

        w_offset = jl["weight_byte_offset"] // DTYPE_BYTES
        b_offset = jl["bias_byte_offset"]   // DTYPE_BYTES
        w_count  = in_size * out_size
        b_count  = out_size

        all_weights.append(weights_raw[w_offset : w_offset + w_count])
        all_biases.append(biases_raw[b_offset   : b_offset + b_count])

    weights_concat = np.concatenate(all_weights).astype(DTYPE)
    biases_concat  = np.concatenate(all_biases).astype(DTYPE)

    return fcl_total(
        control       = control,
        resnet_flags  = resnet_flags,
        weights_total = weights_concat,
        bias_total    = biases_concat,
        data          = input_data
    )


# Inference 
def run_inference(bitstream_path:    str,
                  image_path:        str,
                  instructions_path: str,
                  datatable_path:    str,
                  model_json_path:   str,
                  weights_bin_path:  str):

    print("Inference")

    # Load bitstream 
    print("[STEP 1] Loading bitstream...")
    overlay = Overlay(bitstream_path)
    dma0    = overlay.axi_dma_0          # weights + biases (S2MM only)
    dma1    = overlay.axi_dma_1          # feature maps (S2MM + MM2S)
    mmio_hw = overlay.accelerator_mmio   # AXI Lite — update name to match Vivado IP name
    mmio    = AcceleratorMMIO(mmio_hw)
    mmio.assert_reset()
    print(f"  Bitstream loaded: {bitstream_path}")


    # Load compiler outputs 
    print("[STEP 2] Loading compiler outputs...")
    datatable    = load_datatable(datatable_path)
    instructions = load_instructions(instructions_path)
    model_data   = load_model_json(model_json_path)
    json_layers  = model_data["layers"]

    # Load weights into DDR 
    print("[STEP 3] Loading weights into DDR...")
    weights_raw, biases_raw, w_map, b_map = load_weights_bin(
        weights_bin_path, datatable, model_data
    )
    # w_map and b_map held here for entire inference duration.
    # They must not go out of scope — if they do the mmap closes and
    # weights_raw/biases_raw become invalid numpy views over freed memory.

    # Load image 
    print("[STEP 4] Loading input image...")
    current_data = load_image(image_path)

    # Run inference layer by layer 
    print("[STEP 5] Running inference...\n")

    fcl_op_buffer   = []
    fcl_json_buffer = []

    for idx, op in enumerate(instructions):
        json_layer = json_layers[idx]

        if op.layer_type == LAYER_DENSE:
            # Buffer consecutive FCL layers for batch dispatch
            fcl_op_buffer.append(op)
            fcl_json_buffer.append(json_layer)

        else:
            # Flush any buffered FCL layers before dispatching to PL
            if fcl_op_buffer:
                current_data = run_fcl_layers(
                    fcl_ops     = fcl_op_buffer,
                    json_layers = fcl_json_buffer,
                    weights_raw = weights_raw,
                    biases_raw  = biases_raw,
                    input_data  = current_data
                )
                fcl_op_buffer   = []
                fcl_json_buffer = []

            meta = build_layer_meta(op, json_layer)
            current_data = run_pl_layer(
                op          = op,
                meta        = meta,
                mmio        = mmio,
                dma0        = dma0,
                dma1        = dma1,
                weights_raw = weights_raw,
                biases_raw  = biases_raw,
                input_data  = current_data
            )

    # Flush any remaining FCL layers at end of network
    if fcl_op_buffer:
        current_data = run_fcl_layers(
            fcl_ops     = fcl_op_buffer,
            json_layers = fcl_json_buffer,
            weights_raw = weights_raw,
            biases_raw  = biases_raw,
            input_data  = current_data
        )

    top_class = int(np.argmax(current_data))

    print("\n[RESULT] Softmax probabilities:")
    for i, prob in enumerate(current_data):
        marker = "  ◄ top" if i == top_class else ""
        print(f"  Class {i:>3d} : {prob:.6f}{marker}")
    print(f"  Predicted class : {top_class}")
    print(f"  Confidence      : {float(np.max(current_data)):.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on CNN accelerator (PYNQ Z2)"
    )
    parser.add_argument("--bitstream",    required=True,
                        help="Path to .bit bitstream file")
    parser.add_argument("--image",        required=True,
                        help="Path to raw fixed-point binary image (int32)")
    parser.add_argument("--instructions", required=True,
                        help="Path to instructions.hex from compiler.py")
    parser.add_argument("--datatable",    required=True,
                        help="Path to datatable.txt (fill in 0xTODO entries first)")
    parser.add_argument("--model_json",   required=True,
                        help="Path to model.json from compiler.py")
    parser.add_argument("--weights_bin",  required=True,
                        help="Path to weights.bin from compiler.py")

    args = parser.parse_args()

    run_inference(
        bitstream_path    = args.bitstream,
        image_path        = args.image,
        instructions_path = args.instructions,
        datatable_path    = args.datatable,
        model_json_path   = args.model_json,
        weights_bin_path  = args.weights_bin
    )