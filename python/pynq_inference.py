# ============================================================
# File    : pynq_inference.py
# Target  : PYNQ-Z2 (ARM Cortex-A9 PS)
# Purpose : Full CNN inference driver
#
# PL handles : Conv1-4 + ReLU + MaxPool
# PS handles : Flatten + FC1 + ReLU + FC2 + Sigmoid
#
# PS/PL Handshake:
#   PS polls LOOP_STAT[25:24] = fsm_wait_state
#     00 = FSM busy computing, not waiting for DMA
#     01 = FSM waiting for WEIGHTS  → PS sends 144 weights
#     10 = FSM waiting for PIXELS   → PS sends 4xchxW pixels
#   After sending, FSM receives TLAST and continues
#
# Usage:
#   from pynq_inference import CNNInference
#   cnn = CNNInference()
#   prob, label = cnn.predict('/home/xilinx/test.jpg')
#
# Files needed on PYNQ at /home/xilinx/:
#   design_1.bit
#   design_1.hwh
#   weights_for_fpga.npz
# ============================================================

import numpy as np
from pynq import Overlay, allocate
from PIL import Image
import time
import os


# ============================================================
# AXI-Lite Register Map
# Matches axi_lite_slave.v exactly
# ============================================================

REG_CTRL        = 0x00   # [0]=start  [2:1]=mode
REG_STATUS      = 0x04   # [0]=done   [1]=busy
REG_IMG_DIM     = 0x08   # [7:0]=in_h [15:8]=in_w
REG_OUT_DIM     = 0x0C   # [7:0]=out_h [15:8]=out_w
REG_CH_CFG      = 0x10   # [7:0]=num_in_ch [15:8]=num_out_ch
REG_LOOP_STAT   = 0x20   # [7:0]=cur_oc  [15:8]=cur_group
                          # [23:16]=cur_row_pair  (BUG 5 FIX: full 8-bit, was 7-bit)
                          # [25:24]=fsm_wait_state

# Mode encoding for CTRL[2:1]
MODE_CONV_RELU  = 0b01   # conv + relu fused
MODE_POOL       = 0b10   # maxpool (hardware - NOT USED, pool runs in PS)
MODE_PS_POOL    = 0b11   # sentinel: run this layer as PS numpy pool, not PL

# FSM wait state encoding from LOOP_STAT[25:24]
WAIT_NONE       = 0b00   # FSM computing, not waiting
WAIT_WEIGHTS    = 0b01   # FSM waiting for weight DMA
WAIT_PIXELS     = 0b10   # FSM waiting for pixel DMA


# ============================================================
# CNN Layer Configuration
# name, in_h, in_w, out_h, out_w, in_ch, out_ch, mode
# ============================================================

LAYER_CONFIG = [
    ('conv1_relu', 128, 128, 126, 126,  3,  16, MODE_CONV_RELU),
    ('pool1',      126, 126,  63,  63, 16,  16, MODE_PS_POOL),
    ('conv2_relu',  63,  63,  61,  61, 16,  32, MODE_CONV_RELU),
    ('pool2',       61,  61,  30,  30, 32,  32, MODE_PS_POOL),
    ('conv3_relu',  30,  30,  28,  28, 32,  64, MODE_CONV_RELU),
    ('pool3',       28,  28,  14,  14, 64,  64, MODE_PS_POOL),
    ('conv4_relu',  14,  14,  12,  12, 64,  96, MODE_CONV_RELU),
    ('pool4',       12,  12,   6,   6, 96,  96, MODE_PS_POOL),
]


class CNNInference:

    def __init__(
        self,
        bitstream_path = '/home/xilinx/design_1.bit',
        weights_path   = '/home/xilinx/weights_for_fpga.npz',
        verbose        = True
    ):
        self.verbose = verbose
        self._log("Initializing CNN Accelerator...")

        # -----------------------------------------------
        # Load bitstream and get IP handles
        # -----------------------------------------------
        self._log(f"Loading bitstream: {bitstream_path}")
        self.overlay = Overlay(bitstream_path)
        self.dma     = self.overlay.axi_dma_0
        # Accelerator is plain RTL (not in IP dict) — access via MMIO directly.
        # Base address 0x43C00000 confirmed from Vivado address editor.
        from pynq import MMIO
        self._cnn_mmio = MMIO(0x43C00000, 0x10000)
        self._log("Bitstream loaded ✔")
        self._log(f"DMA IP type : {type(self.dma)}")
        self._log(f"CNN MMIO   : 0x43C00000")

        # -----------------------------------------------
        # Load weights
        # -----------------------------------------------
        self._log(f"Loading weights: {weights_path}")
        self._load_weights(weights_path)
        self._log("Weights loaded ✔")

        # Verify new bitstream is loaded (canary register at 0x3C)
        canary = self._rd(0x3C)
        if canary == 0xC0FFEE42:
            self._log("Bitstream version: NEW (canary 0xC0FFEE42 confirmed) ✓")
        else:
            self._log(f"WARNING: Canary = 0x{canary:08X} (expected 0xC0FFEE42)")
            self._log("WARNING: This may be an OLD bitstream! Recompile and re-SCP.")
        self._log("Ready!\n")

    # -----------------------------------------------
    # Logging
    # -----------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # -----------------------------------------------
    # Register helpers
    # -----------------------------------------------

    def _wr(self, offset, value):
        self._cnn_mmio.write(offset, int(value))

    def _rd(self, offset):
        return self._cnn_mmio.read(offset)

    def _read_loop_stat(self):
        val            = self._rd(REG_LOOP_STAT)
        cur_oc         = (val >>  0) & 0xFF
        cur_group      = (val >>  8) & 0xFF
        cur_row_pair   = (val >> 16) & 0xFF   # BUG 5 FIX: was 0x7F (7-bit); now full 8-bit
        fsm_wait_state = (val >> 24) & 0x03
        return (int(cur_oc), int(cur_group),
                int(cur_row_pair), int(fsm_wait_state))

    def _wait_done(self, timeout_s=60):
        t0 = time.time()
        while True:
            if self._rd(REG_STATUS) & 0x1:
                return
            if time.time() - t0 > timeout_s:
                raise TimeoutError(
                    "Accelerator timed out! "
                    "Check DMA connections and bitstream."
                )
            time.sleep(0.00005)

    # -----------------------------------------------
    # Load weights from npz
    # -----------------------------------------------

    def _load_weights(self, path):
        w = np.load(path)

        # Conv weights: Q6.9 int16, HW format [out_ch, groups, 16, 9]
        self.conv_weights = {
            'conv1': w['conv1_weight'],   # (16, 1, 16, 9)
            'conv2': w['conv2_weight'],   # (32, 1, 16, 9)
            'conv3': w['conv3_weight'],   # (64, 2, 16, 9)
            'conv4': w['conv4_weight'],   # (96, 4, 16, 9)
        }

        # FC weights: float32 for PS computation
        self.fc1_w = w['fc1_weight']     # (128, 3456)
        self.fc1_b = w['fc1_bias']       # (128,)
        self.fc2_w = w['fc2_weight']     # (1, 128)
        self.fc2_b = w['fc2_bias']       # (1,)

        # Normalization (same as training)
        self.norm_mean = w['norm_mean']  # [0.485, 0.456, 0.406]
        self.norm_std  = w['norm_std']   # [0.229, 0.224, 0.225]

        if self.verbose:
            for k, v in self.conv_weights.items():
                print(f"  {k}: {v.shape} {v.dtype}")
            print(f"  fc1: {self.fc1_w.shape}  "
                  f"fc2: {self.fc2_w.shape}")

    # -----------------------------------------------
    # Fixed point conversion
    # -----------------------------------------------

    def _to_q69(self, arr):
        return np.clip(
            arr * 512, -32768, 32767
        ).astype(np.int16)

    def _from_q69(self, arr):
        return arr.astype(np.float32) / 512.0

    # -----------------------------------------------
    # Image preprocessing
    # Must exactly match training transforms:
    #   resize(128) → /255 → normalize → Q6.9
    # -----------------------------------------------

    def _preprocess(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((128, 128), Image.BILINEAR)
        img = np.array(img).astype(np.float32)    # HWC uint8→float

        # Normalize 0..255 → 0..1
        img = img / 255.0

        # Standardize: same mean/std as training
        img = (img - self.norm_mean) / self.norm_std

        # HWC → CHW: [128,128,3] → [3,128,128]
        img = img.transpose(2, 0, 1)

        # float32 → Q6.9 int16
        return self._to_q69(img)                  # [3,128,128] int16

    # -----------------------------------------------
    # DMA helpers
    # -----------------------------------------------

    def _dma_send(self, data_1d):
        # Send 1D int16 array to PL via MM2S channel
        # Uses register polling instead of interrupt-based wait
        # because s2mm_introut UIO device is not available on PYNQ v2.6
        #
        # AXI DMA MM2S Status Register = offset 0x04
        # Bit 1 (Idle) = 1 means transfer complete
        buf    = allocate(shape=data_1d.shape, dtype=np.int16)
        buf[:] = data_1d
        try:
            buf.flush()  # ensure PS writes visible to PL DMA
        except Exception:
            pass
        self.dma.sendchannel.transfer(buf)
        t0 = time.time()
        while True:
            # BUG 9 FIX: was "sr & 0x1002" — 0x1000 is the IOC interrupt flag,
            # only set when DMA interrupts are explicitly enabled in the CR register.
            # Polling it when interrupts are OFF means the bit never sets and the
            # loop can miss completion or behave differently across DMA configs.
            # The safe, config-independent check is bit 1 (Idle) only = 0x0002.
            sr = self.dma.mmio.read(0x04)    # MM2S status register
            if sr & 0x0002:                   # Idle bit only
                break
            if time.time() - t0 > 5:
                raise TimeoutError(
                    f"DMA send timeout SR={hex(sr)}"
                )
            time.sleep(0.000001)
        buf.freebuffer()

    def _dma_recv(self, n_pixels):
        # Receive int16 array from PL via S2MM channel
        # Uses register polling instead of interrupt-based wait
        #
        # AXI DMA S2MM Status Register = offset 0x34
        # Bit 1 (Idle) = 1 means transfer complete
        buf = allocate(shape=(n_pixels,), dtype=np.int16)
        self.dma.recvchannel.transfer(buf)
        t0 = time.time()
        while True:
            # BUG 9 FIX: use Idle bit (0x0002) only — same reasoning as _dma_send
            sr = self.dma.mmio.read(0x34)    # S2MM status register
            if sr & 0x0002:                   # Idle bit only
                break
            if time.time() - t0 > 5:
                raise TimeoutError(
                    f"DMA recv timeout SR={hex(sr)}"
                )
            time.sleep(0.000001)
        result = np.array(buf, dtype=np.int16)
        buf.freebuffer()
        return result

    # -----------------------------------------------
    # Build weight vector for one (oc, group)
    # Shape [16, 9] → flatten to [144]
    # -----------------------------------------------

    def _get_weights_flat(self, weights, cur_oc, cur_group):
        return weights[cur_oc, cur_group].flatten()

    # -----------------------------------------------
    # Build pixel block for one (group, row_pair)
    # Gathers 4 rows × 16 channels → flatten
    # -----------------------------------------------

    def _build_pixel_block(
        self, fmap, cur_group, cur_row_pair,
        in_h, in_w, num_in_ch
    ):
        # 4 input rows needed for 2 output rows
        # row_pair r → input rows r*2, r*2+1, r*2+2, r*2+3
        base = cur_row_pair * 2
        rows = [min(base + r, in_h - 1) for r in range(4)]

        # 16 channels in this group (padded channels stay zero)
        ch_start = cur_group * 16
        ch_end   = min(ch_start + 16, num_in_ch)

        # [16ch, 4rows, in_w]
        block = np.zeros((16, 4, in_w), dtype=np.int16)
        for i, ch in enumerate(range(ch_start, ch_end)):
            for ri, row in enumerate(rows):
                block[i, ri, :] = fmap[ch, row, :]

        # Flatten: ch0_row0_allcols, ch0_row1_allcols...
        #          ch1_row0_allcols, ch1_row1_allcols...
        return block.reshape(-1)

    # -----------------------------------------------
    # Run 2×2 max pool layer on PS (numpy)
    # Faster and correct for all channel counts
    # Input:  fmap [C, H, W] int16
    # Output: fmap [C, H//2, W//2] int16
    # -----------------------------------------------
    def _run_ps_pool(self, name, fmap, out_h, out_w):
        self._log(f"  {name}...")
        t0 = time.time()
        # Ensure fmap is in cached CPU memory before numpy ops
        if not fmap.flags['OWNDATA'] or not fmap.flags['C_CONTIGUOUS']:
            fmap = np.ascontiguousarray(fmap, dtype=np.int16)
        C, H, W = fmap.shape
        # Crop to even dimensions before slicing into 2x2 windows.
        # Odd spatial dims (e.g. 61×61) cause shape mismatch between
        # [0::2] (31 elements) and [1::2] (30 elements).
        H2, W2 = (H // 2) * 2, (W // 2) * 2
        f = fmap[:, :H2, :W2]
        p = np.maximum(
            np.maximum(f[:, 0::2, 0::2], f[:, 0::2, 1::2]),
            np.maximum(f[:, 1::2, 0::2], f[:, 1::2, 1::2])
        ).astype(np.int16)
        self._log(f" {int((time.time()-t0)*1000)}ms  out={p.shape}")
        return p

    # -----------------------------------------------
    # Run one hardware layer
    # Uses fsm_wait_state handshake for correct timing
    # -----------------------------------------------

    def _run_hw_layer(
        self, name, fmap,
        in_h, in_w, out_h, out_w,
        num_in_ch, num_out_ch, mode,
        wgt_key=None
    ):
        self._log(f"  {name}...")
        t0 = time.time()

        num_groups    = (num_in_ch  + 15) // 16
        is_pool       = (mode == MODE_POOL)
        # Pool: one output row per 2-row input pair → iterate over all output_h pairs
        # Conv: two output rows per pair → iterate over output_h/2 pairs
        num_row_pairs = out_h if is_pool else (out_h + 1) // 2
        n_out         = out_h * out_w

        # Weights for this layer
        if wgt_key is not None:
            weights = self.conv_weights[wgt_key]
        else:
            weights = np.zeros(
                (num_out_ch, num_groups, 16, 9), dtype=np.int16
            )

        out_fmap = np.zeros(
            (num_out_ch, out_h, out_w), dtype=np.int16
        )

        # Configure PL registers
        self._wr(REG_IMG_DIM, (int(in_w)      << 8) | int(in_h))
        self._wr(REG_OUT_DIM, (int(out_w)      << 8) | int(out_h))
        self._wr(REG_CH_CFG,  (int(num_out_ch) << 8) | int(num_in_ch))

        
        # Start accelerator
        self._wr(REG_CTRL, (int(mode) << 1) | 0x1)

        # Pre-allocate reusable recv buffer
        recv_buf = allocate(shape=(n_out,), dtype=np.int16)

        # -----------------------------------------------
        # PS-controlled nested loop matching FSM order:
        #   outer:  cur_oc        (0..num_out_ch-1)
        #   middle: cur_row_pair  (0..num_row_pairs-1)
        #   inner:  cur_group     (0..num_groups-1)
        #
        # IMPORTANT: cur_row_pair in LOOP_STAT resets to 0
        # inside NEXT_ROW_PAIR state before FSM streams output.
        # PS CANNOT read cur_row_pair to detect last row_pair.
        # PS uses its own loop variables instead.
        #
        # S2MM must be armed BEFORE last pixel DMA send
        # so receiver is ready when FSM starts streaming.
        # -----------------------------------------------
        for oc in range(num_out_ch):
            for rp in range(num_row_pairs):
                for grp in range(num_groups):

                    is_last_grp = (grp == num_groups    - 1)
                    is_last_rp  = (rp  == num_row_pairs - 1)

                    # 1. Wait FSM wants weights
                    self._wait_for_fsm(WAIT_WEIGHTS)

                    # 2. Send weights (144 int16 values)
                    wgt = weights[oc, grp].flatten()
                    self._dma_send(wgt)

                    # 3. Wait FSM wants pixels
                    self._wait_for_fsm(WAIT_PIXELS)

                    # 4. Arm S2MM before last pixel block
                    #    so DMA receiver is ready when FSM streams
                    if is_last_grp and is_last_rp:
                        self.dma.recvchannel.transfer(recv_buf)

                    # 5. Send pixels (16ch x 4rows x in_w values)
                    pix = self._build_pixel_block(
                        fmap, grp, rp, in_h, in_w, num_in_ch
                    )
                    self._dma_send(pix)

                # After last group of last row_pair:
                # wait for S2MM to complete (FSM streamed output)
                if is_last_rp:
                    t0r = time.time()
                    while True:
                        # BUG 9 FIX: Idle bit (0x0002) only
                        sr = self.dma.mmio.read(0x34)
                        if sr & 0x0002:   # S2MM Idle bit
                            break
                        if time.time() - t0r > 10:
                            loop = self._rd(REG_LOOP_STAT)
                            stat = self._rd(REG_STATUS)
                            raise TimeoutError(
                                f"S2MM timeout oc={oc} SR={hex(sr)} "
                                f"LOOP={hex(loop)} STAT={hex(stat)}"
                            )
                        time.sleep(0.000001)
                    # Force copy from DMA buffer (recv_buf is uncached memory).
                    # Without explicit copy, numpy may hold a view of uncached DDR
                    # which makes subsequent numpy ops (e.g. pool) extremely slow.
                    # Sync cache before reading DMA buffer
                    try:
                        recv_buf.invalidate()
                    except Exception:
                        pass  # non-cached buffer, no action needed
                    out_fmap[oc] = np.array(recv_buf, dtype=np.int16).reshape(out_h, out_w)


        recv_buf.freebuffer()
        # All OCs done - wait for done_latch to be set by FSM ALL_DONE state
        # Poll with short timeout (FSM should be in ALL_DONE within microseconds
        # of last S2MM completing, since stream_done -> NEXT_OC -> ALL_DONE
        # takes only ~3 clock cycles)
        t0d = time.time()
        while True:
            stat = self._rd(REG_STATUS)
            if stat & 0x1:   # done_latch set
                break
            if time.time() - t0d > 5:
                loop = self._rd(REG_LOOP_STAT)
                raise TimeoutError(
                    f"Layer done timeout: STAT={hex(stat)} LOOP={hex(loop)}"
                )
            time.sleep(0.000050)

        elapsed = (time.time() - t0) * 1000
        self._log(f" {elapsed:.0f}ms  out={out_fmap.shape}")
        return out_fmap

    def _wait_for_fsm(self, target_state, timeout_s=10):
        t0 = time.time()
        while True:
            val  = self._rd(REG_LOOP_STAT)
            wait = (val >> 24) & 0x3
            if wait == target_state:
                return
            if self._rd(REG_STATUS) & 0x1:
                return
            if time.time() - t0 > timeout_s:
                stat = self._rd(REG_STATUS)
                raise TimeoutError(
                    f"FSM timeout waiting for state {target_state}. "
                    f"Got wait={wait} LOOP={hex(val)} STAT={hex(stat)}"
                )
            time.sleep(0.00005)

    # -----------------------------------------------
    # PS layers: Flatten → FC1 → ReLU → FC2 → Sigmoid
    # Input : [96, 6, 6] int16 Q6.9  from PL
    # Output: float32 probability 0..1
    # -----------------------------------------------

    def _run_ps_layers(self, pl_output):
        self._log("  PS: Flatten+FC1+ReLU+FC2+Sigmoid...")
        t0 = time.time()

        # -----------------------------------------------
        # Step 1: Q6.9 → float32
        # Divide by 512 to recover real-valued activations
        # -----------------------------------------------
        x = self._from_q69(pl_output)     # [96, 6, 6] float32

        # -----------------------------------------------
        # Step 2: Flatten
        # [96, 6, 6] → [3456]
        #
        # 96 channels × 6 × 6 = 3456 features
        # This is the flatten() layer between conv and FC
        # No computation, just reshape
        # -----------------------------------------------
        x = x.flatten()                   # [3456] float32

        # -----------------------------------------------
        # Step 3: FC1
        # [3456] → [128]
        #
        # Each of 128 output neurons computes:
        #   out_i = sum_j(x_j × W_ij) + b_i
        #         = dot product of input with weight row i
        #
        # fc1_w shape: [128, 3456]
        # x @ fc1_w.T : [3456] @ [3456,128] = [128]
        # -----------------------------------------------
        x = x @ self.fc1_w.T + self.fc1_b  # [128] float32

        # -----------------------------------------------
        # Step 4: ReLU after FC1
        # max(0, x) for each of 128 values
        # Same operation as hardware relu_unit
        # but done in software on ARM
        # -----------------------------------------------
        x = np.maximum(0.0, x)             # [128] float32

        # -----------------------------------------------
        # Step 5: FC2
        # [128] → [1]
        #
        # fc2_w shape: [1, 128]
        # x @ fc2_w.T : [128] @ [128,1] = [1]
        # Output is raw logit (unbounded real number)
        # Positive logit → dog
        # Negative logit → cat
        # -----------------------------------------------
        x = x @ self.fc2_w.T + self.fc2_b  # [1] float32

        # -----------------------------------------------
        # Step 6: Sigmoid
        # Converts logit to probability 0..1
        # sigmoid(z) = 1 / (1 + exp(-z))
        #
        # z >> 0  → prob ≈ 1.0  (dog, very confident)
        # z << 0  → prob ≈ 0.0  (cat, very confident)
        # z == 0  → prob = 0.5  (uncertain)
        # -----------------------------------------------
        prob = 1.0 / (1.0 + np.exp(-x[0]))  # scalar float32

        ms = (time.time() - t0) * 1000
        self._log(f"    logit={x[0]:.4f}  "
                  f"prob={prob:.4f}  ({ms:.1f}ms)")

        return float(prob)

    # -----------------------------------------------
    # Full inference pipeline
    # -----------------------------------------------

    def predict(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Not found: {img_path}")

        self._log(f"\n{'='*52}")
        self._log(f"Image: {os.path.basename(img_path)}")
        self._log(f"{'='*52}")

        t_start = time.time()

        # Step 1: Preprocess image on PS
        self._log("Preprocessing...")
        img = self._preprocess(img_path)   # [3,128,128] int16

        # Step 2: Run conv+relu layers on PL, pool layers on PS
        self._log("\nPL/PS layers:")
        x = img

        wgt_map = {
            'conv1_relu': 'conv1',
            'conv2_relu': 'conv2',
            'conv3_relu': 'conv3',
            'conv4_relu': 'conv4',
            'pool1': None, 'pool2': None,
            'pool3': None, 'pool4': None,
        }

        # Feature map shapes through the network:
        # img          : [3,  128, 128]
        # conv1_relu   : [16, 126, 126]
        # pool1        : [16,  63,  63]
        # conv2_relu   : [32,  61,  61]
        # pool2        : [32,  30,  30]
        # conv3_relu   : [64,  28,  28]
        # pool3        : [64,  14,  14]
        # conv4_relu   : [96,  12,  12]
        # pool4        : [96,   6,   6]  → goes to PS

        for (name, in_h, in_w, out_h, out_w,
             in_ch, out_ch, mode) in LAYER_CONFIG:
            if mode == MODE_PS_POOL:
                # Pool runs in PS: cheaper than fixing HW pool path,
                # correct for all channel counts, ~1ms per layer
                x = self._run_ps_pool(name, x, out_h, out_w)
            else:
                x = self._run_hw_layer(
                    name       = name,
                    fmap       = x,
                    in_h       = in_h,      in_w    = in_w,
                    out_h      = out_h,     out_w   = out_w,
                    num_in_ch  = in_ch,     num_out_ch = out_ch,
                    mode       = mode,
                    wgt_key    = wgt_map[name]
                )

        # Step 3: PS layers
        self._log("\nPS layers:")
        prob = self._run_ps_layers(x)   # x is [96,6,6] int16

        # Step 4: Decision
        label      = 'Dog' if prob > 0.5 else 'Cat'
        confidence = prob if prob > 0.5 else 1.0 - prob
        total_ms   = (time.time() - t_start) * 1000

        self._log(f"\n{'='*52}")
        self._log(f"Prediction : {label}")
        self._log(f"Confidence : {confidence*100:.1f}%")
        self._log(f"Score      : {prob:.6f}")
        self._log(f"Total time : {total_ms:.0f}ms")
        self._log(f"{'='*52}\n")

        return prob, label

    # -----------------------------------------------
    # Batch inference
    # -----------------------------------------------

    def predict_batch(self, img_paths):
        results = []
        n = len(img_paths)
        for i, path in enumerate(img_paths):
            print(f"\n[{i+1}/{n}] {os.path.basename(path)}")
            try:
                prob, label = self.predict(path)
                conf = prob if prob > 0.5 else 1.0 - prob
                results.append({
                    'path': path, 'label': label,
                    'prob': prob, 'conf': conf
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'path': path, 'label': 'ERROR',
                    'prob': -1,   'conf': 0
                })

        self._print_summary(results)
        return results

    def _print_summary(self, results):
        print(f"\n{'='*55}")
        print(f"{'File':<25} {'Label':<5} "
              f"{'Score':>7} {'Conf':>7}")
        print(f"{'-'*55}")
        for r in results:
            name = os.path.basename(r['path'])[:23]
            print(f"{name:<25} {r['label']:<5} "
                  f"{r['prob']:>7.4f} "
                  f"{r['conf']*100:>6.1f}%")
        dogs = sum(1 for r in results if r['label'] == 'Dog')
        cats = sum(1 for r in results if r['label'] == 'Cat')
        print(f"{'='*55}")
        print(f"Dogs: {dogs}  Cats: {cats}  Total: {len(results)}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import sys

    cnn = CNNInference(
        bitstream_path = '/home/xilinx/design_1.bit',
        weights_path   = '/home/xilinx/weights_for_fpga.npz',
        verbose        = True
    )

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        if len(paths) == 1:
            cnn.predict(paths[0])
        else:
            cnn.predict_batch(paths)
    else:
        test = '/home/xilinx/test.jpg'
        if os.path.exists(test):
            cnn.predict(test)
        else:
            print("Usage: python3 pynq_inference.py image.jpg")
            print("       python3 pynq_inference.py *.jpg")
