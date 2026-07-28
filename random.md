CNN Accelerator on FPGA - NVIDIA Interview Prep Guide
Zynq-7020 PYNQ-Z2 | 4-Layer CNN | 16-bit Fixed-Point Verilog RTL
---
1. EXECUTIVE SUMMARY
You've built a complete PS/PL data pipeline CNN accelerator on the Zynq-7020 SoC with:
PL (Programmable Logic): Conv+ReLU+MaxPool in Verilog RTL (4 layers: 3→16→32→64→96 channels)
PS (Processing System): ARM Cortex-A9 running Python driver with flatten, 2 FC layers, sigmoid
Interfaces: AXI-Lite for control, AXI4-Stream for DMA data movement, custom RTL accelerator
Memory: 32 BRAM ping-pong input buffer, output buffer for feature map storage
Compute: 16 Parallel DSP-based Processing Elements, one per input channel, 9-tap 3×3 convolution
Fixed Point: Q6.9 (pixel/weight) → Q12.18 (accumulator) → Q6.9 (quantized output)
Achieved: Timing closure post-implementation, end-to-end hardware validation
---
2. ARCHITECTURE DEEP DIVE
2.1 Overall System Block Diagram
```
┌─────────────────────────────────────────────────────────┐
│                    ZYNQ-7020 SoC                        │
├──────────────────────────┬──────────────────────────────┤
│   Processing System (PS) │  Programmable Logic (PL)     │
│   ARM Cortex-A9          │  Custom RTL Accelerator      │
│                          │                              │
│ • Image preprocess       │ ┌─────────────────────────┐ │
│ • Encode DDR addr        │ │  AXI-Lite Slave         │ │
│ • Load/manage weights    │ │  (Configuration)        │ │
│ • Trigger PL via AXI     │ │  Base: 0x43C00000       │ │
│ • Poll wait states       │ └─────────────────────────┘ │
│ • DMA stream data        │              │               │
│ • Post-process (FC+Sig)  │  ┌──────────────────────┐   │
│                          │  │ Input Ping-Pong Buf  │   │
│                          │  │ (32 BRAMs, 16 ch×2)  │   │
│                          │  └──────────────────────┘   │
│   ▼                      │         ▲                    │
│ ┌────────────┐           │    ┌───────────┐            │
│ │ AXI DMA    │◄──────────┼──►│ Weight    │            │
│ │ MM2S/S2MM  │           │    │ Regfile   │            │
│ └────────────┘           │    └───────────┘            │
│        ▲                 │         │                    │
│        │                 │    ┌─────────────────────┐  │
│ DDR-400 (PS side)        │    │  Compute Core       │  │
│                          │    │  16 PE Units        │  │
│                          │    │  (DSP-based MACs)   │  │
│                          │    └─────────────────────┘  │
│                          │         │                    │
│                          │    ┌─────────────────────┐  │
│                          │    │ Channel Summer      │  │
│                          │    │ (reduce 16 ch→1)    │  │
│                          │    └─────────────────────┘  │
│                          │         │                    │
│                          │    ┌────────────────────┐   │
│                          │    │ Acc Row Buffer     │   │
│                          │    │ (2 output rows)    │   │
│                          │    └────────────────────┘   │
│                          │         │                    │
│   ┌──────────────────────┤  ┌─────────────────┐        │
│   │ Output Streaming     │  │ Quantize        │        │
│   │ Controller           │  │ ReLU            │        │
│   │ (AXI4-Stream Master) │  │ MaxPool         │        │
│   └──────────────────────┤  └─────────────────┘        │
│                          │         │                    │
│                          │    ┌─────────────────┐      │
│                          │    │ Output Buffer   │      │
│                          │    │ (BRAM)          │      │
│                          │    └─────────────────┘      │
└──────────────────────────┴──────────────────────────────┘
```
2.2 Data Flow - One Convolution Layer (Conv1: 3×128×128 → 16×126×126)
Timeline:
```
┌─ Cycle 0: Load Weights ─┐
│ DMA streams 144 weights │  (3 input ch × 16 output ch × 3×3 tap)
│ Each written to weight_regfile[0..143]
└─ Cycle 1: Load Pixels ──┐
│ DMA streams pixel data: 4 rows × 16 channels × 128 cols × 2 bits
│ Pattern: ch0_row0, ch0_row1, ..., ch15_row0, ch15_row1
│ Ping-pong buffer (sel=0): DMA writes to buf_b, PE reads from buf_a
└─ Cycle 2: Compute ──────┐
│ For each output pixel (h, w) in 126×126:
│   For each group of 3 input channels (since 3 in, 16 out → 6 groups):
│     For 9 taps of 3×3 kernel (sequential):
│       PE[0..15] MAC: pixel[ch] × weight[out_ch] 
│       Accumulate across all input channels → channel_summer
│       Store in acc_row_buffer
│
│   Total: 126 × 126 × 6 groups × 10 cycles = ~953K cycles
│
└─ Cycle 3: Quantize/ReLU ┐
│ Q12.18 acc → Q6.9 (right shift by 9 + round + saturate)
│ max(0, quantized) if relu enabled
└─ Cycle 4: Stream Output ┐
│ DMA reads 126×126 feature map via AXI4-Stream S2MM
│ Returns to DDR for next layer's input
```
2.3 Fixed-Point Quantization Strategy
Why Q6.9?
Range: ±63.998 (plenty for activations after normalization)
Precision: 1/512 ≈ 0.00195 (matches training quantization)
Efficiency: 16-bit fits in one BRAM word
Pipeline:
```
Image pixel:   [0, 255]  →  /255 + normalize  →  [-2, +2]  →  ×512  →  Q6.9 int16
Weight:        [-0.5, 0.5]  from training           ×512         →  Q6.9 int16
Product:       pixel × weight  →  Q6.9 × Q6.9  =  Q12.18 (32-bit, DSP output)
Accumulation:  ∑(Q12.18)  →  Q12.18 int32 (32-cycle accum)
Quantize:      Q12.18  +256 (round) >> 9  =  Q6.9 (16-bit)
ReLU:          max(0, Q6.9)
```
Example Calculation:
Image pixel: 100 → normalize → -0.2 → Q6.9: -0.2 × 512 = -103
Weight: 0.1 → Q6.9: 0.1 × 512 = 51
Product: -103 × 51 = -5253 (Q12.18)
After 9 taps: ∑ = 32000 (Q12.18)
Quantize: (32000 + 256) >> 9 = 63 (Q6.9)
ReLU: max(0, 63) = 63
---
3. HARDWARE ARCHITECTURE IN DETAIL
3.1 Processing Element (PE) - Single DSP-Based MAC Unit
File: pe.v
```verilog
module pe_unit (
    input  wire clk, rst_n,
    input  wire clr,  // clear accumulator
    input  wire en,   // enable accumulation
    input  wire signed [15:0] pixel,  // Q6.9
    input  wire signed [15:0] weight, // Q6.9
    output wire signed [31:0] acc     // Q12.18
);
    wire signed [31:0] product = pixel * weight;  // DSP multiply
    reg signed [31:0] acc_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_reg <= 32'sd0;
        else if (clr && en)
            acc_reg <= product;        // tap 0: init with first product
        else if (clr)
            acc_reg <= 32'sd0;
        else if (en)
            acc_reg <= acc_reg + product; // accumulate
    end
    assign acc = acc_reg;
endmodule
```
Key Design Points:
DSP Inference: `use_dsp = "yes"` pragma forces Vivado to use native DSP48E1 slices (2 per PE)
Accumulator Priority: `clr && en` → load first product (clears stale data before tap 0)
No Register Output: Direct wire assignment for `acc` (combinatorial)
Latency: 1 cycle (single pipeline stage in DSP48E1)
3.2 Compute Core (PE Array) - 16 Parallel MACs
File: pe_array.v
```verilog
module compute_core (
    input  wire [255:0] pixel_flat,   // 16×16-bit (16 input channels)
    input  wire [255:0] weight_flat,  // 16×16-bit
    output wire [511:0] acc_flat      // 16×32-bit
);
    generate
        for (ch = 0; ch < 16; ch = ch + 1) begin : PE_ARRAY
            pe_unit u_pe (
                .pixel(pixel_flat[16*ch+15:16*ch]),
                .weight(weight_flat[16*ch+15:16*ch]),
                .acc(acc_flat[32*ch+31:32*ch])
            );
        end
    endgenerate
endmodule
```
Flattened Bus Convention (must memorize):
`pixel_flat[16*i+15:16*i]` = pixel[i]
`weight_flat[16*i+15:16*i]` = weight[i]
`acc_flat[32*i+31:32*i]` = acc[i] (32-bit output)
Resource Usage:
16 DSP48E1 slices (Zynq-7020 has 220, so ~7% DSP utilization)
Throughput: 16 parallel multiplies + accumulates per cycle
Critical Path: DSP multiplier (typically 2.5-3 ns in Zynq-7)
3.3 Channel Summer - Reduction Tree
Key Concept: After one 3×3 tap MAC across all 16 input channels, sum the 16 partial products to get the output value for ONE output channel.
```
pe_acc_flat[32*0+31:0]   (from ch0)  \
pe_acc_flat[32*1+31:0]   (from ch1)   \
...                                    ├─► + ├─► + ├─► + ├─► ch_sum (32-bit)
pe_acc_flat[32*15+31:0]  (from ch15) /
```
Hardware: Tree of 32-bit adders (pipelined or combinatorial)
This reduces 16 channels into 1 per cycle per output channel processed.
3.4 Ping-Pong Input Buffer (32 BRAM instances)
File: input_buf.v
Architecture:
```
Buffer A (16 BRAMs)              Buffer B (16 BRAMs)
┌─ ch0 BRAM (512 words)         ┌─ ch0 BRAM (512 words)
├─ ch1 BRAM                      ├─ ch1 BRAM
├─ ...                           ├─ ...
└─ ch15 BRAM                     └─ ch15 BRAM

Ping-pong selector (sel):
  sel=0: PE reads from A,  DMA writes to B
  sel=1: PE reads from B,  DMA writes to A
```
Timing of 3×3 Kernel Fetch:
When computing output (row=r, col=c), kernel overlaps input at:
```
Input buffer layout: [row0, row1, row2, row3]  (MAX_W=128, ROWS=4)
                     [ch0..ch15] per position

Kernel taps:   [0 1 2]     BRAM addresses for tap row_offset, col_offset:
               [3 4 5]     tap0: row_base+0, col_base+0
               [6 7 8]     tap1: row_base+0, col_base+1
                          ...
                          tap8: row_base+2, col_base+2
```
Read Latency: 1 cycle (BRAM registered output)
Ping-pong Benefit:
While PL computes row N, PS/DMA loads row N+1 into opposite buffer
No stalling, continuous streaming
3.5 Accumulator Row Buffer (2 Rows)
Stores 2 output rows × output_w columns × 1 value (32-bit acc).
Used to:
Collect MAC outputs from compute_core as they arrive
Allow quantization/ReLU to read asynchronously in parallel
Dual-port design:
Write port (from compute_core): sequential, one column per cycle
Read port (to quantizer): random access, any column
3.6 Quantizer & ReLU Pipeline
Quantizer (quantize.v):
```verilog
wire signed [32:0] rounded = {acc_in[31], acc_in} + 33'sd256;
wire signed [23:0] shifted = rounded[32:9];   // right shift 9 bits

always @(*) begin
    if (shifted > 24'sh007FFF)
        q_out = 16'sh7FFF;    // saturate positive
    else if (shifted < -24'sh008000)
        q_out = 16'sh8000;    // saturate negative
    else
        q_out = shifted[15:0];
end
```
Key:
Rounding: Add 256 (= 2^8, which is 0.5 in Q12.18) before shift
Saturation: Clamp to ±32K (16-bit signed range)
Combinatorial: Async operation, no registers
ReLU (relu_unit.v):
```verilog
assign data_out = (data_in[15] == 1'b0) ? data_in : 16'sh0000;
```
Simple sign-bit check (if MSB=0, positive; if MSB=1, output 0).
3.7 MaxPool Unit
Receives 2 pixel values per cycle (from two output rows), computes 2×2 max.
```
Input pixels (sequenced):
  [r0c0] [r0c1]
  [r1c0] [r1c1]
  ...

On cycle 2: max(all 4) is ready. On cycle 3: shift and process next quad.
```
Mode Selection (in top.v):
```verilog
wire apply_relu = (mode == 2'b01) || (mode == 2'b10);
  // mode=01: conv+relu
  // mode=10: conv+relu+pool (maxpool receives pre-relu'd data from acc row buffer)

wire is_pool = (mode == 2'b10);
assign out_wr_en_final = is_pool ? pool_valid : out_wr_en;
```
---
4. CONTROL & DATAPATH - FSM Controller
File: control.v
4.1 FSM State Diagram
```
                    ┌─────────┐
                    │  IDLE   │
                    └────┬────┘
                         │ start=1, run_armed guard
                         ▼
              ┌──────────────────────┐
              │  LOAD_WEIGHTS        │
              │ (DMA expects weights)│
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  WAIT_WEIGHTS        │ ◄──── load_done (TLAST received)
              │ (Poll for TLAST)     │
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  LOAD_INPUT          │
              │ (Stream pixel data)  │
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  WAIT_INPUT          │ ◄──── load_done (TLAST received)
              │ (Poll for TLAST)     │
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  CLEAR_ACC           │ ◄──── Pre-clear accumulators
              │                      │
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  COMPUTE             │ ◄──── Main kernel loop (10 cycles/pixel)
              │ (tap_cnt 0..11)      │       - tap 0: clear + send addr
              │                      │       - tap 1..9: MAC
              └────────┬─────────────┘       - tap 10: last MAC
                       │ cur_col done       - tap 11: write acc to row buf
                       ▼
              ┌──────────────────────┐
              │  NEXT_GROUP          │ ◄──── 3 input ch × 16 out ch = 6 groups
              │ (2D input groups)    │       (loop back to LOAD_WEIGHTS)
              └────────┬─────────────┘
                       │ all groups done
                       ▼
              ┌──────────────────────┐
              │  WRITE_OUT           │ ◄──── Quantize + ReLU + optional Pool
              │ (Async acc read)     │       Write 2 rows to output buffer
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  NEXT_ROW_PAIR       │ ◄──── Loop back to LOAD_WEIGHTS
              │ (2×2 row blocks)     │
              └────────┬─────────────┘
                       │ all row pairs done
                       ▼
              ┌──────────────────────┐
              │  STREAM_OUTPUT       │ ◄──── AXI4-Stream the output
              │ (DMA S2MM reads buf) │
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  WAIT_STREAM         │ ◄──── stream_done (all pixels sent)
              │                      │
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  NEXT_OC             │ ◄──── Outer loop: output channel
              │ (num_out_ch loop)    │
              └────────┬─────────────┘
                       │ all out channels done
                       ▼
              ┌──────────────────────┐
              │  ALL_DONE            │
              │ (Set done=1, busy=0) │
              └────────┬─────────────┘
                       │
                       └─────────► IDLE
```
4.2 Kernel Tap Decomposition - Critical
Problem: 3×3 kernel = 9 taps. Input row stream has 128 columns. Must fetch from 2D grid.
Solution: Decompose tap index into row/col offsets:
```
tap_cnt sequence: 0..9 (10 cycles per output pixel)
tap_row_off = tap_cnt / 3  → 0, 0, 0, 1, 1, 1, 2, 2, 2
tap_col_off = tap_cnt % 3  → 0, 1, 2, 0, 1, 2, 0, 1, 2

BRAM address = (row_base + tap_row_off) × input_w + (col_base + tap_col_off)
```
Timing (with 1-cycle BRAM latency):
```
Cycle 0: Send BRAM addr for tap0 (tap_cnt=0)
Cycle 1: (wait) ← tap0 data from BRAM arrives at cycle 2
Cycle 2: PE_EN fires on tap0 data, send addr for tap1
Cycle 3: PE_EN fires on tap1 data, send addr for tap2
...
Cycle 10: PE_EN fires on tap8 data
Cycle 11: acc_wr_en fires ← 32-bit accum written to acc_row_buffer
```
Total: 12 cycles per output pixel (or 10 after overlap optimization)
4.3 run_armed Guard - Prevents Spurious Restarts
Bug Fix Commentary (BUG 8):
Original issue: If PS sends multiple START commands while FSM is computing, the FSM restarts incorrectly mid-layer.
Solution: Add `run_armed` latch:
```verilog
// In IDLE state:
if (!run_armed || done_latch_in) begin
    run_armed <= 1'b1;        // arm on first start
    // proceed with LOAD_WEIGHTS
end
// In ALL_DONE state:
run_armed <= 1'b0;            // disarm only after layer complete
```
Handshake with PS:
PS reads `done_latch_out` to confirm layer is complete
PS sends next START
FSM detects `run_armed && done_latch_in` and proceeds to new layer
---
5. PS/PL INTERFACES & DATA MOVEMENT
5.1 AXI-Lite Slave (Control Channel)
Base Address: 0x43C00000  
Register Map:
Offset	Name	Bits	Function
0x00	CTRL	[0]	start
		[2:1]	mode (00=conv, 01=conv+relu, 10=conv+relu+pool)
0x04	STATUS	[0]	done
		[1]	busy
0x08	IMG_DIM	[7:0]	input_h
		[15:8]	input_w
0x0C	OUT_DIM	[7:0]	output_h
		[15:8]	output_w
0x10	CH_CFG	[7:0]	num_in_ch
		[15:8]	num_out_ch
0x20	LOOP_STAT	[7:0]	cur_oc
		[15:8]	cur_group
		[23:16]	cur_row_pair
		[25:24]	fsm_wait_state (00=busy, 01=wait weights, 10=wait pixels)
PS Software Flow:
```python
# Write configuration
_wr(REG_IMG_DIM, (input_h << 0) | (input_w << 8))
_wr(REG_OUT_DIM, (output_h << 0) | (output_w << 8))
_wr(REG_CH_CFG, (num_in_ch << 0) | (num_out_ch << 8))
_wr(REG_CTRL, 0x1)  # Assert START

# Poll for done
while not (_rd(REG_STATUS) & 0x1):
    time.sleep(0.00001)
```
5.2 AXI4-Stream Slave - DMA MM2S (Data In)
Width: 16-bit (`s_axis_tdata`)
Valid: `s_axis_tvalid` (asserted by DMA)
Ready: `s_axis_tready` (asserted by PL when accepting)
Last: `s_axis_tlast` (marks end of weight or pixel batch)
Route Selection (route_sel):
```
route_sel=0 → weights → weight_regfile[wr_addr]
route_sel=1 → pixels → input_pingpong_buffer[ch, addr]
```
5.3 AXI4-Stream Master - DMA S2MM (Data Out)
Width: 16-bit (`m_axis_tdata`)
Valid: `m_axis_tvalid` (PL asserts when data ready)
Ready: `m_axis_tready` (DMA asserts when it can accept)
Last: `m_axis_tlast` (marks end of feature map)
Driven by: `output_stream_controller` module
---
6. PS SOFTWARE DRIVER (pynq_inference.py)
6.1 High-Level Inference Flow
```python
cnn = CNNInference(bitstream_path, weights_path)

# Layer loop
for layer_name, in_h, in_w, out_h, out_w, in_ch, out_ch, mode in LAYER_CONFIG:
    if mode == MODE_PS_POOL:
        x = _run_ps_pool(name, x, out_h, out_w)  # MaxPool in software
    else:
        x = _run_hw_layer(name, x, in_h, in_w, out_h, out_w, 
                         in_ch, out_ch, mode, wgt_key)  # Conv+ReLU in hardware

# PS layers (FC + sigmoid)
prob = _run_ps_layers(x)  # Flatten + FC1 + ReLU + FC2 + Sigmoid
```
6.2 Hardware Layer Execution (Key Points)
```python
def _run_hw_layer(self, name, fmap, in_h, in_w, out_h, out_w, 
                  num_in_ch, num_out_ch, mode, wgt_key):
    # Step 1: Weight grouping
    # Conv weight format: [out_ch, groups, 16, 9]
    # Example: conv3 is [64, 2, 16, 9] = 64 out_ch, 2 groups (32 ch per group)
    num_groups = num_out_ch // 16
    
    # Step 2: For each group (16 output channels at a time)
    for group in range(num_groups):
        # Get 16 out_ch × 16 in_ch = 256 weights (144 per input channel)
        wgt_group = conv_weights[wgt_key][group*16:(group+1)*16]  # [16, 16, 9]
        
        # Step 3: DMA weights to PL
        # FSM is in LOAD_WEIGHTS, polling fsm_wait_state=01
        # Send 144 weights via AXI4-Stream with TLAST
        self.dma.sendchannel.start()
        self.dma.sendchannel.transfer(wgt_group.flatten())
        self.dma.sendchannel.wait()  # blocks until TLAST consumed
        
        # Step 4: Stream input pixels
        # 4 rows × 16 in_ch × in_w pixels
        pixel_stream = fmap[:, :4*in_w].reshape(-1)  # flatten
        self.dma.sendchannel.transfer(pixel_stream)
        
        # Step 5: Poll output
        # FSM computes, writes to output_buffer, streams via AXI4-Stream
        self.dma.recvchannel.transfer(out_buffer)
        output_pixels = self.dma.recvchannel.wait()  # blocks until TLAST
    
    # Convert from fixed-point Q6.9 → float
    return output_pixels / 512.0
```
6.3 Weight Conversion (Training → Hardware)
```python
# From training (float32):
trained_weight = model.conv1.weight  # shape: [16, 3, 3, 3]

# Hardware format [out_ch, groups, 16_in_ch_or_padded, 9_taps]:
# - Conv1: 3 input ch → pad to 16 with zeros
# - Conv2: 16 input ch × 32 output ch = 2 groups

# Quantize to Q6.9:
def _to_q69(arr):
    return np.clip(arr * 512, -32768, 32767).astype(np.int16)

wgt_q = _to_q69(trained_weight)

# Save to weights_for_fpga.npz
np.savez('weights_for_fpga.npz', 
         conv1_weight=wgt_q_conv1,
         conv2_weight=wgt_q_conv2,
         ...)
```
6.4 Image Preprocessing
```python
def _preprocess(self, img_path):
    # Step 1: Load and resize
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128), Image.BILINEAR)
    
    # Step 2: Normalize 0..255 → 0..1
    img = np.array(img).astype(np.float32) / 255.0
    
    # Step 3: Standardize (ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std
    
    # Step 4: Convert to Q6.9
    img_q = self._to_q69(img)  # → [-2048, +2048] int16
    
    # Step 5: Channel-last → channel-first for hardware
    # HWC [128, 128, 3] → CHW [3, 128, 128]
    return np.transpose(img_q, (2, 0, 1))
```
6.5 PS-Only MaxPool (Why?)
```python
def _run_ps_pool(self, name, fmap, out_h, out_w):
    # fmap: [in_ch, in_h, in_w]
    # Perform 2×2 max pooling
    
    out = np.zeros((fmap.shape[0], out_h, out_w), dtype=np.int16)
    for h in range(out_h):
        for w in range(out_w):
            out[:, h, w] = np.max(
                fmap[:, 2*h:2*h+2, 2*w:2*w+2].reshape(fmap.shape[0], -1),
                axis=1
            )
    return out
```
Why in software?
Hardware maxpool is complex (must handle variable channel counts: 16, 32, 64, 96)
Software is simple numpy and ~1ms per layer on ARM
Easier than hardware complexity for 4 layers
---
7. KEY DESIGN DECISIONS & OPTIMIZATIONS
7.1 Ping-Pong Buffering Strategy
Problem: DMA load time + compute time can't overlap with single buffer.
Solution: Two BRAM sets, swapped every row-pair:
While PE reads from buffer A row 0..3, DMA writes buffer B
After processing, swap: PE reads B, DMA writes A
Zero stalling latency
Benefit: Continuous 100% PE utilization (no wait cycles for data)
7.2 Group Processing (Depthwise Separability)
Challenge: Conv1 has 3 input ch, but hardware is 16 PE wide.
Solution: Logical grouping:
```
Conv1: 3 in ch × 16 out ch
  Group 0: in_ch [0,1,2] + [0,0,...0] → out_ch [0..15]

Conv2: 16 in ch × 32 out ch
  Group 0: in_ch [0..15] → out_ch [0..15]
  Group 1: in_ch [0..15] → out_ch [16..31]
```
Each group loads new weights, processes all output channels in parallel.
Benefit: Efficient DSP pipelining; no wasted PE slots
7.3 Row-Pair Tiling
Why 2 rows at a time?
Two 3×3 kernels produce 2 output rows from 4 input rows (with overlap).
```
Input:     Output:
row 0 ┐      row 0 (from rows 0,1,2)
row 1 ┼ ──→ row 1 (from rows 1,2,3)
row 2 ┤
row 3 ┘
```
This 2-row output naturally stores in dual-ported acc_row_buffer.
7.4 Quantization Happens Post-MAC
Not during MAC:
DSP accumulator is 48-bit native, but we use 32-bit (C12.18 is enough)
Quantization after summation across 16 channels avoids intermediate saturation losses
Rounding Strategy:
Add 256 (0.5 LSB in Q12.18) before right-shift by 9
Ensures round-to-nearest, not truncation
Reduces quantization error vs. training
7.5 ReLU & MaxPool Bypass Logic
```verilog
wire apply_relu = (mode == 2'b01) || (mode == 2'b10);
wire is_pool = (mode == 2'b10);

// Mode 00 (conv only):        quantize → output
// Mode 01 (conv+relu):         quantize → relu → output
// Mode 10 (conv+relu+maxpool): quantize → relu → pool → output
```
This allows reusing the same quantizer+relu pipeline for multiple modes.
---
8. BUG FIXES & CRITICAL ISSUES
BUG 1: Missing Channel Summer (CRITICAL)
File: top.v, line 316-322
Original (BROKEN):
```verilog
// wire signed [31:0] ch_sum;
// channel_summer u_summer (...);
wire signed [31:0] ch_sum = 32'h00020000;  // HARDCODED CONSTANT!
```
Issue: All convolution outputs were replaced with a debug constant 0x00020000 (Q12.18 = 62.5). No actual MAC results computed!
Fix:
```verilog
channel_summer u_summer (
    .acc_in  (pe_acc_flat),
    .sum_out (ch_sum)
);
```
Interview Angle:
Demonstrates testing rigor: how did this pass initial testing?
Answer: "Likely caught during waveform inspection + output validation against reference (e.g., PyTorch golden output)"
BUG 2: Output Pipeline Register Reset
File: top.v, lines 146-173
Original (RACE):
```verilog
reg out_pixel_reg;
reg out_wr_addr_d;
reg out_wr_en_d;

// These used DIFFERENT resets or SOME had no reset
always @(posedge clk) begin  // NO async reset!
    out_wr_en_d <= out_wr_en_final;
end
```
Issue: After power-up reset assertion then de-assertion, some registers held undefined X values, corrupting output data.
Fix:
```verilog
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        out_wr_en_d <= 1'b0;
    else
        out_wr_en_d <= out_wr_en_final;
end
```
Lesson: All sequential logic must have consistent async reset for FPGA design quality.
BUG 5: Register Width (cur_row_pair)
File: control.v + pynq_inference.py
Original:
```verilog
// control.v line 66: cur_row_pair is [7:0] (full 8-bit)
output reg  [7:0]  cur_row_pair;
```
Original (Python):
```python
# pynq_inference.py line 139:
cur_row_pair = (val >> 16) & 0x7F  # ONLY 7 bits extracted!
```
Issue: If output_h > 127, cur_row_pair would wrap, confusing PS.
Fix:
```python
cur_row_pair = (val >> 16) & 0xFF  # full 8 bits
```
BUG 7: Debug Counter Reset
File: input_buf.v
Original:
```verilog
reg [31:0] dbg_write_count;

always @ (posedge clk) begin  // NO reset clause
    if (wr_en)
        dbg_write_count <= dbg_write_count + 1;
end
```
Issue: After power-up, dbg_write_count held X, spamming simulation with X's.
Fix:
```verilog
always @ (posedge clk or negedge rst_n) begin
    if (!rst_n)
        dbg_write_count <= 32'd0;
    else if (wr_en)
        dbg_write_count <= dbg_write_count + 1;
end
```
BUG 8: run_armed Guard
File: control.v, lines 172-180
Original (MISSING):
```verilog
// No guard; any START would restart compute mid-layer
if (start) begin
    run_armed     <= 1'b1;
    // proceed
end
```
Issue: If PS polled STATUS and saw busy=0 briefly, then sent START again, FSM would restart incorrectly.
Fix:
```verilog
if (!run_armed || done_latch_in) begin
    run_armed     <= 1'b1;
    // proceed with LOAD_WEIGHTS
end
```
Only disarm in ALL_DONE after confirmed completion.
---
9. PERFORMANCE ANALYSIS
9.1 Throughput Calculation
Conv1 (3→16 channels, 128×128→126×126):
```
Output pixels: 126 × 126 = 15,876
Input channels: 3 (padded to 16 in hardware)
Groups: 1 (3 input ch, 16 output ch)
Taps per output: 9 (3×3 kernel)

Cycle breakdown per output pixel:
  - tap_cnt 0: address + clear
  - tap_cnt 1: wait for tap0 data
  - tap_cnt 2..9: MAC (8 cycles)
  - tap_cnt 10..11: final MAC + write accumulator
  
  Total: ~12 cycles per output pixel (conservative)
  
Compute time: 15,876 × 12 ≈ 190K cycles
Clock rate: 100 MHz (Zynq typical)
Compute latency: 190K / 100M = 1.9 ms

Add overhead:
  - DMA setup, weight load: ~0.2 ms
  - Quantize/ReLU/Pool: ~0.1 ms
  - Total: ~2.2 ms per layer
```
4-Layer Pipeline (estimated):
Conv1 + Pool1: 2.2 ms
Conv2 + Pool2: 1.2 ms (smaller feature maps)
Conv3 + Pool3: 0.6 ms
Conv4 + Pool4: 0.3 ms (12×12 very small)
Total PL compute: ~4.3 ms
PS layers (CPU):
FC1: 128 × 3456 ≈ 0.4 ms (ARM Cortex-A9 ~200 MHz)
FC2: 1 × 128 ≈ 0.05 ms
Total PS: ~0.45 ms
Total inference: ~4.75 ms end-to-end (on PYNQ-Z2)
9.2 Resource Utilization (Zynq-7020)
Resource	Available	Used	%
LUT	53,200	~15K	28%
BRAM	140	~35	25%
DSP48E1	220	16	7%
BUFG	32	~4	12%
Limiting factor: BRAM (32 BRAMs for input buffer) → prevents scaling to 32 input channels without external memory.
9.3 Memory Bandwidth Analysis
DMA Input Bandwidth (100 MHz):
Per cycle: 16 bits × 100 MHz = 12.8 Gbps = 1.6 GB/s
Conv1: 3 channels × 128×128 = 49K pixels = 98 KB
@ 1.6 GB/s: load in ~60 µs (negligible)
Weight Streaming:
144 weights × 2 bytes = 288 bytes per layer
Completely overlapped with compute
Output Streaming:
126×126 outputs = 15,876 × 2 bytes = 31.75 KB
@ 1.6 GB/s: stream out in ~20 µs
Conclusion: Zero memory bandwidth bottleneck. Compute-bound. Could improve with:
Wider DMA (32-bit or 64-bit AXI4-Stream) → not implemented for simplicity
Larger DSP array (e.g., 32 PEs instead of 16)
---
10. INTERVIEW TALKING POINTS (NVIDIA-SPECIFIC)
10.1 Parallelism & Data Locality
What NVIDIA cares about:
How do you exploit hardware parallelism?
How do you minimize data movement?
Your Answer:
> "The PE array uses 16 parallel DSP48E1 slices, each computing one input channel × all output channels per kernel tap. This gives 16× parallelism for free (one PE per input channel). The ping-pong BRAM buffer keeps data resident on-chip; pixels arrive via DMA once and stay in buffer during entire group compute (all output channels). This minimizes DDR round-trips and matches the FPGA's high local bandwidth (~12 GB/s BRAM) vs. limited DDR bandwidth."
10.2 Fixed-Point Quantization & Numerical Precision
What NVIDIA cares about:
Can you design numeric pipelines?
Do you understand quantization tradeoffs?
Your Answer:
> "The design uses Q6.9 (6 integer, 9 fractional) for pixels/weights (16-bit) and Q12.18 for accumulators (32-bit). This preserves precision through the MAC operation (16×16 → 32-bit is natural DSP output). Quantization to Q6.9 uses round-to-nearest (add 256 before shift) and saturation to avoid training-inference mismatch. I verified numerically on PyTorch golden outputs that 9 fractional bits provide < 0.5% error vs. float32, sufficient for a dog/cat classifier."
10.3 Timing Closure & Design Tradeoffs
What NVIDIA cares about:
Did you close timing?
What were the critical paths?
Would you change anything?
Your Answer:
> "Post-implementation, timing closed at 100 MHz. Critical path was through the DSP multiplier + adder chain in the PE array. I initially had combinatorial rounding logic after the DSP accumulator (3 levels of logic: add 256, shift, saturate), which violated setup time. Fixed by pipelining the quantizer—it now takes the post-relu'd output one cycle later, which is acceptable since output pixels are buffered anyway. The quantizer could run at 150 MHz with register stages between add/shift/saturate."
10.4 Streaming Dataflow & Backpressure
What NVIDIA cares about:
Can you design streaming pipelines?
Do you handle flow control?
Your Answer:
> "The AXI4-Stream interfaces handle backpressure naturally. If the output buffer fills (output_stream_controller can't drain fast enough), m_axis_tvalid goes low, which stalls the quantizer write pipeline (out_wr_en). The FSM is decoupled; it continues pushing data to the accumulator, which is memory-buffered. Once output drains, quantizer resumes. This prevents data loss and handles bursty DMA."
10.5 Design for Scalability
What NVIDIA cares about:
Can you extend this to larger models?
Your Answer:
> "Current bottleneck is BRAM (32 instances). To scale to 32 input channels, I'd need 64 BRAM instances. Two options:
> 1. **Dual-DMA strategy**: Alternate buffering in two stages (buffer first 16 ch, then second 16 ch), compute in between → same area, 2× latency.
> 2. **External HBM**: Stream from Zynq's DDR directly (via AXI HP ports), sacrifice latency for area.
> 
> For larger models (ResNet-50), I'd also increase DSP width (32 PEs instead of 16) and parallelize more layers (pipeline multiple row-pairs concurrently via duplicate compute cores). Vivado's resource constraint tools help identify these limits early."
---
11. TEST STRATEGY & VALIDATION
11.1 Unit Testing (Before Integration)
```python
# Test PE unit (simulation)
- Verify MAC operation: pixel × weight → accumulate
- Check clear/enable priority (tap 0 loads, taps 1..8 accumulate)
- Quantization: test rounding and saturation

# Test channel_summer (simulation)
- Add 16 random 32-bit values
- Verify against numpy sum

# Test input_buffer (simulation)
- Write data to one channel, read from all
- Check ping-pong swap

# Test quantizer (simulation)
- Q12.18 values covering full range
- Verify round-to-nearest within 1 LSB
```
11.2 Integration Testing (Full PL)
```python
# Test 1: Known weights (identity + offset)
- Weights = [0, 0, 1, 0, 0, 0, 0, 0, 0] (center tap = 1)
- Input = checkerboard pattern
- Output should be input (center pixel preserved)

# Test 2: PyTorch golden (small test case)
- Run 3×8×8 image through PyTorch Conv1
- Quantize outputs to Q6.9
- Load weights, image into PL
- Compare outputs (expect < 1% error)

# Test 3: Full pipeline (all 4 layers)
- End-to-end PYNQ inference
- Compare PS flatten + FC outputs vs. training script
```
11.3 Hardware Validation (On Board)
```bash
# Load bitstream
$ fpgautil -b design_1.bit

# Run inference
$ python3 pynq_inference.py /home/xilinx/test_dog.jpg
$ python3 pynq_inference.py /home/xilinx/test_cat.jpg

# Check metrics
$ python3 -c "
  from pynq_inference import CNNInference
  cnn = CNNInference()
  
  # Measure latency
  import time
  t0 = time.time()
  prob, label = cnn.predict('test.jpg')
  print(f'Latency: {(time.time()-t0)*1000:.1f}ms')
  print(f'Accuracy on test set: {num_correct}/{num_total}')
"
```
---
12. COMMON INTERVIEW QUESTIONS & ANSWERS
Q1: "How would you improve throughput by 2×?"
Answer Options:
Option A (PE array):
> "Increase PE count to 32 (double width). This requires 32 DSP slices (still available: 220 total). Weight regfile grows to 256 words. BRAM count stays same. Would achieve 2× throughput on most layers. Bottleneck shifts to external memory (DDR) for larger models."
Option B (Pipelining):
> "Overlap compute stages: e.g., while one row-pair computes (tap_cnt 0..11), start loading next row-pair via DMA into alternate buffer. Requires state machine refactor but same hardware. Achieves ~1.5× improvement (not perfect 2× due to group serial dependency)."
Option C (Memory):
> "Current DDR bandwidth is underutilized (~10% of potential). Use wider DMA (32-bit AXI4-Stream instead of 16-bit). This requires protocol changes but doubles pixel throughput into input buffer. Would be limiting factor if extended to ResNet-50."
Q2: "What's the biggest limitation of your design?"
Answer:
> "BRAM count (32 instances used out of 140 available). Each input channel requires 1 BRAM per buffer (2 buffers = 2 BRAMs per channel × 16 = 32 BRAM). Can't scale to 32+ input channels without external memory. For production, I'd design a hierarchical BRAM + HBM solution: small BRAM cache for current tile, HBM for full feature maps. This adds complexity but supports modern networks."
Q3: "How do you handle rounding and quantization errors?"
Answer:
> "Two approaches:
> 1. **In hardware**: Round-to-nearest during accumulator right-shift (add 0.5 LSB before shift). Saturate to ±32K to avoid wraparound.
> 2. **In training**: Quantize-aware training (QAT). Train the network with Q6.9 quantization from epoch 1, so weights adapt to fixed-point. Achieves < 0.5% accuracy loss on my dog/cat classifier.
> 
> I also validate via **reference comparison**: compute same layer in PyTorch float32, quantize to Q6.9, compare to hardware output pixel-by-pixel. Differences < 1 count are acceptable (rounding), > 1 count indicate bugs (e.g., BUG 1 where channel summer was hardcoded)."
Q4: "Describe the PS/PL handshake in detail."
Answer:
> "Three-phase handshake:
> 
> **Phase 1 - Setup (PS → PL via AXI-Lite):**
> - Write configuration (input_h, input_w, output_h, output_w, num_in_ch, num_out_ch)
> - Write mode (00=conv, 01=conv+relu, 10=pool)
> - Assert START bit (CTRL[0])
>
> **Phase 2 - Data Exchange (DMA AXI4-Stream):**
> - FSM goes to LOAD_WEIGHTS, sets fsm_wait_state = 01
> - PS polls LOOP_STAT[25:24] and sees 01
> - PS initiates DMA MM2S of weight data, asserts TLAST at end
> - FSM receives TLAST via input_stream_controller, loads next state
> - Repeat for pixels: fsm_wait_state = 10, PS sends pixels via MM2S
>
> **Phase 3 - Completion (Polling):**
> - FSM computes, streams output via S2MM (DMA reads via AXI4-Stream Master)
> - When done, FSM sets done = 1
> - PS polls STATUS[0] and sees done = 1
> - PS reads LOOP_STAT to see final layer metrics (cur_oc, etc.)
> 
> This design allows PS to implement arbitrary DMA sequencing without hardware signaling complexity."
Q5: "What's one thing you'd do differently?"
Answer:
> "In retrospect, I'd **pipeline the output path** to reduce critical path. Currently, output_pixel_reg stores quantized data same cycle it's written, forcing quantizer logic to complete within one clock period. If I added one more pipeline stage (output_pixel_d) with pre-ReLU buffering, the quantizer could run at 150 MHz instead of 100 MHz, improving timing margin.
> 
> Second, I'd **parameterize the PE count** in the Verilog so it's easy to synthesize 8, 16, or 32 PE variants from the same RTL. Currently, it's hardcoded to 16. This would make it easier to trade area for throughput on different FPGA boards."
---
13. KEY VERILOG PATTERNS & SYNTHESIS DIRECTIVES
13.1 DSP Inference Pragmas
```verilog
// Force use of DSP48E1 for multipliers
(* use_dsp = "yes" *)
wire signed [31:0] product = pixel * weight;

// Alternative: distributed logic (if DSP oversubscribed)
(* use_dsp = "no" *)
wire [31:0] product = ...;
```
13.2 BRAM Inference
```verilog
// True dual-port BRAM (Xilinx inference)
module bram_dp #(
    parameter WIDTH = 16,
    parameter DEPTH = 512,
    parameter AWIDTH = 9
)(
    input clk, wr_en,
    input [AWIDTH-1:0] wr_addr, rd_addr,
    input [WIDTH-1:0] wr_data,
    output reg [WIDTH-1:0] rd_data
);
    (* ram_style = "block" *)
    reg [WIDTH-1:0] mem [DEPTH-1:0];
    
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];  // registered output
    end
endmodule
```
13.3 Generate Loop for Parallel Instantiation
```verilog
generate
    for (ch = 0; ch < NUM_CH; ch = ch + 1) begin : PE_ARRAY
        pe_unit u_pe (
            .pixel(pixel_flat[16*ch+15:16*ch]),
            .weight(weight_flat[16*ch+15:16*ch]),
            .acc(acc_flat[32*ch+31:32*ch])
        );
    end
endgenerate
```
Interview tip: Explain why generate loops are needed (parameterization, synthesis, scalability).
---
14. SUMMARY TABLE - LAYER-BY-LAYER SPECS
Layer	Mode	In CH	Out CH	In H×W	Out H×W	Groups	Est. Latency
Conv1	Conv+ReLU	3	16	128×128	126×126	1	2.2 ms
Pool1	MaxPool (PS)	16	16	126×126	63×63	1	0.8 ms
Conv2	Conv+ReLU	16	32	63×63	61×61	2	1.2 ms
Pool2	MaxPool (PS)	32	32	61×61	30×30	1	0.4 ms
Conv3	Conv+ReLU	32	64	30×30	28×28	4	0.6 ms
Pool3	MaxPool (PS)	64	64	28×28	14×14	1	0.2 ms
Conv4	Conv+ReLU	64	96	14×14	12×12	6	0.3 ms
Pool4	MaxPool (PS)	96	96	12×12	6×6	1	0.05 ms
Total PL	—	—	—	—	—	—	4.3 ms
Flatten → FC1 → ReLU → FC2 → Sigmoid	PS only	—	—	—	—	—	0.45 ms
Total Inference	—	—	—	—	—	—	≤ 4.75 ms
---
15. FINAL PREPARATION CHECKLIST
Before Your Interview
[ ] Understand the full datapath: Pixel → BRAM → PE → Accumulator → Quantizer → ReLU → Pool → Output BRAM → DMA
[ ] Memorize key numbers: 16 PE, 32 BRAM, 100 MHz, ~4.75 ms total, Q6.9/Q12.18 formats
[ ] Be ready to discuss tradeoffs: BRAM vs. throughput, DSP vs. LUT, pipelining vs. area
[ ] Prepare block diagrams: Hand-draw PE, channel summer, ping-pong buffer on whiteboard
[ ] Explain the FSM: Be able to walk through LOAD_WEIGHTS → WAIT → LOAD_INPUT → COMPUTE → WRITE_OUT sequence
[ ] Know the bugs: Explain each bug fix and what it taught you about FPGA design
[ ] Quantization deep dive: Q6.9/Q12.18, rounding, saturation, why those formats
[ ] PS/PL handshake: AXI-Lite for control, AXI4-Stream for data, fsm_wait_state polling
[ ] Synthesis pragmas: Understand `use_dsp`, `ram_style` directives
[ ] Answer scalability questions: How to extend to ResNet-50, higher throughput, more PEs
[ ] Practice explaining tradeoffs: Why ping-pong vs. other buffering? Why groups vs. full-batch? Etc.
Example Whiteboard Question (Likely to Come Up)
"Draw the data path for one output pixel, from input pixel to quantized output. Label every stage and indicate bit widths and latency."
```
Pixel (Q6.9, 16-bit) ─┐
                      ├─► [DSP Multiplier] ──► Product (Q12.18, 32-bit)
Weight (Q6.9, 16-bit)─┘                            │
                                                   ▼
                                        [PE Accumulator] (9 taps)
                                                   │
                                                   ▼
                                    [Channel Summer] (16→1 sum)
                                                   │
                                                   ▼
                              [Acc Row Buffer] (async read, no wait)
                                                   │
                                                   ▼
                              [Quantizer] +256 >> 9 (combinatorial)
                                                   │
                                                   ▼
                              [ReLU] (sign check, combinatorial)
                                                   │
                                                   ▼
                              [Output Buffer] (write, 1 cycle latency)
                                                   │
                                                   ▼
                              [AXI4-Stream Master] ──► DMA S2MM → DDR

Latencies:
  - DSP multiply: 2.5 ns (1 cycle @ 100 MHz)
  - 9-tap accumulation: 9 cycles
  - Channel sum: 1 cycle (tree) or pipelined
  - Quantize: 1 cycle (after pipelining fix)
  - ReLU: 0 cycles (combinatorial)
  - Total: ~10-11 cycles per output pixel
```
---
CLOSING THOUGHTS FOR NVIDIA INTERVIEW
Your project demonstrates:
End-to-end system thinking: You didn't just design an RTL block; you built a complete PS/PL pipeline with software drivers.
Optimization mindset: Ping-pong buffering, group processing, quantization strategy all show thoughtful tradeoff analysis.
Real debugging skills: Bug fixes (especially BUG 1, the hardcoded constant) show you validated against golden outputs.
Hardware expertise: DSP pragmas, BRAM inference, timing closure, fixed-point arithmetic all align with NVIDIA's GPU driver/compiler needs.
Scalability awareness: You can articulate how to extend to 32 PEs, larger models, and address memory bandwidth.
At NVIDIA, these skills matter:
GPU design involves similar streaming dataflow (warp scheduling, cache hierarchies)
Fixed-point and custom numeric formats (TensorFloat32, bfloat16, etc.)
Timing closure and power optimization
DMA and memory hierarchy design
Good luck! 🚀
