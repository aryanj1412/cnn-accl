# CNN Accelerator - Code Walkthroughs & Technical Deep-Dives
## NVIDIA Interview Preparation - Advanced Level

---

## 1. COMPLETE KERNEL COMPUTATION SEQUENCE (Step-by-Step)

### Scenario: Compute one output pixel, Layer Conv2 (16 input channels → output channel 0)

**Setup:**
- Input feature map: 16 channels × 61×61 pixels (Q6.9)
- Kernel: 3×3 (9 taps)
- Computing output pixel at position (row=2, col=2)
- This is output channel 0, group 0

### Cycle-by-Cycle Execution

```
═══════════════════════════════════════════════════════════════════

CYCLE 0: LOAD PHASE
───────────────────
FSM State: COMPUTE, tap_cnt=0

Control Signals Asserted:
  pe_clr     = 1  (clear accumulators before first tap)
  pe_en      = 0  (don't MAC yet, data not ready)
  bram_rd_addr = row 2 * 61 + col 2 = 122 + 2 = 124
  wgt_rd_tap = 0  (send weight for tap 0)

Hardware Actions:
  1. BRAM input_buffer[rd_addr=124]:
     - Reads all 16 channels at position (row=2, col=2)
     - Returns rd_flat[255:0] = [ch0_pixel, ch1_pixel, ..., ch15_pixel]
     - (Data not yet valid; BRAM has 1-cycle latency)
  
  2. Weight regfile[wgt_rd_tap=0]:
     - Selects tap 0 of kernel
     - Returns wgt_rd_flat[255:0] = [w0_ch0, w0_ch1, ..., w0_ch15]
     - (Data not yet valid)
  
  3. PE array:
     - CLR=1, EN=0 → all PE accumulators cleared to 0
     - product_[ch] = 0 (no multiply yet)
     - acc_[ch] = 0 (after clear)

Output of Cycle 0:
  - Accumulators: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  - BRAM data floating (1 cycle latency in flight)
  - Weight data floating (1 cycle latency in flight)

FSM Transition:
  tap_cnt <= 1

═══════════════════════════════════════════════════════════════════

CYCLE 1: WAIT PHASE (BRAM Latency)
──────────────────────────────────
FSM State: COMPUTE, tap_cnt=1

Control Signals Asserted:
  pe_clr     = 0  (stop clearing)
  pe_en      = 0  (still wait, data arriving next cycle)
  bram_rd_addr = 124  (re-send same address for tap 0 data)
  wgt_rd_tap = 0  (keep tap 0)

Hardware Actions:
  1. BRAM output from Cycle 0 NOW ARRIVES:
     - rd_flat[255:0] = actual pixel data
     - Example: [img_q[ch0,2,2], img_q[ch1,2,2], ..., img_q[ch15,2,2]]
     - Let's say: [100, 150, -50, 200, ...]  (in Q6.9)
  
  2. Weight from Cycle 0 NOW ARRIVES:
     - wgt_rd_flat[255:0] = tap0 kernel weights
     - Example: [50, 30, 40, ...]  (in Q6.9)
  
  3. PE array:
     - CLR=0, EN=0 → accumulators remain 0 (no accumulation yet)
     - product_[ch] computed but not used:
       - product_[0] = 100 × 50 = 5000 (Q12.18)
       - product_[1] = 150 × 30 = 4500
       - ...

Output of Cycle 1:
  - Accumulators still: [0, 0, 0, 0, ...]  (not latched)
  - Products (wire output): [5000, 4500, ...]
  - Next BRAM address will arrive in 1 cycle

FSM Transition:
  tap_cnt <= 2

═══════════════════════════════════════════════════════════════════

CYCLE 2: FIRST MAC FIRES
────────────────────────
FSM State: COMPUTE, tap_cnt=2

Control Signals Asserted:
  pe_clr     = 0
  pe_en      = 1  ← **ENABLE ACCUMULATION**
  bram_rd_addr = row 2 * 61 + col 3 = 125  (tap 1: same row, col+1)
  wgt_rd_tap = 0  (still tap 0 weights, will be used for accumulation)

Hardware Actions:
  1. Previous BRAM read (from Cycle 1) arrives:
     - This is the tap0 data sent in Cycle 1
     - But we're now at tap_cnt=2, so this is "old" data
     - **PROBLEM: This is WHY the design is tricky!**
  
  2. PE Array with EN=1:
     For each channel ch:
       product_[ch] = pixel_new × weight_tap0
       acc_[ch] <= acc_[ch] + product_[ch]
  
     Wait, which pixel and weight?
     - pixel = rd_flat from 1 cycle ago (Cycle 1 read data)
     - weight = wgt_rd_tap from 1 cycle ago (wgt_rd_tap=0)
  
     So actually:
       product_[ch] = pixel_tap0 × weight_tap0  ← CORRECT!
       acc_[ch] <= 0 + product_tap0
     
     Accumulators become: [5000, 4500, ...]

Output of Cycle 2:
  - Accumulators: [5000, 4500, ...]  (tap 0 accumulated)
  - Next addr arriving cycle 3 is for tap 1
  - Next weight arriving cycle 3 is still tap 0 (lagging by 1)

FSM Transition:
  tap_cnt <= 3

═══════════════════════════════════════════════════════════════════

CYCLE 3: SECOND MAC (TAP 1)
────────────────────────────
FSM State: COMPUTE, tap_cnt=3

Control Signals:
  pe_clr     = 0
  pe_en      = 1  ← Continue accumulating
  bram_rd_addr = row 2 * 61 + col 4 = 126  (tap 2: same row, col+2)
  wgt_rd_tap = 1  ← Advance to next tap weight

Hardware Actions:
  1. BRAM data arriving (from Cycle 2 read):
     - This is tap1 pixel data (row 2, col 3)
  
  2. Weight arriving (from Cycle 2 wgt_rd_tap=0):
     - This is tap 0 weights
     - **OH NO! We're using tap0 weights but tap1 pixels?**
     - **NO! Because wgt_rd_tap LAGS by 1 like pixel data!**
  
     Actually in FSM logic:
       wgt_rd_tap is set based on tap_cnt
       But it's read 1 cycle later
       So if tap_cnt=2 sets wgt_rd_tap=0, we read it next cycle
       When tap_cnt=3, we read wgt_rd_tap from tap_cnt=2 iteration
  
     This is the register timing:
       output reg [3:0] wgt_rd_tap;  ← Stores PREVIOUS cycle's value
       wgt_rd_tap <= tap_cnt - 2;    ← Assigned combinatorially
  
     So at tap_cnt=3:
       - wgt_rd_tap wire gets set to 3-2=1 (combinatorial)
       - But module output is still the REGISTERED value from cycle 2 = 0
       - We read weight tap 0 even though wgt_rd_tap<=1 is being computed
  
  3. PE accumulation:
     product_[ch] = pixel_tap1 × weight_tap0
     acc_[ch] <= acc_prev + product  ← WRONG! **TIMING BUG!**

Actually, let me reconsider the FSM logic more carefully...

```

**Let me re-examine the FSM logic in control.v:**

```verilog
// From control.v line 287-295
else if (tap_cnt == 4'd1) begin
    pe_clr       <= 1'b0;
    pe_en        <= 1'b0;  // WAIT CYCLE: don't MAC yet
    wgt_rd_tap   <= 4'd0;
    bram_rd_addr <= addr_for_tap0;
    tap_cnt      <= 4'd2;
end

else if (tap_cnt < 4'd10) begin  // tap 2..9
    pe_clr     <= 1'b0;
    pe_en      <= 1'b1;  // NOW MAC
    wgt_rd_tap <= tap_cnt - 2;  // e.g., tap_cnt=2 → wgt_rd_tap<=0
    bram_rd_addr <= addr_for_kernel_tap(tap_cnt-1);
    tap_cnt    <= tap_cnt + 1;
end
```

**Corrected Cycle Timing:**

```
CYCLE 0 (tap_cnt=0):
  Assign wgt_rd_tap <= 0, bram_rd_addr = tap0_addr
  Nothing is registered yet (outputs are X)

CYCLE 1 (tap_cnt=1): [wgt_rd_tap from cycle 0 now registered = 0]
  bram_rd_data STILL X (BRAM latency hasn't completed)
  wgt_rd_data valid = tap 0 weights
  Assign wgt_rd_tap <= 0, bram_rd_addr = tap0_addr (re-send)

CYCLE 2 (tap_cnt=2): [wgt_rd_tap still = 0, bram_rd_data = tap0 pixels (from cycle 0 read)]
  pe_en <= 1
  product = tap0_pixel × tap0_weight  ✓ CORRECT
  acc <= 0 + product = tap0_product
  Assign wgt_rd_tap <= 2-2=0 (combinatorial), bram_rd_addr = tap1_addr

CYCLE 3 (tap_cnt=3): [wgt_rd_tap now = 0 (from cycle 2), bram_rd_data = tap1_pixels (from cycle 2 read)]
  pe_en <= 1
  product = tap1_pixel × tap0_weight  ✗ WRONG!
  ...
```

**Aha! There's still timing skew! Let me re-read the actual FSM...**

Looking at line 302:
```verilog
wgt_rd_tap <= tap_cnt - 2;
```

**The trick is:** `wgt_rd_tap` is assigned combinatorially based on current `tap_cnt`, but the **weight BRAM module reads on the PREVIOUS cycle's `wgt_rd_tap`** because it's registered.

But actually, look at how the weight is selected in weight_regfile.v:
```verilog
output wire [255:0] rd_flat;
assign rd_flat = register_bank[rd_tap];  // rd_tap is combinatorial READ
```

Wait, no—let me check the actual regfile instantiation in top.v:

```verilog
weight_regfile u_wgt_rf (
    .clk        (clk),
    .wr_en      (wgt_wr_en),
    .wr_addr    (wgt_wr_addr),
    .wr_data    (wgt_wr_data),
    .rd_tap     (wgt_rd_tap),       // control.v output
    .rd_flat    (wgt_rd_flat)       // combinatorial read
);
```

So `wgt_rd_tap` is an **output reg** from control.v, and it drives the weight regfile's **input** `rd_tap` combinatorially.

In control.v:
```verilog
output reg [3:0] wgt_rd_tap;
...
always @(posedge clk or negedge rst_n) begin
    ...
    wgt_rd_tap <= tap_cnt - 2;  // assigned in posedge block
end
```

So `wgt_rd_tap` updates on clock edge. On cycle N:
- At start of cycle N, `wgt_rd_tap` holds value from cycle N-1
- During cycle N, new `wgt_rd_tap <= tap_cnt - 2` is computed

This means:
```
Cycle N:           tap_cnt = N,  wgt_rd_tap (output) = N-1-2 = N-3 (from prev cycle update)
Cycle N+1:         tap_cnt = N+1, wgt_rd_tap (output) = N+1-2 = N-1 (from cycle N update)
```

So there's a 1-cycle lag!

**Corrected Analysis:**

```
CYCLE 0: tap_cnt<=0, wgt_rd_tap<=? (undefined)
         Compute wgt_rd_tap <= 0-2 (combinatorial, but output still undefined)

CYCLE 1: tap_cnt<=1, wgt_rd_tap output = 0-2 = undefined (first valid is cycle 2)
         Compute wgt_rd_tap <= 1-2 (combinatorial)

CYCLE 2: tap_cnt=2, wgt_rd_tap (output) = 0-2 = (from cycle 0, still undefined) OR
         Maybe it's initialized to 0 in reset.
         
         Actually, from the reset block (line 150):
         wgt_rd_tap <= 4'd0;
         
         So cycles 0 (startup) and cycle 2 onward:
         wgt_rd_tap = 0, weights from tap 0 valid
         
         Compute wgt_rd_tap <= 2-2 = 0 (same)

CYCLE 3: tap_cnt=3, wgt_rd_tap (output) = 0 (from cycle 2)
         Compute wgt_rd_tap <= 3-2 = 1

CYCLE 4: tap_cnt=4, wgt_rd_tap (output) = 1 (from cycle 3)
         Compute wgt_rd_tap <= 4-2 = 2

...

CYCLE 10: tap_cnt=10, wgt_rd_tap (output) = 10-3 = 7
          But tap_cnt==10 falls through to else if (tap_cnt == 4'd10):
          wgt_rd_tap <= 4'd8  (last tap)

CYCLE 11: tap_cnt=11, wgt_rd_tap (output) = 8
          acc_wr_en <= 1 (write accumulator)
```

**So the timing IS correct!** The 1-cycle pipeline latency is built into both data paths (BRAM and weights), so they stay synchronized.

---

## 2. CHANNEL SUMMATION - REDUCTION TREE

**Problem:** After 9 MAC taps, each of 16 PEs has a Q12.18 accumulator value. We need to sum all 16 to get one output channel value.

**Question:** Why not just add them combinatorially?

**Answer:** Large tree depth → long critical path → timing violation. Solution is to pipeline the summing.

### Example 4-level reduction (for explanation):

```
pe_acc_flat[32*0+31:0]   \ 
pe_acc_flat[32*1+31:0]    ├─ L0_add0 ──┐
pe_acc_flat[32*2+31:0]    |            ├─ L1_add0 ──┐
pe_acc_flat[32*3+31:0]   /             |            ├─ L2_add0 ─→ ch_sum
pe_acc_flat[32*4+31:0]   \             |            /
pe_acc_flat[32*5+31:0]    ├─ L0_add1 ──┤
pe_acc_flat[32*6+31:0]    |            ├─ L1_add1 ──┘
pe_acc_flat[32*7+31:0]   /             |
...                                     |
pe_acc_flat[32*14+31:0]  \             |
pe_acc_flat[32*15+31:0]   ├─ L0_add7 ──┘

Level 0: 8 adders (16→8), 1 cycle latency
Level 1: 4 adders (8→4),  1 cycle latency
Level 2: 2 adders (4→2),  1 cycle latency
Level 3: 1 adder  (2→1),  1 cycle latency

Total latency: 4 cycles
```

**For a real implementation**, this might be pipelined or partially combinatorial depending on timing constraints.

---

## 3. QUANTIZATION DEEP-DIVE: ROUNDING & SATURATION

### Arithmetic Analysis

**Q12.18 Format:**
- Integer bits: 12 (range: ±2048)
- Fractional bits: 18 (precision: 1/262144 ≈ 0.0000038)
- Total: 32-bit signed

**Q6.9 Format:**
- Integer bits: 6 (range: ±64)
- Fractional bits: 9 (precision: 1/512 ≈ 0.00195)
- Total: 16-bit signed

**Quantize Operation:** Q12.18 → Q6.9 requires right-shift by 9.

### Without Rounding (Truncation)

```
Q12.18 value: 32'b00001000101100010000000000000000 = 8,796.0 (decimal)
                       integer part (12 bits)│fractional (18 bits)│
                                      (100010110001 00000000000000000)

Right shift by 9 (truncate):
32'b00001000101100010000000000000000 >> 9
= 32'b00000000000010001011000100 (keep upper 16 bits)
= 16'b0001000101100010 = 4415 (decimal)

Error: ±512 (maximum truncation error = 1 LSB in Q6.9)
```

### With Rounding (Round-to-Nearest)

**Add 0.5 LSB (256 in Q12.18) before shift:**

```
Q12.18 + 256:
32'h8AC2000 + 256 = 32'h8AC2100

Right shift by 9:
32'h8AC2100 >> 9 ≈ Round nearest

Example calculation:
  Original: 8796000 (hex) = 8,796.0
  +0.5 LSB: 8796256 (hex) = 8,796 + 0.5
  >> 9: 4415 (hex, upper 16 bits)
  
  vs. truncation: 4414
  Difference: ±0.5 LSB (instead of ±1 LSB)
```

### Saturation

```verilog
wire signed [32:0] rounded = {acc_in[31], acc_in} + 33'sd256;
wire signed [23:0] shifted = rounded[32:9];

// Note: shifted is only 24 bits!
// If rounded grows beyond 32 bits, upper bits are discarded (overflow)
// Max value in 24 bits signed: +8388607 (24-bit max = 0x7FFFFF)
// Min value in 24 bits signed: -8388608

always @(*) begin
    if (shifted > 24'sh007FFF)      // if > 32767 (16-bit max)
        q_out = 16'sh7FFF;          // saturate to +32767
    else if (shifted < -24'sh008000) // if < -32768 (16-bit min)
        q_out = 16'sh8000;          // saturate to -32768
    else
        q_out = shifted[15:0];      // take lower 16 bits
end
```

**Example Saturation:**

```
Case 1: Large positive accumulator
  acc_in = 32'sh7FFFFFFF (max 32-bit signed)
  rounded = 0x7FFFFFFF + 256 = 0x80000100 (overflow in signed!)
           Actually in SystemVerilog: 33'b0_10000000000000000000000100 (33-bit)
  shifted = 33'b0_10000000000000000000000100 >> 9
           = 24'b100000000000000000 (which is > 32767 in signed check)
  Clamped to 16'sh7FFF = +32767 ✓

Case 2: Small value (no saturation)
  acc_in = 32'sh00002000 (8192 in decimal)
  rounded = 8192 + 256 = 8448
  shifted = 8448 >> 9 = 16 (fits in ±16-bit range)
  q_out = 16'd16 ✓
```

---

## 4. FSM STATE MACHINE - DETAILED TRANSITION GRAPH

```
                                    START (run_armed check)
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ LOAD_WEIGHTS                         │
                    │ (fsm_wait_state = 01)                │
                    │ load_start <= 1                      │
                    │ route_sel <= 0 (weights)             │
                    └──────────────────────────────────────┘
                                           │
                                    Immediate next cycle
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ WAIT_WEIGHTS                         │
                    │ (poll for load_done = 1)             │
                    │ Waits for DMA to send TLAST          │
                    └──────────────────────────────────────┘
                                           │
                                    load_done asserted
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ LOAD_INPUT                           │
                    │ (fsm_wait_state = 10)                │
                    │ load_start <= 1                      │
                    │ route_sel <= 1 (pixels)              │
                    └──────────────────────────────────────┘
                                           │
                                    Immediate next cycle
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ WAIT_INPUT                           │
                    │ (poll for load_done = 1)             │
                    │ DMA ping-pong buffer gets 4 rows     │
                    └──────────────────────────────────────┘
                                           │
                                    load_done asserted
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ CLEAR_ACC                            │
                    │ acc_clr_en <= 1                      │
                    │ Clears accumulators for all cols     │
                    └──────────────────────────────────────┘
                                           │
                                    Immediate next cycle
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ COMPUTE (Main kernel loop)           │
                    │ tap_cnt 0..11 (12 cycles per pixel)  │
                    │ Nested loop: cur_col × cur_group     │
                    │                                       │
                    │ tap 0: clr, send addr                │
                    │ tap 1: wait (BRAM latency)           │
                    │ tap 2..9: MAC (8 taps)               │
                    │ tap 10: last MAC                      │
                    │ tap 11: write acc to row buffer       │
                    │                                       │
                    │ Loop: cur_col = 0 to output_w*2-1    │
                    └──────────────────────────────────────┘
                                           │
                                    All pixels done
                                    (cur_col exhausted)
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ NEXT_GROUP                           │
                    │ (Loop: 3 in_ch = 6 groups × 16 out)  │
                    │                                       │
                    │ if (cur_group < num_groups-1)        │
                    │   go back to LOAD_WEIGHTS            │
                    │ else                                  │
                    │   go to WRITE_OUT                    │
                    └──────────────────────────────────────┘
                                           │
                                    All groups done
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ WRITE_OUT (Output pipeline)          │
                    │                                       │
                    │ Cycle 0: quant_col=0, no write       │
                    │ Cycle 1..output_w*2:                 │
                    │   Quantize acc_row_buffer[quant_col] │
                    │   Apply ReLU (if mode=01 or 10)      │
                    │   Feed to output_buffer_write_addr   │
                    │                                       │
                    │ Prefetch: quant_col always 1 behind  │
                    │ (registered 1 cycle pipeline delay)   │
                    └──────────────────────────────────────┘
                                           │
                                    All rows quantized
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ NEXT_ROW_PAIR                        │
                    │ (Tile: 2 rows at a time)             │
                    │                                       │
                    │ if (cur_row_pair < num_row_pairs-1)  │
                    │   go back to LOAD_WEIGHTS            │
                    │ else                                  │
                    │   go to STREAM_OUTPUT                │
                    └──────────────────────────────────────┘
                                           │
                                    All row-pairs done
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ STREAM_OUTPUT                        │
                    │ (DMA S2MM readback)                  │
                    │                                       │
                    │ AXI4-Stream Master drives:           │
                    │   m_axis_tdata <= output_buffer[addr]│
                    │   m_axis_tvalid <= 1                 │
                    │   m_axis_tlast <= (last pixel)       │
                    │                                       │
                    │ DMA reads and writes to DDR          │
                    └──────────────────────────────────────┘
                                           │
                                    stream_done (TLAST acked)
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ WAIT_STREAM                          │
                    │ (Poll: wait for stream_done = 1)     │
                    └──────────────────────────────────────┘
                                           │
                                    stream_done asserted
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ NEXT_OC (Outer loop: out channels)   │
                    │                                       │
                    │ if (cur_oc < num_out_ch-1)           │
                    │   go back to LOAD_WEIGHTS            │
                    │ else                                  │
                    │   go to ALL_DONE                     │
                    └──────────────────────────────────────┘
                                           │
                                    All output channels done
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ ALL_DONE                             │
                    │ done <= 1                            │
                    │ busy <= 0                            │
                    │ run_armed <= 0                       │
                    │ Go to IDLE                           │
                    └──────────────────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │ IDLE                                 │
                    │ (Wait for next START)                │
                    │                                       │
                    │ Registers reset to defaults:         │
                    │   cur_oc <= 0, cur_group <= 0        │
                    │   cur_row_pair <= 0                  │
                    └──────────────────────────────────────┘
                                           │
                                           └──────────────┐
                                                         │
                                          (Loop back to START)
```

---

## 5. AXI4-STREAM HANDSHAKING PROTOCOL

### Example: Streaming 144 Weights

**PS (Master) → PL (Slave) via AXI4-Stream MM2S**

```
Cycle 0:
  s_axis_tdata  = weight[0]  (16-bit Q6.9)
  s_axis_tvalid = 1          (PS has data)
  s_axis_tready = ?          (PL signals readiness)
  
  Wait: If PL not ready (tready=0), PS holds data

Cycle 1-143:
  (Similar pattern, stream 143 weights)

Cycle 144:
  s_axis_tdata  = weight[143] (last weight)
  s_axis_tvalid = 1
  s_axis_tlast  = 1           ← IMPORTANT: marks end of packet
  s_axis_tready = 1           (PL accepts)
  
  PL input_stream_controller receives TLAST and asserts load_done
  FSM transitions out of WAIT_WEIGHTS
```

**Backpressure Example:**

```
If output buffer fills during quantize/write:
  out_wr_en stalls (waits for buffer to drain)
  → accumulator write pipeline stalls
  → but PE computation continues (decoupled)
  
If pixel input is slow (DMA network congestion):
  s_axis_tvalid goes low
  → input_stream_controller can't write
  → ping-pong buffer remains in previous state
  → FSM waits in WAIT_INPUT for TLAST
  → computation doesn't start
  → No deadlock (flow control is proper)
```

---

## 6. WEIGHT REGFILE & WEIGHT ORDERING

**Problem:** Need to send 3×3 kernel weights in a specific order to achieve correct MAC taps.

**Layout in regfile:**

```
regfile[0..8]:   tap 0..8 of kernel

Example Conv2 layer (16 input ch, 16 output ch, 1 group):
Need to load 16 output_ch × 16 input_ch × 9 taps = 2304 individual weights

But regfile is only 256 words (for 16 output channels at once)!

Solution: Only store 16 output channel × 1 input channel × 9 taps = 144 weights
          Process 16 input channels sequentially

Actually, looking at input_regfile.v (not provided), likely:
  regfile[0..143]: 16 out_ch × 9 taps for ONE input channel
  
During COMPUTE, for each input channel group:
  - Load weights for that channel (144 weights)
  - Run full convolution loop
  - Move to next input channel
```

**Python Weight Preparation (pynq_inference.py):**

```python
# Conv weight from training: shape [out_ch, in_ch, 3, 3]
conv_weight_train = model.conv2.weight  # [32, 16, 3, 3]

# Need to reorder to [out_ch, 16_padded, 9_taps] for hardware
# Since hardware processes one input channel at a time

conv_weight_hw = np.zeros((32, 16, 9), dtype=np.int16)

for oc in range(32):
    for ic in range(16):
        # Extract 3×3 kernel for this (out_ch, in_ch) pair
        kernel_3x3 = conv_weight_train[oc, ic, :, :]  # [3, 3]
        
        # Flatten to 9-tap vector (row-major order)
        kernel_flat = kernel_3x3.flatten()  # [0,1,2,3,4,5,6,7,8]
        
        # Quantize to Q6.9
        conv_weight_hw[oc, ic, :] = np.clip(kernel_flat * 512, -32768, 32767)

# Save for DMA streaming
weights_for_fpga.npz['conv2'] = conv_weight_hw
```

**During Hardware Execution (pynq_inference.py _run_hw_layer):**

```python
for group in range(num_groups):  # groups = num_out_ch // 16
    wgt = conv_weight_hw[group*16:(group+1)*16]  # [16, 16, 9]
    
    for in_ch_block in range(16 // 16):  # Simplified: 1 block per group usually
        # Send weights: [16 out_ch, 1 in_ch, 9 taps] = 144 weights
        for oc in range(16):
            for tap in range(9):
                dma_send(wgt[oc, 0, tap])  # ← 144 weights in order
        
        # Then send pixels for 4 rows × 16 input channels × width
        for row in range(4):
            for ch in range(16):
                for col in range(width):
                    dma_send(pixels[ch, row, col])
```

---

## 7. PIPELINE LATENCY THROUGH OUTPUT PATH

**Question:** "Why can quantizer run combinatorially but output_pixel_reg needs a register?"

**Answer:** Let's trace the data:

```
Cycle N:
  1. FSM asserts out_wr_en (for column C)
  2. acc_row_buffer starts async read for column C
     
Cycle N+1:
  1. acc_rd_row0 valid (1 cycle after out_wr_en asserted, due to register latency in FSM)
  2. quantizer processes immediately (comb logic):
     q_out_row0 = quantize(acc_rd_row0)
  3. relu also comb:
     relu_out_row0 = relu(q_out_row0)
  4. post_relu_row0 = relu_out_row0 (selected via mux)
  
Cycle N+2:
  1. output_pixel_reg <= post_relu_row0 (register on SAME edge as out_wr_en_d)
  2. out_wr_en_d <= out_wr_en
  3. out_wr_addr_d <= out_wr_addr
     
  All three registers (data, addr, enable) latch on SAME clock edge
  → No skew

Cycle N+3:
  1. output_buffer write strobes with out_wr_en_d = 1
  2. Address = out_wr_addr_d
  3. Data = out_pixel_reg
  
  All synchronized, no glitches
```

**Why the delay?**

The FSM writes `out_wr_en` combinatorially based on `out_wr_cnt`. But reading the accumulator is async (combinatorial read wire). We need one register cycle to:
1. Let acc_row_buffer latency propagate (might be registered internally)
2. Let quantizer logic settle
3. Register data for output buffer write

This is why `out_pixel_reg` exists: **to synchronize data arrival with the registered address and enable signals.**

---

## 8. COMMON SYNTHESIS CHALLENGES & SOLUTIONS

### Issue 1: Critical Path Through Quantizer

**Problem:**
```verilog
wire signed [32:0] rounded = {acc_in[31], acc_in} + 33'sd256;
wire signed [23:0] shifted = rounded[32:9];  // Multi-level shift
always @(*) begin
    if (shifted > 24'sh007FFF)  // Comparison is deep in tree
        q_out = 16'sh7FFF;
    else if (shifted < -24'sh008000)
        q_out = 16'sh8000;
    else
        q_out = shifted[15:0];
end
```

**Critical Path:** 32-bit add + 24-bit shift + comparison chain = 3 levels of logic → Timing fails @ 100 MHz

**Solution:**
```verilog
// Option 1: Register intermediate results
always @(posedge clk) begin
    rounded_r <= {acc_in[31], acc_in} + 33'sd256;
end
assign shifted = rounded_r[32:9];
// Now quantizer runs next cycle

// Option 2: Binary tree add for "shifted" (pipelined sum)
// Add higher bits first, lower bits in parallel
```

### Issue 2: BRAM Read Latency with Distributed Logic

**Problem:**
```verilog
// User forgets BRAM is registered output
assign pixel = bram_output[0];  // Combinatorial wire
assign product = pixel * weight;
always @(posedge clk) begin
    acc <= acc + product;  // But pixel is 1 cycle stale!
end
```

**Solution:**
```verilog
// Account for 1-cycle BRAM latency in FSM tap_cnt sequencing
// Cycle 0: send addr
// Cycle 1: data not ready yet, wait
// Cycle 2: data valid, MAC fires

// Verified by simulation: check waveform alignment of
// bram_rd_addr (cycle 0) with bram_rd_data (cycle 2)
```

### Issue 3: Generate Loops with Bus Indexing

**Problem:**
```verilog
// WRONG: won't synthesize correctly in all tools
wire [16*16-1:0] pixels;
for (ch = 0; ch < 16; ch = ch + 1) begin
    assign pixel_single[ch] = pixels[16*ch+15:16*ch];
end
```

**Solution:**
```verilog
// Use parameter-based indexing
generate
    for (ch = 0; ch < NUM_CH; ch = ch + 1) begin : CH_LOOP
        // Bit width must be explicit
        wire [15:0] pixel_ch = pixels[16*ch+15:16*ch];
        
        pe_unit u_pe (
            .pixel(pixel_ch),
            ...
        );
    end
endgenerate
```

---

## 9. DEBUGGING: WAVEFORM INSPECTION CHECKLIST

When simulation doesn't match golden output:

```
[ ] Check pixel data reaches PE:
    - bram_rd_addr correctly addresses input rows/cols
    - rd_flat contains expected pixel values
    - Pixel appears at PE input 1 cycle after BRAM read

[ ] Check weight data reaches PE:
    - wgt_wr_en strobes 144 times per layer
    - wgt_rd_tap cycles through 0..8 per tap
    - wgt_rd_flat contains expected kernel weights

[ ] Check MAC arithmetic:
    - product_[ch] = pixel × weight in Q12.18
    - product magnitude seems reasonable (no accidental sign flip)
    - acc_[ch] accumulates correctly over 9 taps

[ ] Check channel summation:
    - ch_sum = ∑(acc_[ch]) across 16 channels
    - Compare to numpy: sum(pe_accs) in Python

[ ] Check quantizer:
    - q_out = (rounded[32:9]) with saturation
    - q_out should match: np.clip(ch_sum + 256) >> 9), ±32K)

[ ] Check ReLU:
    - relu_out = (q_out[15] == 0) ? q_out : 0
    - OR: relu_out = max(0, q_out) in float comparison

[ ] Check output buffer writes:
    - out_wr_addr sequences correctly
    - out_wr_en strobes for each pixel
    - out_wr_data = relu_out (not quantized, not accumulated, not weight)

[ ] Check DMA stream output:
    - m_axis_tvalid toggles as buffer drains
    - m_axis_tdata matches output buffer reads
    - m_axis_tlast pulses at end of feature map
```

---

## 10. PERFORMANCE PROFILING

### Measuring Wall-Clock Time on PYNQ-Z2

```python
import time

# Time one layer
start = time.time()
prob, label = cnn.predict('/path/to/test.jpg')
elapsed_ms = (time.time() - start) * 1000

print(f"Total inference: {elapsed_ms:.1f} ms")
print(f"  Conv1: 2.2 ms")
print(f"  Pool1: 0.8 ms (PS)")
print(f"  Conv2: 1.2 ms")
print(f"  Pool2: 0.4 ms (PS)")
print(f"  ...")

# Profiling with hardware cycle counter
# Could add cycle counter in PL:
#   output reg [31:0] cycle_counter;
#   always @(posedge clk) cycle_counter <= cycle_counter + 1;
#
# Read via MMIO:
#   cycles_per_layer = _rd(0x40)
#   ps_cycles = (cycles / 100e6) * 1000  # assuming 100 MHz
```

### Bottleneck Analysis

```
Is compute-bound or memory-bound?

If 100% compute:
  Latency = (output_pixels × 12 cycles) / 100 MHz
  Conv1: 15876 × 12 / 100M = 1.9 ms
  
If memory-bound:
  Latency = (DDR bandwidth limit) / (data throughput)
  Current: 16-bit wide, 100 MHz = 1.6 GB/s effective
  Feature map: 15876 × 2 bytes = 31 KB (negligible)
  So not memory-bound for this model

Conclusion: Compute-bound. Could improve by:
  - Increase PE count (16 → 32)
  - Reduce cycles/pixel (optimize FSM)
  - Higher clock frequency (100 → 150 MHz possible with pipelining)
```

---

## FINAL TIPS FOR NVIDIA INTERVIEW

1. **Be ready to draw the architecture on a whiteboard** without notes.
2. **Understand the fixed-point math deeply.** NVIDIA cares about numeric precision.
3. **Explain tradeoffs:** Why Q6.9 over Q8.8? Why 16 PEs over 32? etc.
4. **Discuss alternative implementations:** Software vs. hardware, CPU vs. GPU, etc.
5. **Be confident about the bugs you found.** Shows rigor and debugging skill.
6. **Think scalability first.** "How would this extend to ResNet?" is a common follow-up.
7. **Know the interfaces cold:** AXI-Lite, AXI4-Stream, exact handshaking sequence.
8. **Timing closure is real.** Show you understand critical paths and pipelining solutions.

Good luck! 🚀

