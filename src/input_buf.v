// ============================================================
// Module  : input_pingpong_buffer
// Purpose : Ping-pong input buffer using 32 BRAM instances
//           16 channels ? 2 buffers = 32 BRAMs
//           Each BRAM = bram_sp (reliably inferred)
//
// Write: one channel written per cycle (wr_ch selects bank)
//        write goes to inactive buffer only
//
// Read:  all 16 channels read simultaneously every cycle
//        read from active buffer only
//        1 cycle BRAM latency
//
// Ping-pong:
//   sel=0: buf_a ? PE reads, buf_b ? DMA writes
//   sel=1: buf_b ? PE reads, buf_a ? DMA writes
// ============================================================

module input_pingpong_buffer #(
    parameter MAX_W  = 128,
    parameter NUM_CH = 16,
    parameter ROWS   = 4,
    parameter DEPTH  = 512,    // ROWS * MAX_W = 4*128
    parameter AWIDTH = 9
)(
    input  wire          clk,
    input  wire          sel,
    input  wire          wr_en,
    input wire  rst_n,
    input  wire [3:0]    wr_ch,
    input  wire [8:0]    wr_addr,
    input  wire [15:0]   wr_data,
    input  wire [8:0]    rd_addr,
    output wire [255:0]  rd_flat
);

    // -------------------------------------------------------
    // Write enable per channel per buffer
    // Only the inactive buffer gets written
    // Only the selected channel bank gets written
    // -------------------------------------------------------
    wire [0:NUM_CH-1] wr_a ;  // write enables for buffer A banks
    wire [0:NUM_CH-1] wr_b ;  // write enables for buffer B banks
    reg [31:0] dbg_write_count;
    
    // BUG 7 FIX: dbg_write_count had no reset; initialised it with rst_n
    always @ (posedge clk or negedge rst_n) begin
        if (!rst_n)
            dbg_write_count <= 32'd0;
        else if (wr_en)
            dbg_write_count <= dbg_write_count + 1;
    end
    genvar ch;
    generate
        for (ch = 0; ch < NUM_CH; ch = ch + 1) begin : WR_EN_GEN
            // sel=0 ? A is active ? write B
            // sel=1 ? B is active ? write A
            assign wr_a[ch] = wr_en & (sel == 1'b1) & (wr_ch == ch[3:0]);
            assign wr_b[ch] = wr_en & (sel == 1'b0) & (wr_ch == ch[3:0]);
        end
    endgenerate

    // -------------------------------------------------------
    // Read outputs from each BRAM
    // -------------------------------------------------------
    wire [15:0] rd_a [0:NUM_CH-1];
    wire [15:0] rd_b [0:NUM_CH-1];

    // -------------------------------------------------------
    // Instantiate 32 BRAMs
    // Buffer A: 16 BRAMs, one per channel
    // Buffer B: 16 BRAMs, one per channel
    // -------------------------------------------------------
    generate
        for (ch = 0; ch < NUM_CH; ch = ch + 1) begin : BRAM_A
            bram_dp #(
                .WIDTH  (16),
                .DEPTH  (DEPTH),
                .AWIDTH (AWIDTH)
            ) u_bram_a (
                .clk     (clk),
                .wr_en   (wr_a[ch]),
                .wr_addr (wr_addr),
                .rd_addr (rd_addr),
                .wr_data (wr_data),
                .rd_data (rd_a[ch])
            );
        end

        for (ch = 0; ch < NUM_CH; ch = ch + 1) begin : BRAM_B
            bram_dp #(
                .WIDTH  (16),
                .DEPTH  (DEPTH),
                .AWIDTH (AWIDTH)
            ) u_bram_b (
                .clk     (clk),
                .wr_en   (wr_b[ch]),
                .wr_addr (wr_addr),
                .rd_addr (rd_addr),
                .wr_data (wr_data),
                .rd_data (rd_b[ch])
            );
        end
    endgenerate

    // -------------------------------------------------------
    // Output mux: active buffer to rd_flat
    // sel=0 ? read from A
    // sel=1 ? read from B
    // -------------------------------------------------------
    generate
        for (ch = 0; ch < NUM_CH; ch = ch + 1) begin : OUT_MUX
            assign rd_flat[16*ch+15 : 16*ch] =
                (sel == 1'b0) ? rd_a[ch] : rd_b[ch];
        end
    endgenerate

endmodule