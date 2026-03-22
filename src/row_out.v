// ============================================================
// Module  : acc_row_buffer
// Purpose : 2 x 128 x 32-bit row accumulators
//
// Key change from previous version:
//   Removed synchronous for-loop clear
//   Instead use col counter to clear one entry per cycle
//   before compute starts (FSM must allow 128 clear cycles)
//   OR
//   Use separate "valid" tracking (simpler)
//
// Actually simplest fix: use LUTRAM not FF array
//   128 x 32-bit = 4096 bits = 8 RAMB18 if block
//   or 64 LUT6 if distributed
//   Either way much better than 4096 FFs with reset
//
// Clear strategy:
//   clr_en + clr_col: FSM sequences col 0..output_w-1
//   clearing one entry per cycle before compute
//   Takes output_w cycles = 126 cycles (acceptable)
// ============================================================

module acc_row_buffer #(
    parameter MAX_W  = 128,
    parameter AWIDTH = 8
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Clear: FSM sequences clr_col from 0 to output_w-1
    input  wire                  clr_en,
    input  wire [AWIDTH-1:0]     clr_col,

    // Write: accumulate channel_summer output
    input  wire                  wr_en,
    input  wire                  wr_row_sel,
    input  wire [AWIDTH-1:0]     wr_col,
    input  wire signed [31:0]    wr_data,

    // Read: quantizer reads after all groups done
    // Async (combinational) read - lutram supports this in Xilinx tools
    input  wire [AWIDTH-1:0]     rd_col,
    output wire signed [31:0]    rd_row0,
    output wire signed [31:0]    rd_row1
);

    // Use distributed RAM (lutram) for small size
    // 128 x 32 = 4096 bits = 64 LUT6 only
    (* ram_style = "distributed" *) reg signed [31:0] row0 [0:MAX_W-1];
    (* ram_style = "distributed" *) reg signed [31:0] row1 [0:MAX_W-1];

integer i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < MAX_W; i = i + 1) begin
            row0[i] <= 0;
            row1[i] <= 0;
        end
    end
    else begin
        if (clr_en) begin
            row0[clr_col] <= 32'sd0;
            row1[clr_col] <= 32'sd0;
        end

        if (wr_en) begin
            if (!wr_row_sel)
                row0[wr_col] <= row0[wr_col] + wr_data;
            else
                row1[wr_col] <= row1[wr_col] + wr_data;
        end
    end
end


    // Asynchronous (combinational) reads from distributed lutram
    assign rd_row0 = row0[rd_col];
    assign rd_row1 = row1[rd_col];

endmodule