// ============================================================
// Module  : maxpool_unit
// Purpose : 2?2 max pooling
//
// Operation:
//   Receives pixels col by col, both rows simultaneously.
//   Even col c:   buffer both rows
//   Odd  col c+1: horizontal max for each row, then vertical max
//                 ? single output = max of all 4 pixels in 2?2 window
//
// Signals:
//   pixel_r0  : current pixel for row 0 (top row of pair)
//   pixel_r1  : current pixel for row 1 (bottom row of pair)
//   col_idx   : current column index (0..input_w-1)
//   en        : pixel valid this cycle
//   pool_out  : 2?2 max result (one per two input columns)
//   pool_valid: output valid (fires at every odd column)
// ============================================================

module maxpool_unit (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  en,
    input  wire signed [15:0]    pixel_r0,    // Q6.9 row 0
    input  wire signed [15:0]    pixel_r1,    // Q6.9 row 1
    input  wire [7:0]            col_idx,     // BUG 6 FIX: was [6:0], mismatched quant_col in top.v
    output reg  signed [15:0]    pool_out,    // 2?2 max result
    output reg                   pool_valid   // one pulse per 2-col window
);

    reg signed [15:0] buf_r0;   // row 0 pixel at even col
    reg signed [15:0] buf_r1;   // row 1 pixel at even col

    wire is_odd_col = col_idx[0];

    // Combinational 2?2 max
    wire signed [15:0] hmax_r0 = (buf_r0 >= pixel_r0) ? buf_r0 : pixel_r0;
    wire signed [15:0] hmax_r1 = (buf_r1 >= pixel_r1) ? buf_r1 : pixel_r1;
    wire signed [15:0] vmax    = (hmax_r0 >= hmax_r1) ? hmax_r0 : hmax_r1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_r0     <= 16'sh0;
            buf_r1     <= 16'sh0;
            pool_out   <= 16'sh0;
            pool_valid <= 1'b0;
        end else begin
            pool_valid <= 1'b0;
            if (en) begin
                if (!is_odd_col) begin
                    buf_r0 <= pixel_r0;
                    buf_r1 <= pixel_r1;
                end else begin
                    pool_out   <= vmax;   // max of all 4 pixels in window
                    pool_valid <= 1'b1;
                end
            end
        end
    end

endmodule