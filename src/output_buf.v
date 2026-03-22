// ============================================================
// Module  : output_buffer
// Purpose : Single BRAM for one complete output channel
//           (* ram_style = "block" *) forces BRAM inference
//
// Size: max 128x128x16-bit = 16KB = 8 BRAM18
// ============================================================

module output_buffer #(
    parameter MAX_H = 128,
    parameter MAX_W = 128,
    parameter DEPTH = MAX_H * MAX_W   // 16384
)(
    input  wire          clk,
    input  wire          wr_en,
    input  wire [13:0]   wr_addr,
    input  wire [15:0]   wr_data,
    input  wire [13:0]   rd_addr,
    output reg  [15:0]   rd_data
);

    (* ram_style = "block" *) reg [15:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end

endmodule