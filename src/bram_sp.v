module bram_dp #(
    parameter WIDTH = 16,
    parameter DEPTH = 512,
    parameter AWIDTH = 9
)(
    input  wire              clk,

    // write port
    input  wire              wr_en,
    input  wire [AWIDTH-1:0] wr_addr,
    input  wire [WIDTH-1:0]  wr_data,

    // read port
    input  wire [AWIDTH-1:0] rd_addr,
    output reg  [WIDTH-1:0]  rd_data
);

(* ram_style = "block" *) reg [WIDTH-1:0] mem [0:DEPTH-1];

always @(posedge clk) begin
    if (wr_en)
        mem[wr_addr] <= wr_data;

    rd_data <= mem[rd_addr];
end

endmodule