// ============================================================
// Module  : compute_core
// Purpose : 16 PE units in parallel, one per input channel
//
// Array ports not supported in Verilog-2001 synthesis
// Use flattened buses:
//   pixel_flat  [255:0] = 16 x 16-bit pixels
//   weight_flat [255:0] = 16 x 16-bit weights
//   acc_flat    [511:0] = 16 x 32-bit accumulators
//
//   pixel_flat[16*i+15:16*i]  = pixel[i]
//   weight_flat[16*i+15:16*i] = weight[i]
//   acc_flat[32*i+31:32*i]    = acc[i]
// ============================================================

module compute_core (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clr,
    input  wire          en,
    input  wire [255:0]  pixel_flat,    // 16 x 16-bit Q6.9
    input  wire [255:0]  weight_flat,   // 16 x 16-bit Q6.9
    output wire [511:0]  acc_flat       // 16 x 32-bit Q12.18
);

    genvar ch;
    generate
        for (ch = 0; ch < 16; ch = ch + 1) begin : PE_ARRAY
            pe_unit u_pe (
                .clk    (clk),
                .rst_n  (rst_n),
                .clr    (clr),
                .en     (en),
                .pixel  (pixel_flat[16*ch+15 : 16*ch]),
                .weight (weight_flat[16*ch+15 : 16*ch]),
                .acc    (acc_flat   [32*ch+31 : 32*ch])
            );
        end
    endgenerate

endmodule