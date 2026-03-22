// ============================================================
// Module  : channel_summer
// Purpose : Adder tree summing 16 PE accumulator outputs
//           into one 32-bit value per output pixel
//           Purely combinational balanced adder tree
//           4 levels: 16?8?4?2?1
//
// Port:
//   acc_in : 512-bit flat bus = 16 x 32-bit PE outputs
//            acc_in[32*i+31 : 32*i] = PE[i] output
//   sum_out: 32-bit final sum
// ============================================================

module channel_summer (
    input  wire [511:0]          acc_in,    // 16 x 32-bit flattened
    output wire signed [31:0]    sum_out
);

    // Unpack flat bus into individual PE outputs
    wire signed [31:0] pe [0:15];

    genvar k;
    generate
        for (k = 0; k < 16; k = k + 1) begin : UNPACK
            assign pe[k] = acc_in[32*k+31 : 32*k];
        end
    endgenerate

    // Adder tree
    // Level 1: 16 ? 8
    wire signed [32:0] lvl1 [0:7];
    // Level 2: 8 ? 4
    wire signed [33:0] lvl2 [0:3];
    // Level 3: 4 ? 2
    wire signed [34:0] lvl3 [0:1];
    // Level 4: 2 ? 1
    wire signed [35:0] lvl4;

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : L1
            assign lvl1[i] = {pe[2*i][31],   pe[2*i]}   +
                             {pe[2*i+1][31], pe[2*i+1]};
        end
        for (i = 0; i < 4; i = i + 1) begin : L2
            assign lvl2[i] = {lvl1[2*i][32],   lvl1[2*i]}   +
                             {lvl1[2*i+1][32], lvl1[2*i+1]};
        end
        for (i = 0; i < 2; i = i + 1) begin : L3
            assign lvl3[i] = {lvl2[2*i][33],   lvl2[2*i]}   +
                             {lvl2[2*i+1][33], lvl2[2*i+1]};
        end
    endgenerate

    assign lvl4    = {lvl3[0][34], lvl3[0]} + {lvl3[1][34], lvl3[1]};

    // BUG 3 FIX: lvl4 is 36-bit signed; truncating straight to [31:0] would
    // silently drop the top 4 bits and wrap on overflow.
    // Saturate to int32 range before handing off to the quantizer.
    // In practice 16 Q6.9 MACs cannot overflow int32, but this is
    // correct-by-construction and prevents any synthesis-time surprises.
    wire overflow_pos = (lvl4[35:31] != 5'b00000); // positive overflow
    wire overflow_neg = (lvl4[35:31] != 5'b11111); // negative overflow
    wire overflow     = lvl4[35] ? overflow_neg : overflow_pos;

    assign sum_out = overflow
                     ? (lvl4[35] ? 32'sh80000000 : 32'sh7FFFFFFF)
                     : lvl4[31:0];

endmodule