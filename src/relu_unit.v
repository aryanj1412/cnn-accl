// ============================================================
// Module  : relu_unit
// Purpose : ReLU activation - purely combinational, zero cost
//           ReLU(x) = max(0, x)
//           For Q6.9 signed: just check sign bit (MSB)
// ============================================================

module relu_unit (
    input  wire signed [15:0] data_in,
    output wire signed [15:0] data_out
);
    assign data_out = data_in[15] ? 16'sh0 : data_in;
endmodule