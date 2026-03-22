//// ============================================================
//// Module  : quantizer
//// Purpose : Convert Q12.18 accumulator to Q6.9 (16-bit)
////           Shift right by 9 with round-to-nearest and saturation
//// ============================================================

module quantizer (
    input  wire signed [31:0] acc_in,
    output reg  signed [15:0] q_out
);
    // Round-to-nearest: add 0.5 LSB (= 2^8 = 256 in Q12.18) then shift right 9
    wire signed [32:0] rounded = {acc_in[31], acc_in} + 33'sd256;
    wire signed [23:0] shifted = rounded[32:9];   // arithmetic right shift by 9

    always @(*) begin
        if (shifted > 24'sh007FFF)
            q_out = 16'sh7FFF;
        else if (shifted < -24'sh008000)
            q_out = 16'sh8000;
        else
            q_out = shifted[15:0];
    end


endmodule