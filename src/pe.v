// ============================================================
// Module  : pe_unit
// Purpose : Single Processing Element
//           1 DSP reused 9 times (one per tap per cycle)
//
// clr and en priority:
//   clr=1, en=1 ? acc = product  (clear then load tap0)
//   clr=1, en=0 ? acc = 0
//   clr=0, en=1 ? acc = acc + product
//   clr=0, en=0 ? acc unchanged
//
// Fixed Point:
//   pixel/weight: 16-bit Q6.9
//   product:      32-bit Q12.18
//   acc:          32-bit Q12.18
// ============================================================

module pe_unit (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clr,
    input  wire                  en,
    input  wire signed [15:0]    pixel,
    input  wire signed [15:0]    weight,
    output wire signed [31:0]    acc
);

    (* use_dsp = "yes" *)
    wire signed [31:0] product;
    assign product = pixel * weight;

    (* ram_style = "registers" *)
    reg signed [31:0] acc_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_reg <= 32'sd0;
        else if (clr && en)
            // tap 0 of new pixel: clear and load first MAC result
            acc_reg <= product;
        else if (clr)
            acc_reg <= 32'sd0;
        else if (en)
            acc_reg <= acc_reg + product;
    end

    assign acc = acc_reg;

endmodule