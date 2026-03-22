// ============================================================
// Module  : weight_regfile
// Purpose : Kernel weight storage for 16 input channels x 9 taps
//
// Write address mapping:
//   addr = channel*9 + tap
//   addr 0   -> ch0 tap0 ... addr 8  -> ch0 tap8
//   addr 9   -> ch1 tap0 ... addr 17 -> ch1 tap8
//   ...
//   addr 135 -> ch15 tap0 ... addr 143 -> ch15 tap8
//
// Uses a small LUT (case statement) for addr decode - no division.
// ============================================================

module weight_regfile (
    input  wire         clk,
    input  wire         wr_en,
    input  wire [7:0]   wr_addr,    // 0..143
    input  wire [15:0]  wr_data,    // Q6.9 weight
    input  wire [3:0]   rd_tap,     // 0..8
    output wire [255:0] rd_flat     // 16 x 16-bit weights
);

    reg [15:0] weights [0:15][0:8];

    // Decode addr ? (ch, tap) using case - synthesizes to small LUT
    reg [3:0] wr_ch;
    reg [3:0] wr_tap;

    always @(*) begin
        case (wr_addr)
            8'd0:   begin wr_ch=0;  wr_tap=0; end
            8'd1:   begin wr_ch=0;  wr_tap=1; end
            8'd2:   begin wr_ch=0;  wr_tap=2; end
            8'd3:   begin wr_ch=0;  wr_tap=3; end
            8'd4:   begin wr_ch=0;  wr_tap=4; end
            8'd5:   begin wr_ch=0;  wr_tap=5; end
            8'd6:   begin wr_ch=0;  wr_tap=6; end
            8'd7:   begin wr_ch=0;  wr_tap=7; end
            8'd8:   begin wr_ch=0;  wr_tap=8; end
            8'd9:   begin wr_ch=1;  wr_tap=0; end
            8'd10:  begin wr_ch=1;  wr_tap=1; end
            8'd11:  begin wr_ch=1;  wr_tap=2; end
            8'd12:  begin wr_ch=1;  wr_tap=3; end
            8'd13:  begin wr_ch=1;  wr_tap=4; end
            8'd14:  begin wr_ch=1;  wr_tap=5; end
            8'd15:  begin wr_ch=1;  wr_tap=6; end
            8'd16:  begin wr_ch=1;  wr_tap=7; end
            8'd17:  begin wr_ch=1;  wr_tap=8; end
            8'd18:  begin wr_ch=2;  wr_tap=0; end
            8'd19:  begin wr_ch=2;  wr_tap=1; end
            8'd20:  begin wr_ch=2;  wr_tap=2; end
            8'd21:  begin wr_ch=2;  wr_tap=3; end
            8'd22:  begin wr_ch=2;  wr_tap=4; end
            8'd23:  begin wr_ch=2;  wr_tap=5; end
            8'd24:  begin wr_ch=2;  wr_tap=6; end
            8'd25:  begin wr_ch=2;  wr_tap=7; end
            8'd26:  begin wr_ch=2;  wr_tap=8; end
            8'd27:  begin wr_ch=3;  wr_tap=0; end
            8'd28:  begin wr_ch=3;  wr_tap=1; end
            8'd29:  begin wr_ch=3;  wr_tap=2; end
            8'd30:  begin wr_ch=3;  wr_tap=3; end
            8'd31:  begin wr_ch=3;  wr_tap=4; end
            8'd32:  begin wr_ch=3;  wr_tap=5; end
            8'd33:  begin wr_ch=3;  wr_tap=6; end
            8'd34:  begin wr_ch=3;  wr_tap=7; end
            8'd35:  begin wr_ch=3;  wr_tap=8; end
            8'd36:  begin wr_ch=4;  wr_tap=0; end
            8'd37:  begin wr_ch=4;  wr_tap=1; end
            8'd38:  begin wr_ch=4;  wr_tap=2; end
            8'd39:  begin wr_ch=4;  wr_tap=3; end
            8'd40:  begin wr_ch=4;  wr_tap=4; end
            8'd41:  begin wr_ch=4;  wr_tap=5; end
            8'd42:  begin wr_ch=4;  wr_tap=6; end
            8'd43:  begin wr_ch=4;  wr_tap=7; end
            8'd44:  begin wr_ch=4;  wr_tap=8; end
            8'd45:  begin wr_ch=5;  wr_tap=0; end
            8'd46:  begin wr_ch=5;  wr_tap=1; end
            8'd47:  begin wr_ch=5;  wr_tap=2; end
            8'd48:  begin wr_ch=5;  wr_tap=3; end
            8'd49:  begin wr_ch=5;  wr_tap=4; end
            8'd50:  begin wr_ch=5;  wr_tap=5; end
            8'd51:  begin wr_ch=5;  wr_tap=6; end
            8'd52:  begin wr_ch=5;  wr_tap=7; end
            8'd53:  begin wr_ch=5;  wr_tap=8; end
            8'd54:  begin wr_ch=6;  wr_tap=0; end
            8'd55:  begin wr_ch=6;  wr_tap=1; end
            8'd56:  begin wr_ch=6;  wr_tap=2; end
            8'd57:  begin wr_ch=6;  wr_tap=3; end
            8'd58:  begin wr_ch=6;  wr_tap=4; end
            8'd59:  begin wr_ch=6;  wr_tap=5; end
            8'd60:  begin wr_ch=6;  wr_tap=6; end
            8'd61:  begin wr_ch=6;  wr_tap=7; end
            8'd62:  begin wr_ch=6;  wr_tap=8; end
            8'd63:  begin wr_ch=7;  wr_tap=0; end
            8'd64:  begin wr_ch=7;  wr_tap=1; end
            8'd65:  begin wr_ch=7;  wr_tap=2; end
            8'd66:  begin wr_ch=7;  wr_tap=3; end
            8'd67:  begin wr_ch=7;  wr_tap=4; end
            8'd68:  begin wr_ch=7;  wr_tap=5; end
            8'd69:  begin wr_ch=7;  wr_tap=6; end
            8'd70:  begin wr_ch=7;  wr_tap=7; end
            8'd71:  begin wr_ch=7;  wr_tap=8; end
            8'd72:  begin wr_ch=8;  wr_tap=0; end
            8'd73:  begin wr_ch=8;  wr_tap=1; end
            8'd74:  begin wr_ch=8;  wr_tap=2; end
            8'd75:  begin wr_ch=8;  wr_tap=3; end
            8'd76:  begin wr_ch=8;  wr_tap=4; end
            8'd77:  begin wr_ch=8;  wr_tap=5; end
            8'd78:  begin wr_ch=8;  wr_tap=6; end
            8'd79:  begin wr_ch=8;  wr_tap=7; end
            8'd80:  begin wr_ch=8;  wr_tap=8; end
            8'd81:  begin wr_ch=9;  wr_tap=0; end
            8'd82:  begin wr_ch=9;  wr_tap=1; end
            8'd83:  begin wr_ch=9;  wr_tap=2; end
            8'd84:  begin wr_ch=9;  wr_tap=3; end
            8'd85:  begin wr_ch=9;  wr_tap=4; end
            8'd86:  begin wr_ch=9;  wr_tap=5; end
            8'd87:  begin wr_ch=9;  wr_tap=6; end
            8'd88:  begin wr_ch=9;  wr_tap=7; end
            8'd89:  begin wr_ch=9;  wr_tap=8; end
            8'd90:  begin wr_ch=10; wr_tap=0; end
            8'd91:  begin wr_ch=10; wr_tap=1; end
            8'd92:  begin wr_ch=10; wr_tap=2; end
            8'd93:  begin wr_ch=10; wr_tap=3; end
            8'd94:  begin wr_ch=10; wr_tap=4; end
            8'd95:  begin wr_ch=10; wr_tap=5; end
            8'd96:  begin wr_ch=10; wr_tap=6; end
            8'd97:  begin wr_ch=10; wr_tap=7; end
            8'd98:  begin wr_ch=10; wr_tap=8; end
            8'd99:  begin wr_ch=11; wr_tap=0; end
            8'd100: begin wr_ch=11; wr_tap=1; end
            8'd101: begin wr_ch=11; wr_tap=2; end
            8'd102: begin wr_ch=11; wr_tap=3; end
            8'd103: begin wr_ch=11; wr_tap=4; end
            8'd104: begin wr_ch=11; wr_tap=5; end
            8'd105: begin wr_ch=11; wr_tap=6; end
            8'd106: begin wr_ch=11; wr_tap=7; end
            8'd107: begin wr_ch=11; wr_tap=8; end
            8'd108: begin wr_ch=12; wr_tap=0; end
            8'd109: begin wr_ch=12; wr_tap=1; end
            8'd110: begin wr_ch=12; wr_tap=2; end
            8'd111: begin wr_ch=12; wr_tap=3; end
            8'd112: begin wr_ch=12; wr_tap=4; end
            8'd113: begin wr_ch=12; wr_tap=5; end
            8'd114: begin wr_ch=12; wr_tap=6; end
            8'd115: begin wr_ch=12; wr_tap=7; end
            8'd116: begin wr_ch=12; wr_tap=8; end
            8'd117: begin wr_ch=13; wr_tap=0; end
            8'd118: begin wr_ch=13; wr_tap=1; end
            8'd119: begin wr_ch=13; wr_tap=2; end
            8'd120: begin wr_ch=13; wr_tap=3; end
            8'd121: begin wr_ch=13; wr_tap=4; end
            8'd122: begin wr_ch=13; wr_tap=5; end
            8'd123: begin wr_ch=13; wr_tap=6; end
            8'd124: begin wr_ch=13; wr_tap=7; end
            8'd125: begin wr_ch=13; wr_tap=8; end
            8'd126: begin wr_ch=14; wr_tap=0; end
            8'd127: begin wr_ch=14; wr_tap=1; end
            8'd128: begin wr_ch=14; wr_tap=2; end
            8'd129: begin wr_ch=14; wr_tap=3; end
            8'd130: begin wr_ch=14; wr_tap=4; end
            8'd131: begin wr_ch=14; wr_tap=5; end
            8'd132: begin wr_ch=14; wr_tap=6; end
            8'd133: begin wr_ch=14; wr_tap=7; end
            8'd134: begin wr_ch=14; wr_tap=8; end
            8'd135: begin wr_ch=15; wr_tap=0; end
            8'd136: begin wr_ch=15; wr_tap=1; end
            8'd137: begin wr_ch=15; wr_tap=2; end
            8'd138: begin wr_ch=15; wr_tap=3; end
            8'd139: begin wr_ch=15; wr_tap=4; end
            8'd140: begin wr_ch=15; wr_tap=5; end
            8'd141: begin wr_ch=15; wr_tap=6; end
            8'd142: begin wr_ch=15; wr_tap=7; end
            8'd143: begin wr_ch=15; wr_tap=8; end
            default: begin wr_ch=0; wr_tap=0; end
        endcase
    end

    always @(posedge clk) begin
        if (wr_en)
            weights[wr_ch][wr_tap] <= wr_data;
    end

    genvar ch;
    generate
        for (ch = 0; ch < 16; ch = ch + 1) begin : RD
            assign rd_flat[16*ch+15 : 16*ch] = weights[ch][rd_tap];
        end
    endgenerate

endmodule