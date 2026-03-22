// ============================================================
// Module  : axi_lite_slave
// Register Map:
//   0x00 CTRL       [0]=start (fires 1 cycle; FSM ignores replays via run_armed)
//                   [2:1]=mode  00=conv 01=conv+relu 10=pool
//   0x04 STATUS     [0]=done (sticky) [1]=busy
//   0x08 IMG_DIM    [7:0]=input_h  [15:8]=input_w
//   0x0C OUT_DIM    [7:0]=output_h [15:8]=output_w
//   0x10 CH_CFG     [7:0]=num_in_ch [15:8]=num_out_ch
//   0x20 LOOP_STAT  [7:0]=cur_oc [15:8]=cur_group
//                   [22:16]=cur_row_pair [25:24]=fsm_wait_state
//
// AXI-Lite write handling:
//   Both AW and W channels are accepted independently.
//   Address and data are latched separately.
//   Write executes when BOTH have been received (order-independent).
// ============================================================

module axi_lite_slave #(
    parameter ADDR_W = 6,
    parameter DATA_W = 32
)(
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire [ADDR_W-1:0]     s_axi_awaddr,
    input  wire                  s_axi_awvalid,
    output reg                   s_axi_awready,
    input  wire [DATA_W-1:0]     s_axi_wdata,
    input  wire [DATA_W/8-1:0]   s_axi_wstrb,
    input  wire                  s_axi_wvalid,
    output reg                   s_axi_wready,
    output reg  [1:0]            s_axi_bresp,
    output reg                   s_axi_bvalid,
    input  wire                  s_axi_bready,
    input  wire [ADDR_W-1:0]     s_axi_araddr,
    input  wire                  s_axi_arvalid,
    output reg                   s_axi_arready,
    output reg  [DATA_W-1:0]     s_axi_rdata,
    output reg  [1:0]            s_axi_rresp,
    output reg                   s_axi_rvalid,
    input  wire                  s_axi_rready,

    output reg                   start,
    output reg  [1:0]            mode,
    output reg  [7:0]            input_h,
    output reg  [7:0]            input_w,
    output reg  [7:0]            output_h,
    output reg  [7:0]            output_w,
    output reg  [7:0]            num_in_ch,
    output reg  [7:0]            num_out_ch,
    output reg  [31:0]           ddr_input,
    output reg  [31:0]           ddr_weight,
    output reg  [31:0]           ddr_output,

    input  wire [7:0]            cur_oc,
    input  wire [7:0]            cur_group,
    input  wire [7:0]            cur_row_pair,   // BUG 5 FIX: was [6:0], mismatched FSM/top.v
    input  wire [1:0]            fsm_wait_state,

    input  wire                  done,
    input  wire                  busy,
    output wire                  done_latch_out
);

    // Shadow registers for readback
    reg [31:0] reg_img_dim;
    reg [31:0] reg_out_dim;
    reg [31:0] reg_ch_cfg;
    reg [31:0] reg_ddr_input;
    reg [31:0] reg_ddr_weight;
    reg [31:0] reg_ddr_output;

    // AXI write handshake state
    // Accept AW and W independently; execute write when both received
    reg [ADDR_W-1:0] wr_addr_lat;
    reg [DATA_W-1:0] wr_data_lat;
    reg              wr_addr_got;   // AW channel received
    reg              wr_data_got;   // W channel received

    // Sticky done bit
    reg              done_latch;

    // -------------------------------------------------------
    // Write Address Channel - accept immediately
    // -------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b0;
            wr_addr_lat   <= {ADDR_W{1'b0}};
            wr_addr_got   <= 1'b0;
        end else begin
            // Accept AW as soon as it arrives (if not already holding one)
            if (s_axi_awvalid && !s_axi_awready && !wr_addr_got) begin
                s_axi_awready <= 1'b1;
                wr_addr_lat   <= s_axi_awaddr;
                wr_addr_got   <= 1'b1;
            end else begin
                s_axi_awready <= 1'b0;
                // Clear after write executes
                if (wr_addr_got && wr_data_got)
                    wr_addr_got <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------
    // Write Data Channel - accept immediately
    // -------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_wready <= 1'b0;
            wr_data_lat  <= {DATA_W{1'b0}};
            wr_data_got  <= 1'b0;
        end else begin
            // Accept W as soon as it arrives (if not already holding one)
            if (s_axi_wvalid && !s_axi_wready && !wr_data_got) begin
                s_axi_wready <= 1'b1;
                wr_data_lat  <= s_axi_wdata;
                wr_data_got  <= 1'b1;
            end else begin
                s_axi_wready <= 1'b0;
                // Clear after write executes
                if (wr_addr_got && wr_data_got)
                    wr_data_got <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------
    // Register Write + config outputs + done latch
    // Executes when BOTH wr_addr_got AND wr_data_got are set
    // ONE always block = no multi-driver conflicts
    // -------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start          <= 1'b0;
            done_latch     <= 1'b0;
            mode           <= 2'b00;
            input_h        <= 8'd0;
            input_w        <= 8'd0;
            output_h       <= 8'd0;
            output_w       <= 8'd0;
            num_in_ch      <= 8'd0;
            num_out_ch     <= 8'd0;
            ddr_input      <= 32'd0;
            ddr_weight     <= 32'd0;
            ddr_output     <= 32'd0;
            reg_img_dim    <= 32'd0;
            reg_out_dim    <= 32'd0;
            reg_ch_cfg     <= 32'd0;
            reg_ddr_input  <= 32'd0;
            reg_ddr_weight <= 32'd0;
            reg_ddr_output <= 32'd0;
        end else begin
            // start is 1-cycle pulse only; cleared every cycle by default
            start <= 1'b0;

            // Capture FSM done pulse into sticky latch
            if (done) done_latch <= 1'b1;

            // Execute register write when both AW and W have been received
            if (wr_addr_got && wr_data_got) begin
                case (wr_addr_lat[5:2])
                    4'd0: begin   // CTRL 0x00
                        mode <= wr_data_lat[2:1];
                        if (wr_data_lat[0]) begin
                            start      <= 1'b1;
                            done_latch <= 1'b0;
                        end
                    end
                    4'd2: begin   // IMG_DIM 0x08
                        reg_img_dim <= wr_data_lat;
                        input_h     <= wr_data_lat[7:0];
                        input_w     <= wr_data_lat[15:8];
                    end
                    4'd3: begin   // OUT_DIM 0x0C
                        reg_out_dim <= wr_data_lat;
                        output_h    <= wr_data_lat[7:0];
                        output_w    <= wr_data_lat[15:8];
                    end
                    4'd4: begin   // CH_CFG 0x10
                        reg_ch_cfg  <= wr_data_lat;
                        num_in_ch   <= wr_data_lat[7:0];
                        num_out_ch  <= wr_data_lat[15:8];
                    end
                    4'd5: begin   // DDR_INPUT 0x14
                        reg_ddr_input <= wr_data_lat;
                        ddr_input     <= wr_data_lat;
                    end
                    4'd6: begin   // DDR_WEIGHT 0x18
                        reg_ddr_weight <= wr_data_lat;
                        ddr_weight     <= wr_data_lat;
                    end
                    4'd7: begin   // DDR_OUTPUT 0x1C
                        reg_ddr_output <= wr_data_lat;
                        ddr_output     <= wr_data_lat;
                    end
                    default: ;
                endcase
            end
        end
    end

    // -------------------------------------------------------
    // Write Response - send BVALID when write executes
    // -------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;
        end else begin
            if (wr_addr_got && wr_data_got && !s_axi_bvalid) begin
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;
            end else if (s_axi_bready && s_axi_bvalid)
                s_axi_bvalid <= 1'b0;
        end
    end

    // -------------------------------------------------------
    // Read Channel
    // -------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= 32'd0;
            s_axi_rresp   <= 2'b00;
        end else begin
            if (s_axi_arvalid && !s_axi_arready) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid  <= 1'b1;
                s_axi_rresp   <= 2'b00;
                case (s_axi_araddr[5:2])
                    4'd0: s_axi_rdata <= {29'd0, mode, 1'b0};
                    4'd1: s_axi_rdata <= {30'd0, busy, done_latch};
                    4'd2: s_axi_rdata <= reg_img_dim;
                    4'd3: s_axi_rdata <= reg_out_dim;
                    4'd4: s_axi_rdata <= reg_ch_cfg;
                    4'd5: s_axi_rdata <= reg_ddr_input;
                    4'd6: s_axi_rdata <= reg_ddr_weight;
                    4'd7: s_axi_rdata <= reg_ddr_output;
                    4'd8: s_axi_rdata <= {6'b0, fsm_wait_state,
                                          cur_row_pair,         // BUG 5 FIX: full 8-bit
                                          cur_group, cur_oc};
                    4'd15: s_axi_rdata <= 32'hC0FFEE42; // canary: confirms new bitstream
                    default: s_axi_rdata <= 32'd0;
                endcase
            end else begin
                s_axi_arready <= 1'b0;
                if (s_axi_rready && s_axi_rvalid)
                    s_axi_rvalid <= 1'b0;
            end
        end
    end

    assign done_latch_out = done_latch;

endmodule