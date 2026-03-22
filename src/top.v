// ============================================================
// Module  : cnn_accelerator_top
// Purpose : Top-level CNN Accelerator
//
// Mode encoding (from PS via AXI-Lite CTRL[2:1]):
//   00 = conv only   : quantizer ? output_buffer
//   01 = conv + relu : quantizer ? relu ? output_buffer
//   10 = maxpool     : quantizer ? relu ? maxpool ? output_buffer
//
// Note on maxpool mode:
//   ReLU always applied before maxpool
//   because in your CNN: Conv ? ReLU ? MaxPool
//   MaxPool layer receives already-relu'd feature map
//   So in hardware maxpool mode does NOT re-apply relu
//   (relu was applied in previous conv+relu pass)
//   Maxpool just pools the incoming feature map as-is
// ============================================================

module cnn_accelerator_top (
    input  wire        clk,
    input  wire        rst_n,

    // AXI4-Lite Slave
    input  wire [5:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [5:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,

    // AXI4-Stream Slave (from DMA MM2S)
    input  wire [15:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    // AXI4-Stream Master (to DMA S2MM)
    output wire [15:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast
);

    // -------------------------------------------------
    // Config wires
    // -------------------------------------------------
    wire        start;
    wire [1:0]  mode;      // 00=conv 01=conv+relu 10=pool
    wire [7:0]  input_h,  input_w;
    wire [7:0]  output_h, output_w;
    wire [7:0]  num_in_ch, num_out_ch;
    wire [31:0] ddr_input, ddr_weight, ddr_output;
    wire        done, busy;
    wire [7:0]  cur_oc, cur_group;
    wire [7:0]  cur_row_pair;

    // -------------------------------------------------
    // FSM control wires
    // -------------------------------------------------
    wire        load_start, route_sel, load_done;
    wire [3:0]  wgt_rd_tap;
    wire        in_buf_sel;
    wire        pe_clr, pe_en;
    wire        acc_clr_en;
    wire [7:0]  acc_clr_col;
    wire        acc_wr_en, acc_row_sel;
    wire [7:0]  acc_col;
    wire        out_wr_en;
    wire [13:0] out_wr_addr;
    wire        stream_start, stream_done;
    wire [13:0] total_pixels;
    wire [8:0]  bram_rd_addr;
    wire        quant_en;
    wire [7:0]  quant_col;
    wire        quant_row_sel;
    wire [1:0]  fsm_wait_state;
    wire        done_latch_out;

    // -------------------------------------------------
    // Data path wires
    // -------------------------------------------------
    wire        wgt_wr_en;
    wire [7:0]  wgt_wr_addr;
    wire [15:0] wgt_wr_data;
    wire [255:0] wgt_rd_flat;
    wire        pix_wr_en;
    wire [3:0]  pix_wr_ch;
    wire [8:0]  pix_wr_addr;
    wire [15:0] pix_wr_data;
    wire [255:0] bram_rd_flat;
    wire [511:0]       pe_acc_flat;
    wire signed [31:0] ch_sum;
    wire signed [31:0] acc_rd_row0, acc_rd_row1;

    // -------------------------------------------------
    // Quantizer outputs
    // -------------------------------------------------
    wire signed [15:0] q_out_row0, q_out_row1;

    // -------------------------------------------------
    // ReLU outputs
    // -------------------------------------------------
    wire signed [15:0] relu_out_row0, relu_out_row1;

    // -------------------------------------------------
    // After ReLU mux:
    //   mode=00 (conv only) : bypass relu
    //   mode=01 (conv+relu) : apply relu
    //   mode=10 (maxpool)   : apply relu before pool
    //                         (relu always before pool)
    // -------------------------------------------------
    wire apply_relu = (mode == 2'b01) || (mode == 2'b10);

    wire signed [15:0] post_relu_row0, post_relu_row1;
    assign post_relu_row0 = apply_relu ? relu_out_row0 : q_out_row0;
    assign post_relu_row1 = apply_relu ? relu_out_row1 : q_out_row1;

    // -------------------------------------------------
    // MaxPool wires
    // -------------------------------------------------
    wire signed [15:0] pool_out;
    wire               pool_valid;

    // -------------------------------------------------
    // Final output pixel selection:
    //   mode=00 or 01: write post_relu pixel (conv or conv+relu)
    //   mode=10:       write pooled pixel when pool_valid
    // -------------------------------------------------
    wire is_pool = (mode == 2'b10);

    wire out_wr_en_final;
    wire signed [15:0] out_pixel;

    assign out_wr_en_final = is_pool ? pool_valid : out_wr_en;
// BUG 2 FIX: All three output pipeline registers now share the same
// rst_n async reset so they can never hold stale/X values after reset.
reg signed [15:0] out_pixel_reg;
reg [13:0] out_wr_addr_d;
reg out_wr_en_d;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        out_wr_en_d <= 1'b0;
    else
        out_wr_en_d <= out_wr_en_final;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        out_pixel_reg <= 16'sh0;
    else if (is_pool)
        out_pixel_reg <= pool_out;
    else
        out_pixel_reg <= (quant_row_sel ? post_relu_row1 : post_relu_row0);
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        out_wr_addr_d <= 14'd0;
    else
        out_wr_addr_d <= out_wr_addr;
end

    // -------------------------------------------------
    // Output buffer wires
    // -------------------------------------------------
    wire [13:0]        out_buf_rd_addr;
    wire signed [15:0] out_buf_rd_data;

    // -------------------------------------------------
    // Module Instantiations
    // -------------------------------------------------

    axi_lite_slave u_axi_lite (
        .clk            (clk),
        .rst_n          (rst_n),
        .s_axi_awaddr   (s_axi_awaddr),
        .s_axi_awvalid  (s_axi_awvalid),
        .s_axi_awready  (s_axi_awready),
        .s_axi_wdata    (s_axi_wdata),
        .s_axi_wstrb    (s_axi_wstrb),
        .s_axi_wvalid   (s_axi_wvalid),
        .s_axi_wready   (s_axi_wready),
        .s_axi_bresp    (s_axi_bresp),
        .s_axi_bvalid   (s_axi_bvalid),
        .s_axi_bready   (s_axi_bready),
        .s_axi_araddr   (s_axi_araddr),
        .s_axi_arvalid  (s_axi_arvalid),
        .s_axi_arready  (s_axi_arready),
        .s_axi_rdata    (s_axi_rdata),
        .s_axi_rresp    (s_axi_rresp),
        .s_axi_rvalid   (s_axi_rvalid),
        .s_axi_rready   (s_axi_rready),
        .start          (start),
        .mode           (mode),
        .input_h        (input_h),
        .input_w        (input_w),
        .output_h       (output_h),
        .output_w       (output_w),
        .num_in_ch      (num_in_ch),
        .num_out_ch     (num_out_ch),
        .ddr_input      (ddr_input),
        .ddr_weight     (ddr_weight),
        .ddr_output     (ddr_output),
        .cur_oc         (cur_oc),
        .cur_group      (cur_group),
        .cur_row_pair   (cur_row_pair),
        .fsm_wait_state (fsm_wait_state),
        .done           (done),
        .busy           (busy),
        .done_latch_out (done_latch_out)
    );

    fsm_controller u_fsm (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .mode           (mode[1]),   // bit1=1 for pool mode (2'b10)
        .input_h        (input_h),
        .input_w        (input_w),
        .output_h       (output_h),
        .output_w       (output_w),
        .num_in_ch      (num_in_ch),
        .num_out_ch     (num_out_ch),
        .load_start     (load_start),
        .route_sel      (route_sel),
        .load_done      (load_done),
        .wgt_rd_tap     (wgt_rd_tap),
        .in_buf_sel     (in_buf_sel),
        .pe_clr         (pe_clr),
        .pe_en          (pe_en),
        .acc_clr_en     (acc_clr_en),
        .acc_clr_col    (acc_clr_col),
        .acc_wr_en      (acc_wr_en),
        .acc_row_sel    (acc_row_sel),
        .acc_col        (acc_col),
        .out_wr_en      (out_wr_en),
        .out_wr_addr    (out_wr_addr),
        .stream_start   (stream_start),
        .total_pixels   (total_pixels),
        .stream_done    (stream_done),
        .bram_rd_addr   (bram_rd_addr),
        .quant_en       (quant_en),
        .quant_col      (quant_col),
        .quant_row_sel  (quant_row_sel),
        .cur_oc         (cur_oc),
        .cur_group      (cur_group),
        .cur_row_pair   (cur_row_pair),
        .fsm_wait_state (fsm_wait_state),
        .busy           (busy),
        .done           (done),
        .done_latch_in  (done_latch_out)
    );

    input_stream_controller u_isc (
        .clk            (clk),
        .rst_n          (rst_n),
        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tready  (s_axis_tready),
        .s_axis_tlast   (s_axis_tlast),
        .load_start     (load_start),
        .route_sel      (route_sel),
        .wgt_wr_en      (wgt_wr_en),
        .wgt_wr_addr    (wgt_wr_addr),
        .wgt_wr_data    (wgt_wr_data),
        .pix_wr_en      (pix_wr_en),
        .pix_wr_ch      (pix_wr_ch),
        .pix_wr_addr    (pix_wr_addr),
        .pix_wr_data    (pix_wr_data),
        .load_done      (load_done)
    );

    weight_regfile u_wgt_rf (
        .clk        (clk),
        .wr_en      (wgt_wr_en),
        .wr_addr    (wgt_wr_addr),
        .wr_data    (wgt_wr_data),
        .rd_tap     (wgt_rd_tap),
        .rd_flat    (wgt_rd_flat)
    );

    input_pingpong_buffer u_in_buf (
        .clk        (clk),
        .sel        (in_buf_sel),
        .rst_n(rst_n),
        .wr_en      (pix_wr_en),
        .wr_ch      (pix_wr_ch),
        .wr_addr    (pix_wr_addr),
        .wr_data    (pix_wr_data),
        .rd_addr    (bram_rd_addr),
        .rd_flat    (bram_rd_flat)
    );

    compute_core u_compute (
        .clk         (clk),
        .rst_n       (rst_n),
        .clr         (pe_clr),
        .en          (pe_en),
        .pixel_flat  (bram_rd_flat),
        .weight_flat (wgt_rd_flat),
        .acc_flat    (pe_acc_flat)
    );

    // BUG 1 FIX: channel_summer was incorrectly commented out and replaced
    // with a hardcoded debug constant (0x00020000). This discarded ALL real
    // convolution results. Restored the real channel_summer instantiation.
    channel_summer u_summer (
        .acc_in  (pe_acc_flat),
        .sum_out (ch_sum)
    );

    acc_row_buffer u_acc_row (
        .clk         (clk),
        .rst_n       (rst_n),
        .clr_en      (acc_clr_en),
        .clr_col     (acc_clr_col),
        .wr_en       (acc_wr_en),
        .wr_row_sel  (acc_row_sel),
        .wr_col      (acc_col),
        .wr_data     (ch_sum),
        .rd_col      (quant_col),
        .rd_row0     (acc_rd_row0),
        .rd_row1     (acc_rd_row1)
    );

    // Quantize: Q12.18 ? Q6.9
quantizer u_quant_r0 (.acc_in(acc_rd_row0), .q_out(q_out_row0));
quantizer u_quant_r1 (.acc_in(acc_rd_row1), .q_out(q_out_row1));

    // ReLU: applied after quantization
    relu_unit u_relu_r0 (.data_in(q_out_row0), .data_out(relu_out_row0));
    relu_unit u_relu_r1 (.data_in(q_out_row1), .data_out(relu_out_row1));

    // MaxPool: receives post-relu pixels
    maxpool_unit u_pool (
        .clk        (clk),
        .rst_n      (rst_n),
        .en         (quant_en),
        .pixel_r0   (post_relu_row0),
        .pixel_r1   (post_relu_row1),
        .col_idx    (quant_col),
        .pool_out   (pool_out),
        .pool_valid (pool_valid)
    );

    output_buffer u_out_buf (
        .clk      (clk),
        .wr_en    (out_wr_en_d),
        .wr_addr  (out_wr_addr_d),
        .wr_data  (out_pixel_reg),
        .rd_addr  (out_buf_rd_addr),
        .rd_data  (out_buf_rd_data)
    );

    output_stream_controller u_osc (
        .clk           (clk),
        .rst_n         (rst_n),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready),
        .m_axis_tlast  (m_axis_tlast),
        .buf_rd_addr   (out_buf_rd_addr),
        .buf_rd_data   (out_buf_rd_data),
        .stream_start  (stream_start),
        .total_pixels  (total_pixels),
        .stream_done   (stream_done)
    );

endmodule