`timescale 1 ps / 1 ps

module system_top (
    inout  [14:0] DDR_addr,
    inout  [2:0]  DDR_ba,
    inout         DDR_cas_n,
    inout         DDR_ck_n,
    inout         DDR_ck_p,
    inout         DDR_cke,
    inout         DDR_cs_n,
    inout  [3:0]  DDR_dm,
    inout  [31:0] DDR_dq,
    inout  [3:0]  DDR_dqs_n,
    inout  [3:0]  DDR_dqs_p,
    inout         DDR_odt,
    inout         DDR_ras_n,
    inout         DDR_reset_n,
    inout         DDR_we_n,
    inout         FIXED_IO_ddr_vrn,
    inout         FIXED_IO_ddr_vrp,
    inout  [53:0] FIXED_IO_mio,
    inout         FIXED_IO_ps_clk,
    inout         FIXED_IO_ps_porb,
    inout         FIXED_IO_ps_srstb
);

    // -------------------------------------------------------
    // Internal wires between block design and accelerator
    // -------------------------------------------------------
    wire        clk;
    wire        rst_n;

    // AXI-Lite: block design ? accelerator
    wire [31:0] M01_AXI_0_araddr;
    wire [2:0]  M01_AXI_0_arprot;
    wire [0:0]  M01_AXI_0_arready;
    wire [0:0]  M01_AXI_0_arvalid;
    wire [31:0] M01_AXI_0_awaddr;
    wire [2:0]  M01_AXI_0_awprot;
    wire [0:0]  M01_AXI_0_awready;
    wire [0:0]  M01_AXI_0_awvalid;
    wire [0:0]  M01_AXI_0_bready;
    wire [1:0]  M01_AXI_0_bresp;
    wire [0:0]  M01_AXI_0_bvalid;
    wire [31:0] M01_AXI_0_rdata;
    wire [0:0]  M01_AXI_0_rready;
    wire [1:0]  M01_AXI_0_rresp;
    wire [0:0]  M01_AXI_0_rvalid;
    wire [31:0] M01_AXI_0_wdata;
    wire [0:0]  M01_AXI_0_wready;
    wire [3:0]  M01_AXI_0_wstrb;
    wire [0:0]  M01_AXI_0_wvalid;

    // AXI4-Stream MM2S: DMA ? accelerator
    wire [15:0] M_AXIS_MM2S_0_tdata;
    wire [1:0]  M_AXIS_MM2S_0_tkeep;
    wire        M_AXIS_MM2S_0_tlast;
    wire        M_AXIS_MM2S_0_tready;
    wire        M_AXIS_MM2S_0_tvalid;

    // AXI4-Stream S2MM: accelerator ? DMA
    wire [15:0] S_AXIS_S2MM_0_tdata;
    wire [1:0]  S_AXIS_S2MM_0_tkeep;
    wire        S_AXIS_S2MM_0_tlast;
    wire        S_AXIS_S2MM_0_tready;
    wire        S_AXIS_S2MM_0_tvalid;

    // -------------------------------------------------------
    // Block design: PS + DMA + interconnect (no accelerator)
    // -------------------------------------------------------
    design_1_wrapper u_bd (
        .DDR_addr               (DDR_addr),
        .DDR_ba                 (DDR_ba),
        .DDR_cas_n              (DDR_cas_n),
        .DDR_ck_n               (DDR_ck_n),
        .DDR_ck_p               (DDR_ck_p),
        .DDR_cke                (DDR_cke),
        .DDR_cs_n               (DDR_cs_n),
        .DDR_dm                 (DDR_dm),
        .DDR_dq                 (DDR_dq),
        .DDR_dqs_n              (DDR_dqs_n),
        .DDR_dqs_p              (DDR_dqs_p),
        .DDR_odt                (DDR_odt),
        .DDR_ras_n              (DDR_ras_n),
        .DDR_reset_n            (DDR_reset_n),
        .DDR_we_n               (DDR_we_n),
        .FIXED_IO_ddr_vrn       (FIXED_IO_ddr_vrn),
        .FIXED_IO_ddr_vrp       (FIXED_IO_ddr_vrp),
        .FIXED_IO_mio           (FIXED_IO_mio),
        .FIXED_IO_ps_clk        (FIXED_IO_ps_clk),
        .FIXED_IO_ps_porb       (FIXED_IO_ps_porb),
        .FIXED_IO_ps_srstb      (FIXED_IO_ps_srstb),
        .clk                    (clk),
        .rst_n                  (rst_n),
        // AXI-Lite to accelerator
        .M01_AXI_0_araddr       (M01_AXI_0_araddr),
        .M01_AXI_0_arprot       (M01_AXI_0_arprot),
        .M01_AXI_0_arready      (M01_AXI_0_arready),
        .M01_AXI_0_arvalid      (M01_AXI_0_arvalid),
        .M01_AXI_0_awaddr       (M01_AXI_0_awaddr),
        .M01_AXI_0_awprot       (M01_AXI_0_awprot),
        .M01_AXI_0_awready      (M01_AXI_0_awready),
        .M01_AXI_0_awvalid      (M01_AXI_0_awvalid),
        .M01_AXI_0_bready       (M01_AXI_0_bready),
        .M01_AXI_0_bresp        (M01_AXI_0_bresp),
        .M01_AXI_0_bvalid       (M01_AXI_0_bvalid),
        .M01_AXI_0_rdata        (M01_AXI_0_rdata),
        .M01_AXI_0_rready       (M01_AXI_0_rready),
        .M01_AXI_0_rresp        (M01_AXI_0_rresp),
        .M01_AXI_0_rvalid       (M01_AXI_0_rvalid),
        .M01_AXI_0_wdata        (M01_AXI_0_wdata),
        .M01_AXI_0_wready       (M01_AXI_0_wready),
        .M01_AXI_0_wstrb        (M01_AXI_0_wstrb),
        .M01_AXI_0_wvalid       (M01_AXI_0_wvalid),
        // MM2S stream: DMA ? accelerator
        .M_AXIS_MM2S_0_tdata    (M_AXIS_MM2S_0_tdata),
        .M_AXIS_MM2S_0_tkeep    (M_AXIS_MM2S_0_tkeep),
        .M_AXIS_MM2S_0_tlast    (M_AXIS_MM2S_0_tlast),
        .M_AXIS_MM2S_0_tready   (M_AXIS_MM2S_0_tready),
        .M_AXIS_MM2S_0_tvalid   (M_AXIS_MM2S_0_tvalid),
        // S2MM stream: accelerator ? DMA
        .S_AXIS_S2MM_0_tdata    (S_AXIS_S2MM_0_tdata),
        .S_AXIS_S2MM_0_tkeep    (S_AXIS_S2MM_0_tkeep),
        .S_AXIS_S2MM_0_tlast    (S_AXIS_S2MM_0_tlast),
        .S_AXIS_S2MM_0_tready   (S_AXIS_S2MM_0_tready),
        .S_AXIS_S2MM_0_tvalid   (S_AXIS_S2MM_0_tvalid)
    );

    // -------------------------------------------------------
    // CNN Accelerator: pure RTL, synthesized from source
    // AXI-Lite address is [5:0] in your top.v
    // M01_AXI_0 uses [31:0] addresses - take lower 6 bits
    // -------------------------------------------------------
    cnn_accelerator_top u_cnn (
        .clk                (clk),
        .rst_n              (rst_n),
        // AXI-Lite slave
        .s_axi_awaddr       (M01_AXI_0_awaddr[5:0]),
        .s_axi_awvalid      (M01_AXI_0_awvalid[0]),
        .s_axi_awready      (M01_AXI_0_awready[0]),
        .s_axi_wdata        (M01_AXI_0_wdata),
        .s_axi_wstrb        (M01_AXI_0_wstrb),
        .s_axi_wvalid       (M01_AXI_0_wvalid[0]),
        .s_axi_wready       (M01_AXI_0_wready[0]),
        .s_axi_bresp        (M01_AXI_0_bresp),
        .s_axi_bvalid       (M01_AXI_0_bvalid[0]),
        .s_axi_bready       (M01_AXI_0_bready[0]),
        .s_axi_araddr       (M01_AXI_0_araddr[5:0]),
        .s_axi_arvalid      (M01_AXI_0_arvalid[0]),
        .s_axi_arready      (M01_AXI_0_arready[0]),
        .s_axi_rdata        (M01_AXI_0_rdata),
        .s_axi_rresp        (M01_AXI_0_rresp),
        .s_axi_rvalid       (M01_AXI_0_rvalid[0]),
        .s_axi_rready       (M01_AXI_0_rready[0]),
        // AXI4-Stream slave (from DMA MM2S)
        .s_axis_tdata       (M_AXIS_MM2S_0_tdata),
        .s_axis_tvalid      (M_AXIS_MM2S_0_tvalid),
        .s_axis_tready      (M_AXIS_MM2S_0_tready),
        .s_axis_tlast       (M_AXIS_MM2S_0_tlast),
        // AXI4-Stream master (to DMA S2MM)
        .m_axis_tdata       (S_AXIS_S2MM_0_tdata),
        .m_axis_tvalid      (S_AXIS_S2MM_0_tvalid),
        .m_axis_tready      (S_AXIS_S2MM_0_tready),
        .m_axis_tlast       (S_AXIS_S2MM_0_tlast)
    );

    // tkeep is not used by accelerator - tie S2MM tkeep high
    assign S_AXIS_S2MM_0_tkeep = 2'b11;

endmodule
