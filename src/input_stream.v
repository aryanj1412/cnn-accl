// ============================================================
// Module  : input_stream_controller
// Purpose : Receives AXI4-Stream from DMA MM2S
//           Routes incoming data to either:
//             weight_regfile (when FSM in LOAD_WEIGHTS state)
//             input_pingpong_buffer (when FSM in LOAD_INPUT state)
//
// FSM controls route_sel:
//   route_sel=0 ? weights  (144 values per group)
//   route_sel=1 ? input pixels (4?16?W values per group)
//
// Tracks write address internally, resets on load_start
// Signals load_done when TLAST received
// ============================================================

module input_stream_controller #(
    parameter MAX_W = 128
    )(
    input  wire                  clk,
    input  wire                  rst_n,

    // AXI4-Stream Slave (from DMA MM2S)
    input  wire [15:0]           s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output reg                   s_axis_tready,
    input  wire                  s_axis_tlast,

    // Control from FSM
    input  wire                  load_start,    // pulse to begin receiving
    input  wire                  route_sel,     // 0=weights, 1=pixels

    // Output to weight_regfile write port
    output reg                   wgt_wr_en,
    output reg  [7:0]            wgt_wr_addr,
    output wire signed [15:0]    wgt_wr_data,

    // Output to input_pingpong_buffer write port
    output reg                   pix_wr_en,
    output reg  [3:0]            pix_wr_ch,
    output reg  [8:0]            pix_wr_addr,
    output wire signed [15:0]    pix_wr_data,

    // Status
    output reg                   load_done      // pulse when TLAST received
);

    localparam IDLE = 1'b0;
    localparam RECV = 1'b1;
    localparam CH_STRIDE = 4*MAX_W;
    reg state;

    // Both outputs take same data from stream
    assign wgt_wr_data = s_axis_tdata;
    assign pix_wr_data = s_axis_tdata;

    // Pixel address decomposition:
    // Data arrives as: ch0_row0..ch0_row3, ch1_row0..ch1_row3, ...
    // pix_wr_ch   = pixel_count / (4*MAX_W)
    // pix_wr_addr = pixel_count % (4*MAX_W)
    // Using 9-bit counter for addr within bank (max 512)

    reg [12:0] pix_cnt;   // up to 4*16*128 = 8192 pixels
    reg [7:0]  wgt_cnt;   // up to 144 weights

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            s_axis_tready <= 1'b0;
            wgt_wr_en     <= 1'b0;
            pix_wr_en     <= 1'b0;
            load_done     <= 1'b0;
            wgt_cnt       <= 8'd0;
            pix_cnt       <= 13'd0;
            wgt_wr_addr   <= 8'd0;
            pix_wr_ch     <= 4'd0;
            pix_wr_addr   <= 9'd0;
        end else begin
            wgt_wr_en <= 1'b0;
            pix_wr_en <= 1'b0;
            load_done <= 1'b0;

            case (state)
                IDLE: begin
                    s_axis_tready <= 1'b0;
                    wgt_cnt       <= 8'd0;
                    pix_cnt       <= 13'd0;
                    if (load_start) begin
                        s_axis_tready <= 1'b1;
                        state         <= RECV;
                    end
                end

                RECV: begin
                    s_axis_tready <= 1'b1;

                    if (s_axis_tvalid && s_axis_tready) begin

                        if (route_sel == 1'b0) begin
                            // Route to weight regfile
                            wgt_wr_en   <= 1'b1;
                            wgt_wr_addr <= wgt_cnt;
                            wgt_cnt     <= wgt_cnt + 1;
                        end else begin
                            // Route to input pixel buffer
                            // ch = pix_cnt / 512, addr = pix_cnt % 512
                            pix_wr_en   <= 1'b1;
                            pix_wr_ch   <= pix_cnt[12:9];  // upper 4 bits = channel (0..15)
                            pix_wr_addr <= pix_cnt[8:0];  // lower 9 bits = addr in bank (0..511)
                            pix_cnt     <= pix_cnt + 1;
                        end

                        if (s_axis_tlast) begin
                            s_axis_tready <= 1'b0;
                            load_done     <= 1'b1;
                            state         <= IDLE;
                        end
                    end
                end
            endcase
        end
    end

endmodule