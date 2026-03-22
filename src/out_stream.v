// ============================================================
// Module  : output_stream_controller
// Purpose : Reads output buffer sequentially
//           Drives AXI4-Stream Master to DMA S2MM
//           Sends complete output channel pixel by pixel
//
// Operation:
//   FSM asserts stream_start with total pixel count
//   Controller reads output_buffer sequentially
//   Sends each pixel on AXI-S with TVALID
//   Asserts TLAST on final pixel
//   Asserts stream_done when transfer complete
//
// Accounts for 1 cycle BRAM read latency
// ============================================================

module output_stream_controller (
    input  wire                  clk,
    input  wire                  rst_n,

    // AXI4-Stream Master (to DMA S2MM)
    output reg  [15:0]           m_axis_tdata,
    output reg                   m_axis_tvalid,
    input  wire                  m_axis_tready,
    output reg                   m_axis_tlast,

    // Output buffer read port
    output reg  [13:0]           buf_rd_addr,
    input  wire signed [15:0]    buf_rd_data,

    // Control from FSM
    input  wire                  stream_start,
    input  wire [13:0]           total_pixels,  // output_h * output_w
    output reg                   stream_done
);

    localparam IDLE    = 2'd0;
    localparam FETCH   = 2'd1;
    localparam STREAM  = 2'd2;

    reg [1:0]  state;
    reg [13:0] rd_idx;
    reg [13:0] total_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
            m_axis_tdata  <= 16'd0;
            buf_rd_addr   <= 14'd0;
            rd_idx        <= 14'd0;
            stream_done   <= 1'b0;
        end else begin
            stream_done <= 1'b0;

            case (state)
                // -------------------------------------------
                IDLE: begin
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast  <= 1'b0;
                    rd_idx        <= 14'd0;
                    if (stream_start) begin
                        total_reg   <= total_pixels;
                        buf_rd_addr <= 14'd0;
                        state       <= FETCH;
                    end
                end

                // -------------------------------------------
                // Issue read address, wait 1 cycle for BRAM
                FETCH: begin
                    buf_rd_addr <= rd_idx;
                    state       <= STREAM;
                end

                // -------------------------------------------
                // Data from BRAM available, send on AXI-S
                STREAM: begin
                    if (!m_axis_tvalid || m_axis_tready) begin
                        m_axis_tdata  <= buf_rd_data;
                        m_axis_tvalid <= 1'b1;
                        m_axis_tlast  <= (rd_idx == total_reg - 1);

                        if (rd_idx == total_reg - 1) begin
                            // Last pixel
                            stream_done <= 1'b1;
                            state       <= IDLE;
                        end else begin
                            rd_idx      <= rd_idx + 1;
                            buf_rd_addr <= rd_idx + 1;
                            state       <= STREAM;
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule