// ============================================================
// Module  : fsm_controller
//
// Timing note - BRAM latency:
//   Both input_pingpong_buffer and weight_regfile have
//   1 cycle registered read latency
//
//   tap_cnt sequences 0..9 (10 cycles per output pixel):
//     tap 0..8: send BRAM read address + weight read address
//     tap 1..9: data arrives from BRAM + weight BRAM
//               PE MAC fires on data arrival
//
//   pe_clr fires on tap_cnt=0 (before first data arrives)
//   pe_en  fires on tap_cnt=1..9 (when data is valid)
//   acc_wr_en fires on tap_cnt=9 (after last MAC)
//
//   Effective: 10 cycles per output pixel
//   (1 addr cycle + 9 MAC cycles)
// ============================================================

module fsm_controller #(
    parameter MAX_W = 128
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    input  wire        mode,
    input  wire [7:0]  input_h,
    input  wire [7:0]  input_w,
    input  wire [7:0]  output_h,
    input  wire [7:0]  output_w,
    input  wire [7:0]  num_in_ch,
    input  wire [7:0]  num_out_ch,

    output reg         load_start,
    output reg         route_sel,
    input  wire        load_done,

    output reg  [3:0]  wgt_rd_tap,

    output reg         in_buf_sel,

    output reg         pe_clr,
    output reg         pe_en,

    output reg         acc_clr_en,
    output reg  [7:0]  acc_clr_col,
    output reg         acc_wr_en,
    output reg         acc_row_sel,
    output reg  [7:0]  acc_col,

    output reg         out_wr_en,
    output reg  [13:0] out_wr_addr,

    output reg         stream_start,
    output reg  [13:0] total_pixels,
    input  wire        stream_done,

    output reg  [8:0]  bram_rd_addr,

    output reg         quant_en,
    output reg  [7:0]  quant_col,
    output reg         quant_row_sel,

    output reg  [7:0]  cur_oc,
    output reg  [7:0]  cur_group,
    output reg  [7:0]  cur_row_pair,

    output reg         busy,
    output reg         done,

    // done_latch from AXI slave: =1 after layer complete, cleared on new start
    // Used to distinguish intentional new-layer start from spurious AXI replay
    input  wire        done_latch_in,

    // PS handshake: tells PS what FSM is waiting for
    // 00 = not waiting for DMA
    // 01 = waiting for weights (send 144 weights then assert TLAST)
    // 10 = waiting for pixels  (send 4xchxW pixels then assert TLAST)
    output reg  [1:0]  fsm_wait_state
);

    localparam IDLE          = 4'd0;
    localparam LOAD_WEIGHTS  = 4'd1;
    localparam WAIT_WEIGHTS  = 4'd2;
    localparam LOAD_INPUT    = 4'd3;
    localparam WAIT_INPUT    = 4'd4;
    localparam CLEAR_ACC     = 4'd5;
    localparam COMPUTE       = 4'd6;
    localparam NEXT_GROUP    = 4'd7;
    localparam QUANTIZE      = 4'd8;
    localparam WRITE_OUT     = 4'd9;
    localparam NEXT_ROW_PAIR = 4'd10;
    localparam STREAM_OUTPUT = 4'd11;
    localparam WAIT_STREAM   = 4'd12;
    localparam NEXT_OC       = 4'd13;
    localparam ALL_DONE      = 4'd14;

    reg [3:0]  state;
    reg        run_armed;   // set on start, cleared on ALL_DONE; blocks spurious restarts
    // tap_cnt 0..10: 0=addr/clr, 1..9=MAC, 10=write acc (PE output valid)
    reg [3:0]  tap_cnt;
    reg [7:0]  cur_col;       // 8-bit: needs to hold output_w*2 up to 252
    reg [7:0]  quant_cnt;
    reg [13:0] out_wr_cnt;
    reg [7:0]  num_groups;
    reg [7:0]  num_row_pairs;

    // tap index sent to BRAM is tap_cnt (0..8), only used at tap_cnt=0..8
    // data arrives 1 cycle later: PE MACs at tap_cnt=1..9
    // at tap_cnt=10: PE register holds complete sum, acc_wr_en fires

    // Tap decomposition for BRAM address
    // tap_cnt=0 ? addr for tap0, tap_cnt=1 ? addr for tap1...
    wire [1:0] tap_row_off = tap_cnt / 3;
    wire [1:0] tap_col_off = tap_cnt % 3;
    wire [7:0] row_base;
    assign row_base = cur_row_pair * 2;
    wire out_row_sel = (cur_col >= output_w) ? 1'b1 : 1'b0;
    wire [7:0] col_idx = out_row_sel ?
                         cur_col - output_w :
                         cur_col;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= IDLE;
            run_armed    <= 1'b0;
            cur_oc       <= 8'd0;
            cur_group    <= 8'd0;
            cur_row_pair <= 8'd0;
            cur_col      <= 8'd0;
            acc_row_sel <= 1'b0;
            acc_col     <= 8'd0;
            tap_cnt      <= 4'd0;
            quant_cnt    <= 8'd0;
            out_wr_cnt   <= 14'd0;
            busy         <= 1'b0;
            done         <= 1'b0;
            pe_clr       <= 1'b0;
            pe_en        <= 1'b0;
            acc_clr_en   <= 1'b0;
            acc_clr_col  <= 8'd0;
            acc_wr_en    <= 1'b0;
            load_start   <= 1'b0;
            stream_start <= 1'b0;
            quant_en     <= 1'b0;
            out_wr_en    <= 1'b0;
            in_buf_sel   <= 1'b0;
            wgt_rd_tap   <= 4'd0;
            bram_rd_addr <= 9'd0;
            fsm_wait_state <= 2'b00;
        end else begin
            pe_clr       <= 1'b0;
            pe_en        <= 1'b0;
            acc_clr_en   <= 1'b0;
            acc_wr_en    <= 1'b0;
            load_start   <= 1'b0;
            stream_start <= 1'b0;
            quant_en     <= 1'b0;
            out_wr_en    <= 1'b0;
            done         <= 1'b0;
            fsm_wait_state <= 2'b00;

            case (state)

                IDLE: begin
                    busy         <= 1'b0;
                    cur_oc       <= 8'd0;
                    cur_group    <= 8'd0;
                    cur_row_pair <= 8'd0;
                    if (start) begin
                        // BUG 8 FIX: Guard was described in comments but never coded.
                        // Block spurious mid-run AXI replays:
                        //   Allow start if: not armed yet (first layer), OR
                        //                   armed but done_latch=1 (PS intentionally
                        //                   starting a new layer after reading done).
                        //   Block if:       armed and done_latch=0 (spurious replay
                        //                   while a layer is still computing).
                        if (!run_armed || done_latch_in) begin
                            run_armed     <= 1'b1;
                            busy          <= 1'b1;
                            num_groups    <= (num_in_ch + 8'd15) >> 4;
                            // Pool: one output row per 2-row input pair ? num_row_pairs = output_h
                            // Conv: two output rows per pair ? num_row_pairs = (output_h+1)/2
                            num_row_pairs <= mode ? output_h : ((output_h + 8'd1) >> 1);
                            state         <= LOAD_WEIGHTS;
                        end
                        // else: run_armed=1, done_latch_in=0 ? spurious replay, ignore
                    end
                end

                LOAD_WEIGHTS: begin
                    route_sel      <= 1'b0;
                    load_start     <= 1'b1;
                    fsm_wait_state <= 2'b01;  // waiting for weights
                    state          <= WAIT_WEIGHTS;
                end

                WAIT_WEIGHTS: begin
                    fsm_wait_state <= 2'b01;  // still waiting
                    load_start     <= 1'b0;
                    if (load_done) begin
                        fsm_wait_state <= 2'b00;
                        state          <= LOAD_INPUT;
                    end
                end

                LOAD_INPUT: begin
                    route_sel      <= 1'b1;
                    load_start     <= 1'b1;
                    fsm_wait_state <= 2'b10;  // waiting for pixels
                    state          <= WAIT_INPUT;
                end

                WAIT_INPUT: begin
                    fsm_wait_state <= 2'b10;  // still waiting
                    load_start     <= 1'b0;
                    if (load_done) begin
                        fsm_wait_state <= 2'b00;
                        in_buf_sel     <= ~in_buf_sel;
                        if (cur_group == 8'd0) begin
                            // Clear acc_row at start of new row pair
                            acc_clr_col <= 8'd0;
                            state       <= CLEAR_ACC;
                        end else begin
                            cur_col <= 8'd0;
                            tap_cnt <= 4'd0;
                            state   <= COMPUTE;
                        end
                    end
                end

                // Sequential clear: output_w cycles
                CLEAR_ACC: begin
                    acc_clr_en  <= 1'b1;
                    if (acc_clr_col < output_w - 1) begin
                        acc_clr_col <= acc_clr_col + 1;
                    end else begin
                        acc_clr_col <= 8'd0;
                        acc_clr_en  <= 1'b0;
                        cur_col     <= 8'd0;
                        tap_cnt     <= 4'd0;
                        state       <= COMPUTE;
                    end
                end

                // ----------------------------------------
                // COMPUTE: 10 cycles per output pixel
                //
                // tap_cnt=0:
                //   Send BRAM addr for tap0
                //   Send weight addr for tap0
                //   Assert pe_clr (clear acc for new pixel)
                //   pe_en=0 (data not yet arrived)
                //
                // tap_cnt=1..8:
                //   Send BRAM addr for tap(N)
                //   Send weight addr for tap(N)
                //   pe_en=1: data for tap(N-1) arrives, MAC fires
                //
                // tap_cnt=9:
                //   No new addr needed (tap8 was last)
                //   pe_en=1: data for tap8 arrives, last MAC fires
                //   acc_wr_en=1: latch sum to acc_row
                //   Reset tap_cnt=0, move to next col
                // ----------------------------------------
                COMPUTE: begin
                    // 12-cycle sequence per output pixel:
                    // tap0:  pe_clr=1, pe_en=0   clear PE, send bram addr for tap0
                    // tap1:  pe_clr=0, pe_en=0   wait for BRAM data (1-cycle latency)
                    //        send bram addr for tap1
                    // tap2..9: pe_en=1, wgt=tap-2   MAC taps 0..7
                    //          send bram addr for next tap
                    // tap10: pe_en=1, wgt=8   MAC tap8 (last multiply)
                    // tap11: pe_en=0, acc_wr_en=1   PE reg holds complete sum, WRITE

                    if (tap_cnt == 4'd0) begin
                        pe_clr       <= 1'b1;
                        pe_en        <= 1'b0;
                        acc_wr_en    <= 1'b0;
                        wgt_rd_tap   <= 4'd0;
                        bram_rd_addr <= ({7'b0,out_row_sel} + {7'b0,tap_row_off}) * {1'b0,input_w} +
                                        {1'b0,col_idx} + {7'b0,tap_col_off};
                        tap_cnt      <= 4'd1;

                    end else if (tap_cnt == 4'd1) begin
                        // Wait cycle: BRAM data for tap0 not yet arrived
                        // Re-send tap0 address so tap0 data is ready at tap2
                        pe_clr       <= 1'b0;
                        pe_en        <= 1'b0;
                        wgt_rd_tap   <= 4'd0;
                        bram_rd_addr <= {8'b0,out_row_sel} * {1'b0,input_w} +
                                        {1'b0,col_idx};   // tap0: row_off=0, col_off=0
                        tap_cnt      <= 4'd2;

                    end else if (tap_cnt < 4'd10) begin
                        // tap2..9: MAC kernel_tap(cnt-2), send addr for kernel_tap(cnt-1)
                        // addr uses (tap_cnt-1) as the kernel index
                        pe_clr     <= 1'b0;
                        pe_en      <= 1'b1;
                        wgt_rd_tap <= tap_cnt - 2;
                        // Compute addr for kernel_tap = tap_cnt-1
                        // (tap_cnt-1)/3 and (tap_cnt-1)%3 via case:
                        case (tap_cnt)
                            4'd2: bram_rd_addr <= {8'b0,out_row_sel} * {1'b0,input_w} + {1'b0,col_idx} + 9'd1; // tap1
                            4'd3: bram_rd_addr <= {8'b0,out_row_sel} * {1'b0,input_w} + {1'b0,col_idx} + 9'd2; // tap2
                            4'd4: bram_rd_addr <= ({8'b0,out_row_sel}+9'd1) * {1'b0,input_w} + {1'b0,col_idx};  // tap3
                            4'd5: bram_rd_addr <= ({8'b0,out_row_sel}+9'd1) * {1'b0,input_w} + {1'b0,col_idx} + 9'd1; // tap4
                            4'd6: bram_rd_addr <= ({8'b0,out_row_sel}+9'd1) * {1'b0,input_w} + {1'b0,col_idx} + 9'd2; // tap5
                            4'd7: bram_rd_addr <= ({8'b0,out_row_sel}+9'd2) * {1'b0,input_w} + {1'b0,col_idx};  // tap6
                            4'd8: bram_rd_addr <= ({8'b0,out_row_sel}+9'd2) * {1'b0,input_w} + {1'b0,col_idx} + 9'd1; // tap7
                            4'd9: bram_rd_addr <= ({8'b0,out_row_sel}+9'd2) * {1'b0,input_w} + {1'b0,col_idx} + 9'd2; // tap8
                            default: bram_rd_addr <= 9'd0;
                        endcase
                        tap_cnt    <= tap_cnt + 1;

                    end else if (tap_cnt == 4'd10) begin
                        // tap10: last MAC (tap8), no new bram addr needed
                        pe_clr     <= 1'b0;
                        pe_en      <= 1'b1;
                        wgt_rd_tap <= 4'd8;
                        tap_cnt    <= 4'd11;
                        acc_row_sel <= out_row_sel;
                        acc_col     <= col_idx;
                    end else begin
                        // tap11: PE register holds complete sum(tap0..tap8), WRITE NOW
                        pe_clr    <= 1'b0;
                        pe_en     <= 1'b0;
                        acc_wr_en <= 1'b1;
                        tap_cnt   <= 4'd0;

                        if (cur_col < ({1'b0,output_w} * 8'd2) - 8'd1) begin
                            cur_col <= cur_col + 8'd1;
                        end else begin
                            cur_col <= 8'd0;
                            state   <= NEXT_GROUP;
                        end
                    end
                end

                NEXT_GROUP: begin
                    acc_wr_en <= 1'b0;
                    if (cur_group < num_groups - 1) begin
                        cur_group <= cur_group + 1;
                        state     <= LOAD_WEIGHTS;
                    end else begin
                        cur_group  <= 8'd0;
                        out_wr_cnt <= 14'd0;
                        state      <= WRITE_OUT;
                    end
                end

                QUANTIZE: begin
                    // State eliminated: async acc_row_buffer read means
                    // quant_col is driven directly in WRITE_OUT state.
                    // This state is now unreachable (kept for encoding compatibility).
                    state <= WRITE_OUT;
                end

                WRITE_OUT: begin
                    // out_wr_cnt=0: pipeline delay, no write
                    // Conv: out_wr_cnt=1..output_w   ? row 0
                    //       out_wr_cnt=output_w+1..output_w*2 ? row 1
                    // Pool: out_wr_cnt=1..output_w   ? single output row
                    //       (pool_valid fires every 2 input cols, output_w = input_w/2)
                    // BUG FIX: removed incorrect look-ahead. quant_col is an output reg   1 cycle lag relative to out_wr_cnt.
                    // The output path: quant_col(reg) -> acc_rd(comb) -> q_out(comb)
                    // -> post_relu(comb) -> out_pixel_reg(reg). out_wr_en_d and
                    // out_wr_addr_d are also registered on the SAME edge as out_pixel_reg.
                    // All three register at the same edge, so quant_col must = current
                    // column being written, i.e. (out_wr_cnt-1). No look-ahead needed.
                    // The old look-ahead wrote acc[col+1] to col[N] addr, and wrote
                    // acc[output_w]=0 (out of cleared range) to the last column.
                    if (out_wr_cnt == 14'd0) begin
                        out_wr_en     <= 1'b0;
                        quant_col     <= 8'd0;     // prefetch col=0 for cnt=1
                        quant_row_sel <= 1'b0;
                        out_wr_cnt    <= out_wr_cnt + 1;
                    end else if (mode) begin
                        // Pool mode: write output_w pixels, row = cur_row_pair
                        out_wr_en     <= 1'b1;
                        out_wr_addr   <= {7'b0,cur_row_pair} * {7'b0,output_w} +
                                          (out_wr_cnt - 1);
                        // FIX: quant_col = current col (out_wr_cnt-1), NOT look-ahead
                        // out_wr_en_d/out_wr_addr_d/out_pixel_reg all register on the SAME
                        // clock edge, so there is no skew - no look-ahead is needed.
                        quant_col     <= (out_wr_cnt - 14'd1) & 14'hFF;
                        quant_row_sel <= 1'b0;
                        if (out_wr_cnt < {7'b0,output_w})
                            out_wr_cnt <= out_wr_cnt + 1;
                        else
                            state <= NEXT_ROW_PAIR;
                    end else begin
                        // Conv mode: write output_w*2 pixels (row0 then row1)
                        out_wr_en <= 1'b1;
                        if ((out_wr_cnt - 1) < {7'b0, output_w}) begin
                            out_wr_addr   <= {7'b0,cur_row_pair} * 2 *
                                              {7'b0,output_w} +
                                              (out_wr_cnt - 1);
                            // FIX: quant_col = out_wr_cnt-1 (current col, not look-ahead)
                            quant_col     <= (out_wr_cnt - 14'd1) & 14'hFF;
                            quant_row_sel <= 1'b0;
                        end else begin
                            out_wr_addr   <= ({7'b0,cur_row_pair}*2+14'd1) *
                                              {7'b0,output_w} +
                                              (out_wr_cnt - 1 - {7'b0,output_w});
                            // FIX: quant_col = out_wr_cnt-1-output_w (current col, not look-ahead)
                            quant_col <= (out_wr_cnt - 14'd1 - {6'b0,output_w}) & 14'hFF;
                            quant_row_sel <= 1'b1;
                        end
                        if (out_wr_cnt < {7'b0,output_w}*2)
                            out_wr_cnt <= out_wr_cnt + 1;
                        else
                            state <= NEXT_ROW_PAIR;
                    end
                end

                NEXT_ROW_PAIR: begin
                    acc_wr_en <= 1'b0;
                    out_wr_en <= 1'b0;
                    if (cur_row_pair < num_row_pairs - 1) begin
                        cur_row_pair <= cur_row_pair + 1;
                        state        <= LOAD_WEIGHTS;
                    end else begin
                        cur_row_pair <= 8'd0;
                        total_pixels <= {7'b0,output_h} * {7'b0,output_w};
                        stream_start <= 1'b1;
                        state        <= WAIT_STREAM;
                    end
                end

                WAIT_STREAM: begin
                    if (stream_done) begin
                        stream_start <= 1'b0;
                        state        <= NEXT_OC;
                    end
                end

                NEXT_OC: begin
                    if (cur_oc < num_out_ch - 1) begin
                        cur_oc <= cur_oc + 1;
                        state  <= LOAD_WEIGHTS;
                    end else
                        state <= ALL_DONE;
                end

                ALL_DONE: begin
                    done      <= 1'b1;
                    busy      <= 1'b0;
                    run_armed <= 1'b0;   // allow next layer start
                    state     <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule