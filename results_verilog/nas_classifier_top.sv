// =============================================================================
// NAS Wireless Signal Classifier – Synthesizable SystemVerilog
//
// Conv1D(5,2->16,ELU) -> Conv1D(5,16->32,ELU) -> GAP -> Dense(32->16,ELU)
//                     -> Dense(16->3,argmax)
// Fixed-point Q8.8  |  Classes: 0=LTE  1=DVB-T  2=WiFi
// Compatible: Icarus Verilog / Vivado / Quartus
// =============================================================================
`timescale 1ns / 1ps

module nas_classifier_top #(
    parameter int DW    = 16,
    parameter int FRAC  = 12,
    parameter int ACC_W = 40,
    parameter int SEQ   = 512,
    parameter int IN_CH = 2,
    parameter int K     = 5,
    parameter int C1F   = 16,
    parameter int C2F   = 32,
    parameter int D1U   = 16,
    parameter int NC    = 3
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              start,
    input  logic              sample_valid,
    input  logic signed [DW-1:0] iq_real,
    input  logic signed [DW-1:0] iq_imag,
    output logic [1:0]        class_out,
    output logic              result_valid
);

    // ── Local parameters ──────────────────────────────────────────────────────
    localparam int PAD    = K / 2;       // 2
    localparam int C1_MAC = K * IN_CH;   // 10
    localparam int C2_MAC = K * C1F;     // 80
    localparam int KC1_END = C1_MAC + 1; // 11  (bias + 10 MACs + ELU-write)
    localparam int KC2_END = C2_MAC + 1; // 81

    // ── States ────────────────────────────────────────────────────────────────
    typedef enum logic [3:0] {
        S_IDLE, S_RECV, S_CONV1, S_CONV2,
        S_GAP, S_GAP_DIV, S_DENSE1, S_DENSE2, S_ARGMAX, S_DONE
    } state_t;
    state_t state;

    // ── Weight ROMs ───────────────────────────────────────────────────────────
    // Loaded from *.hex files at elaboration time via $readmemh.
    // Indexing: conv kernel[k, cin, fout] -> flat = k*IN_CH*C1F + cin*C1F + fout
    //           dense    W[in, out]       -> flat = in*D1U + out
    logic signed [DW-1:0] c1w [0:K*IN_CH*C1F-1];
    logic signed [DW-1:0] c1b [0:C1F-1];
    logic signed [DW-1:0] c2w [0:K*C1F*C2F-1];
    logic signed [DW-1:0] c2b [0:C2F-1];
    logic signed [DW-1:0] d1w [0:C2F*D1U-1];
    logic signed [DW-1:0] d1b [0:D1U-1];
    logic signed [DW-1:0] d2w [0:D1U*NC-1];
    logic signed [DW-1:0] d2b [0:NC-1];

    initial begin
        $readmemh("c1w.hex", c1w);  $readmemh("c1b.hex", c1b);
        $readmemh("c2w.hex", c2w);  $readmemh("c2b.hex", c2b);
        $readmemh("d1w.hex", d1w);  $readmemh("d1b.hex", d1b);
        $readmemh("d2w.hex", d2w);  $readmemh("d2b.hex", d2b);
    end

    // ── Feature map memories (synthesis tool infers BRAMs) ────────────────────
    logic signed [DW-1:0] input_mem [0:SEQ-1][0:IN_CH-1];
    logic signed [DW-1:0] c1_mem   [0:SEQ-1][0:C1F-1];
    logic signed [DW-1:0] c2_mem   [0:SEQ-1][0:C2F-1];

    // ── Accumulators and output registers ─────────────────────────────────────
    logic signed [ACC_W-1:0] c1_acc [0:C1F-1];
    logic signed [ACC_W-1:0] c2_acc [0:C2F-1];
    logic signed [ACC_W-1:0] gap_acc[0:C2F-1];
    logic signed [DW-1:0]    gap_out[0:C2F-1];
    logic signed [ACC_W-1:0] d1_acc [0:D1U-1];
    logic signed [DW-1:0]    d1_out [0:D1U-1];
    logic signed [ACC_W-1:0] d2_acc [0:NC-1];

    // ── Counters ──────────────────────────────────────────────────────────────
    logic [8:0]  t_cnt;    // time-step counter (0..511)
    logic [6:0]  kc_cnt;   // MAC counter       (0..81)

    // ── Combinatorial index wires for Conv1 ───────────────────────────────────
    // kc_cnt=1..C1_MAC maps kc=0..9  (k*IN_CH + ch)
    // IN_CH=2 is a power of 2: k = kc>>1,  ch = kc[0]
    logic [3:0]  c1_kc;
    logic [2:0]  c1_k;
    logic        c1_ch;
    logic signed [9:0] c1_ts;

    assign c1_kc = kc_cnt - 1;
    assign c1_k  = c1_kc[3:1];
    assign c1_ch = c1_kc[0];
    assign c1_ts = $signed({1'b0, t_cnt}) + $signed({7'b0, c1_k}) - $signed(10'(PAD));

    // ── Combinatorial index wires for Conv2 ───────────────────────────────────
    // C1F=16 is a power of 2: k = kc>>4,  ch = kc[3:0]
    logic [6:0]  c2_kc;
    logic [2:0]  c2_k;
    logic [3:0]  c2_ch;
    logic signed [9:0] c2_ts;

    assign c2_kc = kc_cnt - 1;
    assign c2_k  = c2_kc[6:4];
    assign c2_ch = c2_kc[3:0];
    assign c2_ts = $signed({1'b0, t_cnt}) + $signed({7'b0, c2_k}) - $signed(10'(PAD));

    // ── ELU activation function ───────────────────────────────────────────────
    // Input: 40-bit raw accumulator.  Output: Q8.8 16-bit result.
    // x >= 0         ->  x
    // -1.0 <= x < 0  ->  x >> 1   (linear fit of alpha*(e^x-1), alpha=1)
    // x < -1.0       ->  -1.0     (saturate)
    localparam logic signed [DW-1:0] ELU_MIN = -16'sh1000; // -1.0 in Q-format

    function automatic logic signed [DW-1:0] elu(
        input logic signed [ACC_W-1:0] acc
    );
        logic signed [DW-1:0] x;
        x = acc[FRAC +: DW];           // extract fixed-point result (shift right by FRAC)
        if      (x >= 0)        return x;
        else if (x >= ELU_MIN)  return x >>> 1;
        else                    return ELU_MIN;
    endfunction

    // ── Main state machine ─────────────────────────────────────────────────────
    integer f, n, kk;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            result_valid <= 1'b0;
            class_out    <= 2'd0;
            t_cnt        <= '0;
            kc_cnt       <= '0;
        end else begin
            result_valid <= 1'b0;
            case (state)

                // ── Idle ─────────────────────────────────────────────────
                S_IDLE: if (start) begin
                    state  <= S_RECV;
                    t_cnt  <= '0;
                end

                // ── Receive 512 IQ samples ────────────────────────────────
                S_RECV: if (sample_valid) begin
                    input_mem[t_cnt][0] <= iq_real;
                    input_mem[t_cnt][1] <= iq_imag;
                    if (t_cnt == SEQ - 1) begin
                        state  <= S_CONV1;
                        t_cnt  <= '0;
                        kc_cnt <= '0;
                    end else
                        t_cnt <= t_cnt + 1;
                end

                // ── Conv1: kernel(5, 2->16, ELU, same-padding) ────────────
                // kc_cnt=0       : load biases into accumulators
                // kc_cnt=1..10  : accumulate MAC (k, ch) for all 16 filters
                // kc_cnt=11     : write ELU output to c1_mem (acc is final here)
                S_CONV1: begin
                    case (kc_cnt)
                        0: begin
                            for (f=0; f<C1F; f=f+1)
                                c1_acc[f] <= {{(ACC_W-DW){c1b[f][DW-1]}}, c1b[f]} <<< FRAC;
                            kc_cnt <= kc_cnt + 1;
                        end
                        KC1_END: begin
                            for (f=0; f<C1F; f=f+1)
                                c1_mem[t_cnt][f] <= elu(c1_acc[f]);
                            kc_cnt <= '0;
                            if (t_cnt == SEQ - 1) begin
                                state <= S_CONV2;  t_cnt <= '0;
                            end else
                                t_cnt <= t_cnt + 1;
                        end
                        default: begin
                            if (c1_ts >= 0 && c1_ts < SEQ)
                                for (f=0; f<C1F; f=f+1)
                                    c1_acc[f] <= c1_acc[f]
                                        + {{(ACC_W-DW){input_mem[c1_ts][c1_ch][DW-1]},
                                           input_mem[c1_ts][c1_ch]}
                                        * {{(ACC_W-DW){c1w[c1_k*IN_CH*C1F + c1_ch*C1F + f][DW-1]}},
                                           c1w[c1_k*IN_CH*C1F + c1_ch*C1F + f]};
                            kc_cnt <= kc_cnt + 1;
                        end
                    endcase
                end

                // ── Conv2: kernel(5, 16->32, ELU, same-padding) ───────────
                S_CONV2: begin
                    case (kc_cnt)
                        0: begin
                            for (f=0; f<C2F; f=f+1)
                                c2_acc[f] <= {{(ACC_W-DW){c2b[f][DW-1]}}, c2b[f]} <<< FRAC;
                            kc_cnt <= kc_cnt + 1;
                        end
                        KC2_END: begin
                            for (f=0; f<C2F; f=f+1)
                                c2_mem[t_cnt][f] <= elu(c2_acc[f]);
                            kc_cnt <= '0;
                            if (t_cnt == SEQ - 1) begin
                                state  <= S_GAP;  t_cnt <= '0;
                                for (f=0; f<C2F; f=f+1)  gap_acc[f] <= '0;
                            end else
                                t_cnt <= t_cnt + 1;
                        end
                        default: begin
                            if (c2_ts >= 0 && c2_ts < SEQ)
                                for (f=0; f<C2F; f=f+1)
                                    c2_acc[f] <= c2_acc[f]
                                        + {{(ACC_W-DW){c1_mem[c2_ts][c2_ch][DW-1]}},
                                           c1_mem[c2_ts][c2_ch]}
                                        * {{(ACC_W-DW){c2w[c2_k*C1F*C2F + c2_ch*C2F + f][DW-1]}},
                                           c2w[c2_k*C1F*C2F + c2_ch*C2F + f]};
                            kc_cnt <= kc_cnt + 1;
                        end
                    endcase
                end

                // ── GlobalAveragePooling1D ────────────────────────────────
                S_GAP: begin
                    for (f=0; f<C2F; f=f+1)
                        gap_acc[f] <= gap_acc[f]
                                    + {{(ACC_W-DW){c2_mem[t_cnt][f][DW-1]}},
                                       c2_mem[t_cnt][f]};
                    if (t_cnt == SEQ - 1)
                        state <= S_GAP_DIV;
                    else
                        t_cnt <= t_cnt + 1;
                end

                // Divide by 512 (shift right 9), init Dense1 biases
                S_GAP_DIV: begin
                    for (f=0; f<C2F; f=f+1)
                        gap_out[f] <= DW'(gap_acc[f] >>> 9);
                    for (n=0; n<D1U; n=n+1)
                        d1_acc[n] <= {{(ACC_W-DW){d1b[n][DW-1]}}, d1b[n]} <<< FRAC;
                    state <= S_DENSE1;
                    t_cnt <= '0;
                end

                // ── Dense1: (32->16, ELU) ─────────────────────────────────
                // Iterate t_cnt over 32 inputs; all 16 outputs accumulate in parallel.
                S_DENSE1: begin
                    for (n=0; n<D1U; n=n+1)
                        d1_acc[n] <= d1_acc[n]
                                   + {{(ACC_W-DW){gap_out[t_cnt][DW-1]}}, gap_out[t_cnt]}
                                   * {{(ACC_W-DW){d1w[t_cnt*D1U + n][DW-1]}},
                                      d1w[t_cnt*D1U + n]};
                    if (t_cnt == C2F - 1) begin
                        for (n=0; n<D1U; n=n+1)
                            d1_out[n] <= elu(d1_acc[n]);
                        for (kk=0; kk<NC; kk=kk+1)
                            d2_acc[kk] <= {{(ACC_W-DW){d2b[kk][DW-1]}}, d2b[kk]} <<< FRAC;
                        state <= S_DENSE2;  t_cnt <= '0;
                    end else
                        t_cnt <= t_cnt + 1;
                end

                // ── Dense2: (16->3, no activation) ────────────────────────
                S_DENSE2: begin
                    for (kk=0; kk<NC; kk=kk+1)
                        d2_acc[kk] <= d2_acc[kk]
                                    + {{(ACC_W-DW){d1_out[t_cnt][DW-1]}}, d1_out[t_cnt]}
                                    * {{(ACC_W-DW){d2w[t_cnt*NC + kk][DW-1]}},
                                       d2w[t_cnt*NC + kk]};
                    if (t_cnt == D1U - 1)
                        state <= S_ARGMAX;
                    else
                        t_cnt <= t_cnt + 1;
                end

                // ── Argmax (unrolled for NC=3) ────────────────────────────
                S_ARGMAX: begin
                    if      (d2_acc[0] >= d2_acc[1] && d2_acc[0] >= d2_acc[2])
                        class_out <= 2'd0;
                    else if (d2_acc[1] >= d2_acc[2])
                        class_out <= 2'd1;
                    else
                        class_out <= 2'd2;
                    state <= S_DONE;
                end

                // ── Done: pulse result_valid ───────────────────────────────
                S_DONE: begin
                    result_valid <= 1'b1;
                    state        <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
