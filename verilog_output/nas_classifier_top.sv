// =============================================================================
// NAS Wireless Signal Classifier – Synthesizable SystemVerilog
//
// Architecture : Conv1D(5,2→16,ELU) → Conv1D(5,16→32,ELU)
//                → GlobalAvgPool1D → Dense(32→16,ELU) → Dense(16→3,argmax)
// Fixed-point  : Q8.8  (signed 16-bit, 8 fractional bits)
// Classes      : 0=LTE  1=DVB-T  2=WiFi
//
// Interfaces
//   clk          : system clock
//   rst_n        : active-low synchronous reset
//   start        : single-cycle pulse to begin a new inference
//   sample_valid : asserted while IQ samples are being streamed in
//   iq_real/iq_imag : one 16-bit Q8.8 sample per clock (512 total)
//   class_out    : 2-bit result (valid when result_valid is high)
//   result_valid : single-cycle pulse when inference is complete
//
// Weight ROMs are loaded via $readmemh at elaboration time.
// Place the *.hex files in the simulation/synthesis working directory.
// =============================================================================

`timescale 1ns / 1ps

module nas_classifier_top #(
    parameter int DW    = 16,    // data / weight width
    parameter int FRAC  = 8,    // fractional bits
    parameter int ACC_W = 40,   // accumulator width
    parameter int SEQ   = 512,  // input sequence length
    parameter int IN_CH = 2,     // IQ channels
    parameter int K     = 5,     // conv kernel size (must be odd)
    parameter int C1F   = 16,    // conv1 output filters
    parameter int C2F   = 32,    // conv2 output filters
    parameter int D1U   = 16,    // dense1 units
    parameter int NC    = 3      // number of classes
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

    // ── Local parameters ─────────────────────────────────────────────────────
    localparam int PAD     = K / 2;          // same-padding half-width (2)
    localparam int C1_MAC  = K * IN_CH;      // MACs per t for conv1 (10)
    localparam int C2_MAC  = K * C1F;        // MACs per t for conv2 (80)
    localparam int KC1_MAX = C1_MAC;         // kc counter max for conv1
    localparam int KC2_MAX = C2_MAC;         // kc counter max for conv2

    // ── State machine ────────────────────────────────────────────────────────
    typedef enum logic [3:0] {
        S_IDLE, S_RECV,
        S_CONV1, S_CONV2,
        S_GAP, S_DENSE1, S_DENSE2,
        S_ARGMAX, S_DONE
    } state_t;
    state_t state;

    // ── Weight ROMs ──────────────────────────────────────────────────────────
    // conv1: kernel[k, cin, fout]  flat = k*IN_CH*C1F + cin*C1F + fout
    logic signed [DW-1:0] c1w [0:K*IN_CH*C1F-1];
    logic signed [DW-1:0] c1b [0:C1F-1];
    // conv2: kernel[k, cin, fout]  flat = k*C1F*C2F + cin*C2F + fout
    logic signed [DW-1:0] c2w [0:K*C1F*C2F-1];
    logic signed [DW-1:0] c2b [0:C2F-1];
    // dense1: W[in, out]           flat = in*D1U + out
    logic signed [DW-1:0] d1w [0:C2F*D1U-1];
    logic signed [DW-1:0] d1b [0:D1U-1];
    // dense2: W[in, out]           flat = in*NC + out
    logic signed [DW-1:0] d2w [0:D1U*NC-1];
    logic signed [DW-1:0] d2b [0:NC-1];

    initial begin
        $readmemh("c1w.hex", c1w);
        $readmemh("c1b.hex", c1b);
        $readmemh("c2w.hex", c2w);
        $readmemh("c2b.hex", c2b);
        $readmemh("d1w.hex", d1w);
        $readmemh("d1b.hex", d1b);
        $readmemh("d2w.hex", d2w);
        $readmemh("d2b.hex", d2b);
    end

    // ── Feature map memories (inferred as BRAMs by synthesis) ────────────────
    logic signed [DW-1:0] input_mem [0:SEQ-1][0:IN_CH-1];
    logic signed [DW-1:0] c1_mem   [0:SEQ-1][0:C1F-1];
    logic signed [DW-1:0] c2_mem   [0:SEQ-1][0:C2F-1];

    // ── Intermediate registers ───────────────────────────────────────────────
    logic signed [ACC_W-1:0] c1_acc [0:C1F-1];
    logic signed [ACC_W-1:0] c2_acc [0:C2F-1];
    logic signed [ACC_W-1:0] gap_acc[0:C2F-1];  // wider: accumulates 512 values
    logic signed [DW-1:0]   gap_out [0:C2F-1];
    logic signed [ACC_W-1:0] d1_acc [0:D1U-1];
    logic signed [DW-1:0]   d1_out  [0:D1U-1];
    logic signed [ACC_W-1:0] d2_acc [0:NC-1];

    // ── Counters ─────────────────────────────────────────────────────────────
    logic [$clog2(SEQ)-1:0]   t_cnt;
    logic [$clog2(C2_MAC):0]  kc_cnt;   // sized for max(C1_MAC, C2_MAC) = 80

    // ── ELU approximation ────────────────────────────────────────────────────
    // ELU(x) = x            if x >= 0
    //        = x / 2        if -1.0 <= x < 0   (linear approx of e^x - 1)
    //        = -1.0         if x < -1.0         (saturate)
    // The input `acc` is the raw accumulator value; shift right by FRAC to get
    // the Q8.8 result, then apply the activation.
    function automatic logic signed [DW-1:0] elu(
        input logic signed [ACC_W-1:0] acc
    );
        logic signed [DW-1:0] x;
        x = acc[FRAC +: DW];  // arithmetic shift right by FRAC (extract Q8.8)
        if (x >= 0)
            return x;
        else if (x >= -$signed(DW'(SCALE)))   // -1.0 in Q8.8 = -256
            return x >>> 1;                    // ≈ (e^x - 1) for small |x|
        else
            return -$signed(DW'(SCALE));       // saturate at -1.0
    endfunction

    // ── Main state machine ───────────────────────────────────────────────────
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

                // ── Wait for start pulse ──────────────────────────────────
                S_IDLE: begin
                    if (start) begin
                        state  <= S_RECV;
                        t_cnt  <= '0;
                    end
                end

                // ── Receive 512 IQ samples ────────────────────────────────
                S_RECV: begin
                    if (sample_valid) begin
                        input_mem[t_cnt][0] <= iq_real;
                        input_mem[t_cnt][1] <= iq_imag;
                        if (t_cnt == SEQ - 1) begin
                            state  <= S_CONV1;
                            t_cnt  <= '0;
                            kc_cnt <= '0;
                        end else begin
                            t_cnt <= t_cnt + 1;
                        end
                    end
                end

                // ── Conv1: kernel(5,2→16) with same padding ───────────────
                // kc_cnt = 0        : load bias into accumulator
                // kc_cnt = 1..C1_MAC: accumulate MAC for (k,c) pair
                S_CONV1: begin
                    if (kc_cnt == 0) begin
                        // Bias initialisation – scale up so accumulator is Q.2*FRAC
                        for (int f = 0; f < C1F; f++)
                            c1_acc[f] <= ACC_W'($signed(c1b[f])) <<< FRAC;
                        kc_cnt <= kc_cnt + 1;
                    end else begin
                        automatic int kc  = kc_cnt - 1;
                        automatic int k   = kc / IN_CH;
                        automatic int ch  = kc % IN_CH;
                        automatic int t_s = int'(t_cnt) + k - PAD;

                        for (int f = 0; f < C1F; f++) begin
                            if (t_s >= 0 && t_s < SEQ) begin
                                c1_acc[f] <= c1_acc[f] +
                                    ACC_W'($signed(input_mem[t_s][ch])) *
                                    ACC_W'($signed(c1w[k*IN_CH*C1F + ch*C1F + f]));
                            end
                        end

                        if (kc_cnt == C1_MAC) begin
                            // Apply ELU and write feature map
                            for (int f = 0; f < C1F; f++)
                                c1_mem[t_cnt][f] <= elu(c1_acc[f]);
                            if (t_cnt == SEQ - 1) begin
                                state  <= S_CONV2;
                                t_cnt  <= '0;
                            end else begin
                                t_cnt <= t_cnt + 1;
                            end
                            kc_cnt <= '0;
                        end else begin
                            kc_cnt <= kc_cnt + 1;
                        end
                    end
                end

                // ── Conv2: kernel(5,16→32) with same padding ──────────────
                S_CONV2: begin
                    if (kc_cnt == 0) begin
                        for (int f = 0; f < C2F; f++)
                            c2_acc[f] <= ACC_W'($signed(c2b[f])) <<< FRAC;
                        kc_cnt <= kc_cnt + 1;
                    end else begin
                        automatic int kc  = kc_cnt - 1;
                        automatic int k   = kc / C1F;
                        automatic int ch  = kc % C1F;
                        automatic int t_s = int'(t_cnt) + k - PAD;

                        for (int f = 0; f < C2F; f++) begin
                            if (t_s >= 0 && t_s < SEQ) begin
                                c2_acc[f] <= c2_acc[f] +
                                    ACC_W'($signed(c1_mem[t_s][ch])) *
                                    ACC_W'($signed(c2w[k*C1F*C2F + ch*C2F + f]));
                            end
                        end

                        if (kc_cnt == C2_MAC) begin
                            for (int f = 0; f < C2F; f++)
                                c2_mem[t_cnt][f] <= elu(c2_acc[f]);
                            if (t_cnt == SEQ - 1) begin
                                state  <= S_GAP;
                                t_cnt  <= '0;
                                for (int f = 0; f < C2F; f++)
                                    gap_acc[f] <= '0;
                            end else begin
                                t_cnt <= t_cnt + 1;
                            end
                            kc_cnt <= '0;
                        end else begin
                            kc_cnt <= kc_cnt + 1;
                        end
                    end
                end

                // ── GlobalAveragePooling1D: accumulate over SEQ ───────────
                // gap_out[f] = sum(c2_mem[t][f], t=0..SEQ-1) / SEQ
                // Division by 512 = arithmetic right-shift by 9.
                S_GAP: begin
                    for (int f = 0; f < C2F; f++)
                        gap_acc[f] <= gap_acc[f] + ACC_W'($signed(c2_mem[t_cnt][f]));

                    if (t_cnt == SEQ - 1) begin
                        // Divide by SEQ (512 = 2^9) via arithmetic shift
                        for (int f = 0; f < C2F; f++)
                            gap_out[f] <= DW'(gap_acc[f] >>> 9);
                        state  <= S_DENSE1;
                        t_cnt  <= '0;
                        for (int n = 0; n < D1U; n++)
                            d1_acc[n] <= ACC_W'($signed(d1b[n])) <<< FRAC;
                    end else begin
                        t_cnt <= t_cnt + 1;
                    end
                end

                // ── Dense1: (32→16) with ELU ──────────────────────────────
                // Iterate over the 32 input neurons; all 16 output neurons
                // accumulate in parallel.
                S_DENSE1: begin
                    for (int n = 0; n < D1U; n++)
                        d1_acc[n] <= d1_acc[n] +
                            ACC_W'($signed(gap_out[t_cnt])) *
                            ACC_W'($signed(d1w[int'(t_cnt)*D1U + n]));

                    if (t_cnt == C2F - 1) begin
                        for (int n = 0; n < D1U; n++)
                            d1_out[n] <= elu(d1_acc[n]);
                        state  <= S_DENSE2;
                        t_cnt  <= '0;
                        for (int k = 0; k < NC; k++)
                            d2_acc[k] <= ACC_W'($signed(d2b[k])) <<< FRAC;
                    end else begin
                        t_cnt <= t_cnt + 1;
                    end
                end

                // ── Dense2: (16→3) – no activation, argmax follows ────────
                S_DENSE2: begin
                    for (int k = 0; k < NC; k++)
                        d2_acc[k] <= d2_acc[k] +
                            ACC_W'($signed(d1_out[t_cnt])) *
                            ACC_W'($signed(d2w[int'(t_cnt)*NC + k]));

                    if (t_cnt == D1U - 1) begin
                        state <= S_ARGMAX;
                    end else begin
                        t_cnt <= t_cnt + 1;
                    end
                end

                // ── Argmax: find class with highest score ─────────────────
                S_ARGMAX: begin
                    begin
                        automatic logic signed [ACC_W-1:0] best     = d2_acc[0];
                        automatic logic [1:0]              best_idx = 2'd0;
                        for (int k = 1; k < NC; k++) begin
                            if (d2_acc[k] > best) begin
                                best     = d2_acc[k];
                                best_idx = 2'(k);
                            end
                        end
                        class_out <= best_idx;
                    end
                    state <= S_DONE;
                end

                // ── Output result ─────────────────────────────────────────
                S_DONE: begin
                    result_valid <= 1'b1;
                    state        <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
