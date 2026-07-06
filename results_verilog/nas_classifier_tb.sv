// =============================================================================
// Testbench for nas_classifier_top
//
// Feeds 9 real IQ test vectors (3 per class: LTE, DVB-T, WiFi) through the
// hardware model and checks the predicted class against the expected label.
//
// Simulation tool: Icarus Verilog (iverilog) v10+
//   iverilog -g2012 -o sim nas_classifier_top.sv nas_classifier_tb.sv
//   vvp sim
// =============================================================================

`timescale 1ns / 1ps

module nas_classifier_tb;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int CLK_PERIOD  = 10;   // 100 MHz
    localparam int N_SAMPLES   = 9;
    localparam int SEQ_LEN     = 512;
    localparam int DW          = 16;

    // ── DUT signals ───────────────────────────────────────────────────────────
    logic        clk          = 0;
    logic        rst_n        = 0;
    logic        start        = 0;
    logic        sample_valid = 0;
    logic signed [DW-1:0] iq_real = 0;
    logic signed [DW-1:0] iq_imag = 0;
    logic [1:0]  class_out;
    logic        result_valid;

    // ── DUT instantiation ─────────────────────────────────────────────────────
    nas_classifier_top dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .sample_valid (sample_valid),
        .iq_real      (iq_real),
        .iq_imag      (iq_imag),
        .class_out    (class_out),
        .result_valid (result_valid)
    );

    // ── Clock ─────────────────────────────────────────────────────────────────
    always #(CLK_PERIOD/2) clk = ~clk;

    // ── Test vector storage ───────────────────────────────────────────────────
    // Each entry is 32 bits: {real[15:0], imag[15:0]}
    logic [31:0] vectors [0:N_SAMPLES*SEQ_LEN-1];
    integer      labels  [0:N_SAMPLES-1];

    // ── Class name helper ─────────────────────────────────────────────────────
    function string class_name(input int idx);
        case (idx)
            0: return "LTE  ";
            1: return "DVB-T";
            2: return "WiFi ";
            default: return "?????";
        endcase
    endfunction

    // ── Main test sequence ────────────────────────────────────────────────────
    integer i, t, expected, predicted;
    integer pass_count, fail_count;
    integer timeout_cnt;

    initial begin
        // Load test vectors and labels from hex/text files
        $readmemh("test_vectors.hex", vectors);
        $readmemh("test_labels.hex",  labels);

        pass_count = 0;
        fail_count = 0;

        // Reset (hold low for 4 cycles, release #1 after edge)
        rst_n = 0;
        repeat(4) @(posedge clk);
        #1; rst_n = 1;
        @(posedge clk); #1;

        $display("=============================================================");
        $display("  NAS Wireless Classifier – Hardware Simulation");
        $display("  Fixed-point Q8.8 | 100 MHz | %0d test samples", N_SAMPLES);
        $display("=============================================================");
        $display("  #  Expected   Predicted  Result");
        $display("  -  --------   ---------  ------");

        // Run one inference per test sample
        for (i = 0; i < N_SAMPLES; i++) begin
            expected = labels[i];

            // Pulse start for one clock cycle.
            // All signal changes happen #1 AFTER the posedge so the DUT
            // samples the previous value at the edge and sees the new
            // value only at the NEXT edge (avoids testbench/DUT race).
            @(posedge clk); #1;
            start = 1;
            @(posedge clk); #1;   // DUT samples start=1 at this edge
            start = 0;

            // Stream 512 IQ samples, one per clock
            sample_valid = 1;
            for (t = 0; t < SEQ_LEN; t++) begin
                iq_real = $signed(vectors[i*SEQ_LEN + t][31:16]);
                iq_imag = $signed(vectors[i*SEQ_LEN + t][15:0]);
                @(posedge clk); #1;   // DUT reads sample at this edge
            end
            sample_valid = 0;
            iq_real = 0;
            iq_imag = 0;

            // Wait for result (with timeout)
            timeout_cnt = 0;
            while (!result_valid && timeout_cnt < 100_000) begin
                @(posedge clk);
                timeout_cnt++;
            end

            if (timeout_cnt >= 100_000) begin
                $display("  %1d  %-9s  TIMEOUT    FAIL", i, class_name(expected));
                fail_count++;
            end else begin
                predicted = class_out;
                if (predicted == expected) begin
                    $display("  %1d  %-9s  %-9s  PASS", i,
                             class_name(expected), class_name(predicted));
                    pass_count++;
                end else begin
                    $display("  %1d  %-9s  %-9s  FAIL  <---", i,
                             class_name(expected), class_name(predicted));
                    fail_count++;
                end
            end

            // Wait a few cycles before next sample
            repeat(5) @(posedge clk);
        end

        $display("=============================================================");
        $display("  Results: %0d / %0d passed", pass_count, N_SAMPLES);
        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  %0d FAILED", fail_count);
        $display("=============================================================");

        $finish;
    end

    // ── State monitoring for first inference ─────────────────────────────────
    initial begin
        @(posedge dut.rst_n);  // wait for reset release
        forever begin
            @(posedge clk);
            if ($time > 0 && ($time / CLK_PERIOD) % 5000 == 0)
                $display("  [t=%0t cyc=%0d] state=%0d t_cnt=%0d kc_cnt=%0d",
                         $time, $time/CLK_PERIOD,
                         dut.state, dut.t_cnt, dut.kc_cnt);
        end
    end

    // ── Simulation timeout guard ──────────────────────────────────────────────
    initial begin
        #(CLK_PERIOD * 600_000);
        $display("GLOBAL TIMEOUT: simulation exceeded 600k cycles");
        $finish;
    end

endmodule
