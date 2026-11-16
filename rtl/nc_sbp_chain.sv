// Spiking-band power for N_CH channels through one shared datapath.
//
//   8-bit ADC code -> template artifact canceller -> 300 Hz high-pass
//                  -> |.| -> sum over BIN_LEN samples (50 ms at 2 kS/s)
//
// One channel's sample enters per clock cycle and moves through four stages.
// The datapath holds no channel state; each stage reads the words it needs from
// a small RAM addressed by the channel number and writes them back one cycle
// later:
//
//   edge 0  sample accepted           read ctl[ch]            {taps left, loaded}
//   edge 1  S1 tap index k            write ctl, read tpl[ch,k]
//   edge 2  S2 canceller              write tpl, read flt[ch]  {x1, x2, y1, y2}
//   edge 3  S3 high-pass              write flt, read bin[ch]  {sum, count, n}
//   edge 4  S4 |y| and accumulate     write bin; sample taps valid
//   edge 5  bin outputs valid (once every BIN_LEN samples of a channel)
//
// Spacing rule: the same channel must not be presented on two consecutive
// clock edges, because the second read would meet the first write. With
// N_CH >= 2 channels taken in turn the rule holds with no idle cycles, so the
// clock can be as slow as N_CH * fs (192 kHz for 96 channels at 2 kS/s).
//
// in_trig marks the sample on which a stimulation pulse starts; the stimulator
// runs off the ADC clock. Pulses must be at least N_TAPS samples apart.
//
// After reset the three state RAMs are cleared, one word per cycle, and `ready`
// stays low meanwhile. The template RAM needs no clearing: the `loaded` flag
// forces a load before its first use.
//
// The reference for every output is Python: neurochain.rtl.spec.ChainSpec.reference.
`default_nettype none

module nc_sbp_chain #(
    parameter int N_CH = 96,  // channels sharing the datapath
    parameter int CODE_W = 8,  // ADC resolution
    parameter int N_TAPS = 16,  // template length in samples (>= 2)
    parameter int TPL_FRAC = 6,  // fractional bits of the stored template
    parameter int OUT_FRAC = 2,  // fractional bits kept after the canceller
    parameter int ADAPT_SHIFT = 4,  // template step size 2^-ADAPT_SHIFT
    parameter int COEF_FRAC = 8,  // fractional bits of the filter coefficients
    parameter int B0 = 129,  // filter coefficients, scaled by 2^COEF_FRAC
    parameter int A1 = -191,
    parameter int A2 = 70,
    parameter int Y_W = 12,  // filter output width (worst case, see spec.py)
    parameter int BIN_LEN = 100,  // samples per feature bin
    parameter int SUM_W = 18,  // bin accumulator width (worst case)
    parameter bit SHIFT_ADD = 1'b1,  // filter products as shifts and adds (0: "*")
    localparam int CH_W = (N_CH > 1) ? $clog2(N_CH) : 1,
    localparam int C_W = CODE_W + OUT_FRAC + 1,
    localparam int CNT_W = $clog2(BIN_LEN + 1)
) (
    input  wire                      clk,
    input  wire                      rst,        // synchronous, active high
    output logic                     ready,      // low while the state RAMs are cleared
    // one sample of one channel
    input  wire                      in_valid,
    input  wire         [  CH_W-1:0] in_ch,
    input  wire signed  [CODE_W-1:0] in_code,
    input  wire                      in_trig,    // a pulse starts on this sample
    // one feature value per channel per bin
    output logic                     bin_valid,
    output logic        [  CH_W-1:0] bin_ch,
    output logic        [ SUM_W-1:0] bin_sum,    // sum of |y| over the unblanked samples
    output logic        [ CNT_W-1:0] bin_cnt,    // how many samples were not blanked
    // per-sample taps, for verification against the Python reference
    output logic                     smp_valid,
    output logic        [  CH_W-1:0] smp_ch,
    output logic signed [   C_W-1:0] smp_c,      // canceller output
    output logic                     smp_ok,     // 0 when the sample was blanked
    output logic signed [   Y_W-1:0] smp_y       // filter output
);
  localparam int X_W = CODE_W + TPL_FRAC;
  localparam int K_W = $clog2(N_TAPS);
  localparam int CTL_W = K_W + 1;
  localparam int FLT_W = 2 * C_W + 2 * Y_W;
  localparam int BIN_W = SUM_W + 2 * CNT_W;
  localparam logic [K_W-1:0] TAPS = N_TAPS[K_W-1:0];  // N_TAPS modulo 2^K_W
  localparam logic [K_W-1:0] LAST_TAP = K_W'(N_TAPS - 1);
  localparam logic [CH_W-1:0] LAST_CH = CH_W'(N_CH - 1);
  localparam logic [CNT_W-1:0] LAST_SAMPLE = CNT_W'(BIN_LEN - 1);

  // ------------------------------------------------------------ reset and clear
  logic            clearing;
  logic [CH_W-1:0] clr_ch;

  always_ff @(posedge clk) begin
    if (rst) begin
      clearing <= 1'b1;
      clr_ch   <= '0;
    end else if (clearing) begin
      clearing <= (clr_ch != LAST_CH);
      clr_ch   <= clr_ch + 1'b1;
    end
  end

  assign ready = !clearing;
  wire in_fire = in_valid && ready && !rst;

  // -------------------------------------------------- edge 0: accept, read ctl
  logic                     s1_valid;
  logic        [  CH_W-1:0] s1_ch;
  logic signed [CODE_W-1:0] s1_code;
  logic                     s1_trig;
  logic        [ CTL_W-1:0] ctl_q;

  always_ff @(posedge clk) begin
    s1_valid <= in_fire;
    if (in_fire) begin
      s1_ch   <= in_ch;
      s1_code <= in_code;
      s1_trig <= in_trig;
    end
  end

  // --------------------------------------- S1: where in the pulse window are we
  wire  [K_W-1:0] s1_left = ctl_q[CTL_W-1:1];  // taps of the window still to come
  wire            s1_loaded = ctl_q[0];
  wire            s1_in_window = s1_trig || (s1_left != '0);
  wire  [K_W-1:0] s1_k = s1_trig ? '0 : TAPS - s1_left;
  wire  [K_W-1:0] s1_left_next = s1_in_window ? LAST_TAP - s1_k : '0;
  wire            s1_loaded_next = s1_loaded || (s1_in_window && (s1_k == LAST_TAP));

  nc_sdp_ram #(
      .DEPTH(N_CH),
      .WIDTH(CTL_W)
  ) u_ctl (
      .clk  (clk),
      .re   (in_fire),
      .raddr(in_ch),
      .rdata(ctl_q),
      .we   (clearing || s1_valid),
      .waddr(clearing ? clr_ch : s1_ch),
      .wdata(clearing ? '0 : {s1_left_next, s1_loaded_next})
  );

  logic                     s2_valid;
  logic        [  CH_W-1:0] s2_ch;
  logic signed [CODE_W-1:0] s2_code;
  logic                     s2_in_window;
  logic                     s2_load;
  logic        [   K_W-1:0] s2_k;

  always_ff @(posedge clk) begin
    s2_valid <= s1_valid && !rst;
    if (s1_valid) begin
      s2_ch        <= s1_ch;
      s2_code      <= s1_code;
      s2_in_window <= s1_in_window;
      s2_load      <= s1_in_window && !s1_loaded;
      s2_k         <= s1_k;
    end
  end

  // ------------------------------------------------------------- S2: canceller
  logic signed [X_W-1:0] tpl_q, tpl_next;
  logic signed [C_W-1:0] c2;
  logic                  ok2;

  // Addressed by {channel, tap}, so its depth is a power of two.
  nc_sdp_ram #(
      .DEPTH(1 << (CH_W + K_W)),
      .WIDTH(X_W)
  ) u_tpl (
      .clk  (clk),
      .re   (s1_valid && s1_in_window),
      .raddr({s1_ch, s1_k}),
      .rdata(tpl_q),
      .we   (s2_valid && s2_in_window),
      .waddr({s2_ch, s2_k}),
      .wdata(tpl_next)
  );

  nc_cancel_core #(
      .CODE_W     (CODE_W),
      .TPL_FRAC   (TPL_FRAC),
      .OUT_FRAC   (OUT_FRAC),
      .ADAPT_SHIFT(ADAPT_SHIFT)
  ) u_cancel (
      .code     (s2_code),
      .tpl      (tpl_q),
      .in_window(s2_in_window),
      .load     (s2_load),
      .out      (c2),
      .ok       (ok2),
      .tpl_next (tpl_next)
  );

  logic                  s3_valid;
  logic        [CH_W-1:0] s3_ch;
  logic signed [ C_W-1:0] s3_c;
  logic                  s3_ok;

  always_ff @(posedge clk) begin
    s3_valid <= s2_valid && !rst;
    if (s2_valid) begin
      s3_ch <= s2_ch;
      s3_c  <= c2;
      s3_ok <= ok2;
    end
  end

  // ------------------------------------------------------------- S3: high-pass
  logic        [FLT_W-1:0] flt_q;
  logic signed [  Y_W-1:0] y3;
  wire signed  [  C_W-1:0] x1 = flt_q[FLT_W-1-:C_W];
  wire signed  [  C_W-1:0] x2 = flt_q[2*Y_W+:C_W];
  wire signed  [  Y_W-1:0] y1 = flt_q[Y_W+:Y_W];
  wire signed  [  Y_W-1:0] y2 = flt_q[0+:Y_W];

  nc_sdp_ram #(
      .DEPTH(N_CH),
      .WIDTH(FLT_W)
  ) u_flt (
      .clk  (clk),
      .re   (s2_valid),
      .raddr(s2_ch),
      .rdata(flt_q),
      .we   (clearing || s3_valid),
      .waddr(clearing ? clr_ch : s3_ch),
      .wdata(clearing ? '0 : {s3_c, x1, y3, y1})
  );

  nc_highpass_core #(
      .C_W      (C_W),
      .Y_W      (Y_W),
      .COEF_FRAC(COEF_FRAC),
      .B0       (B0),
      .A1       (A1),
      .A2       (A2),
      .SHIFT_ADD(SHIFT_ADD)
  ) u_highpass (
      .x (s3_c),
      .x1(x1),
      .x2(x2),
      .y1(y1),
      .y2(y2),
      .y (y3)
  );

  logic                  s4_valid;
  logic        [CH_W-1:0] s4_ch;
  logic signed [ C_W-1:0] s4_c;
  logic                  s4_ok;
  logic signed [ Y_W-1:0] s4_y;

  always_ff @(posedge clk) begin
    s4_valid <= s3_valid && !rst;
    if (s3_valid) begin
      s4_ch <= s3_ch;
      s4_c  <= s3_c;
      s4_ok <= s3_ok;
      s4_y  <= y3;
    end
  end

  // ------------------------------------------------- S4: magnitude, accumulate
  logic [BIN_W-1:0] bin_q;
  wire  [SUM_W-1:0] sum = bin_q[BIN_W-1-:SUM_W];
  wire  [CNT_W-1:0] cnt = bin_q[CNT_W+:CNT_W];
  wire  [CNT_W-1:0] n = bin_q[0+:CNT_W];

  wire  [  Y_W-1:0] mag = s4_y[Y_W-1] ? -s4_y : s4_y;
  wire  [SUM_W-1:0] sum_next = sum + (s4_ok ? SUM_W'(mag) : '0);
  wire  [CNT_W-1:0] cnt_next = cnt + CNT_W'(s4_ok);
  wire              bin_end = (n == LAST_SAMPLE);

  nc_sdp_ram #(
      .DEPTH(N_CH),
      .WIDTH(BIN_W)
  ) u_bin (
      .clk  (clk),
      .re   (s3_valid),
      .raddr(s3_ch),
      .rdata(bin_q),
      .we   (clearing || s4_valid),
      .waddr(clearing ? clr_ch : s4_ch),
      .wdata((clearing || bin_end) ? '0 : {sum_next, cnt_next, n + 1'b1})
  );

  always_ff @(posedge clk) begin
    bin_valid <= s4_valid && bin_end && !rst;
    if (s4_valid && bin_end) begin
      bin_ch  <= s4_ch;
      bin_sum <= sum_next;
      bin_cnt <= cnt_next;
    end
  end

  assign smp_valid = s4_valid;
  assign smp_ch    = s4_ch;
  assign smp_c     = s4_c;
  assign smp_ok    = s4_ok;
  assign smp_y     = s4_y;

`ifndef SYNTHESIS
  // The spacing rule, checked in simulation.
  always_ff @(posedge clk) begin
    if (in_fire && s1_valid && (in_ch == s1_ch)) begin
      $error("nc_sbp_chain: channel %0d presented on consecutive clock edges", in_ch);
    end
  end
`endif
endmodule

`default_nettype wire
