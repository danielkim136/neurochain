// Template artifact canceller, the arithmetic for one sample (combinational).
//
// Reference: neurochain.stim.cancel.cancel_template (Python, integers).
//
//   X      = code << TPL_FRAC                 the sample at the template's scale
//   d      = X - T                            error against the stored template
//   out    = (d + half) >> (TPL_FRAC-OUT_FRAC)   round half up, keep OUT_FRAC bits
//   T_next = T + (d >> ADAPT_SHIFT)           LMS update with step 2^-ADAPT_SHIFT
//
// Outside a pulse window the sample passes through, shifted left by OUT_FRAC so
// that both cases share one scale. In the first window after reset the template
// is loaded with X and the output is blanked. A sample on either ADC rail is
// blanked too (the artifact there is unknown), but the template still adapts.
//
// Widths (CODE_W = 8, TPL_FRAC = 6, OUT_FRAC = 2):
//   X, T   14 bits signed. T never leaves [X_min, X_max]: each update moves it
//          a fraction 2^-ADAPT_SHIFT of the way toward X and never past it.
//   d      15 bits signed, |d| <= (2^8 - 1) * 2^6 = 16320.
//   out    11 bits signed, |out| <= (16320 + 8) >> 4 = 1020.
// ">>>" on a signed value floors, as Python's ">>" does on a negative int.
`default_nettype none

module nc_cancel_core #(
    parameter int CODE_W = 8,
    parameter int TPL_FRAC = 6,
    parameter int OUT_FRAC = 2,
    parameter int ADAPT_SHIFT = 4,
    localparam int X_W = CODE_W + TPL_FRAC,
    localparam int C_W = CODE_W + OUT_FRAC + 1
) (
    input  wire signed  [CODE_W-1:0] code,      // ADC code
    input  wire signed  [   X_W-1:0] tpl,       // stored template tap T[k]
    input  wire                      in_window, // this sample is within N_TAPS of a pulse
    input  wire                      load,      // first window: load the template, blank
    output logic signed [   C_W-1:0] out,       // cancelled sample, scaled by 2^OUT_FRAC
    output logic                     ok,        // 0 when the sample is blanked
    output logic signed [   X_W-1:0] tpl_next   // value to write back to T[k]
);
  localparam int DROP = TPL_FRAC - OUT_FRAC;
  localparam int D_W = X_W + 1;
  localparam logic signed [D_W-1:0] HALF = (DROP > 0) ? D_W'(1) << (DROP - 1) : '0;
  localparam logic signed [CODE_W-1:0] CODE_MIN = {1'b1, {(CODE_W - 1) {1'b0}}};
  localparam logic signed [CODE_W-1:0] CODE_MAX = {1'b0, {(CODE_W - 1) {1'b1}}};

  logic signed [X_W-1:0] x_scaled;
  logic signed [D_W-1:0] d, rounded, step;
  logic signed [X_W-1:0] updated;
  logic signed [C_W-1:0] error, pass;
  logic                  clipped;

  always_comb begin
    x_scaled = {code, {TPL_FRAC{1'b0}}};
    d        = {x_scaled[X_W-1], x_scaled} - {tpl[X_W-1], tpl};
    rounded  = (d + HALF) >>> DROP;
    error    = rounded[C_W-1:0];
    step     = d >>> ADAPT_SHIFT;
    updated  = tpl + step[X_W-1:0];
    clipped  = (code == CODE_MIN) || (code == CODE_MAX);
    pass     = {code[CODE_W-1], code, {OUT_FRAC{1'b0}}};

    if (!in_window) begin
      out      = pass;
      ok       = 1'b1;
      tpl_next = tpl;
    end else if (load) begin
      out      = '0;
      ok       = 1'b0;
      tpl_next = x_scaled;
    end else begin
      out      = clipped ? '0 : error;
      ok       = !clipped;
      tpl_next = updated;
    end
  end
endmodule

`default_nettype wire
