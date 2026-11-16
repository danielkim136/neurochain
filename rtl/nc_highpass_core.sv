// Second-order high-pass, the arithmetic for one sample (combinational).
//
// Reference: neurochain.fixedpoint.fixed_highpass with guard_bits = 0 (the
// canceller's output already carries the two guard bits).
//
//   s   = x - 2*x1 + x2                      numerator (1 - z^-1)^2, exact
//   acc = B0*s - A1*y1 - A2*y2               coefficients scaled by 2^COEF_FRAC
//   y   = (acc + 2^(COEF_FRAC-1)) >> COEF_FRAC    round half up
//
// Widths (C_W = 11, COEF_FRAC = 8, Y_W = 12):
//   s    13 bits signed, |s| <= 4 * 1020.
//   y    Y_W bits. Worst case |y| <= ||h||_1 * 1020 + ||1/A||_1 / 2 = 1542,
//        from the impulse response of the quantised filter
//        (neurochain.rtl.spec.ChainSpec.y_bound).
//   acc  Y_W + COEF_FRAC bits. The three products can each exceed that, but
//        their sum cannot, and two's-complement addition is exact modulo
//        2^ACC_W: overflow in a partial sum cancels as long as the final sum
//        fits. So every operand is extended to ACC_W and nothing wider is built.
//
// The three products are constant multiplications, built from shifts and adds
// (nc_const_mult): eight adders in place of three multipliers.
`default_nettype none

module nc_highpass_core #(
    parameter int C_W = 11,
    parameter int Y_W = 12,
    parameter int COEF_FRAC = 8,
    parameter int B0 = 129,
    parameter int A1 = -191,
    parameter int A2 = 70,
    parameter bit SHIFT_ADD = 1'b1  // 0: use "*" for the three products
) (
    input  wire signed  [C_W-1:0] x,   // input sample
    input  wire signed  [C_W-1:0] x1,  // x[n-1]
    input  wire signed  [C_W-1:0] x2,  // x[n-2]
    input  wire signed  [Y_W-1:0] y1,  // y[n-1]
    input  wire signed  [Y_W-1:0] y2,  // y[n-2]
    output logic signed [Y_W-1:0] y    // y[n]
);
  localparam int ACC_W = Y_W + COEF_FRAC;
  localparam int S_W = C_W + 2;
  localparam logic signed [ACC_W-1:0] ROUND = ACC_W'(1) << (COEF_FRAC - 1);

  logic signed [S_W-1:0] s;
  logic signed [ACC_W-1:0] s_ext, y1_ext, y2_ext, acc, rounded;
  logic signed [ACC_W-1:0] p_b0, p_a1, p_a2;

  nc_const_mult #(
      .W(ACC_W),
      .COEF(B0),
      .SHIFT_ADD(SHIFT_ADD)
  ) u_b0 (
      .x(s_ext),
      .p(p_b0)
  );
  nc_const_mult #(
      .W(ACC_W),
      .COEF(A1),
      .SHIFT_ADD(SHIFT_ADD)
  ) u_a1 (
      .x(y1_ext),
      .p(p_a1)
  );
  nc_const_mult #(
      .W(ACC_W),
      .COEF(A2),
      .SHIFT_ADD(SHIFT_ADD)
  ) u_a2 (
      .x(y2_ext),
      .p(p_a2)
  );

  always_comb begin
    s       = {{2{x[C_W-1]}}, x} - {x1[C_W-1], x1, 1'b0} + {{2{x2[C_W-1]}}, x2};
    s_ext   = ACC_W'(s);
    y1_ext  = ACC_W'(y1);
    y2_ext  = ACC_W'(y2);
    acc     = p_b0 - p_a1 - p_a2;
    rounded = acc + ROUND;
    y       = rounded[ACC_W-1:COEF_FRAC];
  end
endmodule

`default_nettype wire
