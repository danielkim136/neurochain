// Multiply by a constant with shifts and adds only, modulo 2^W.
//
// The coefficient is recoded at elaboration into canonical signed digits
// (CSD): digits +1, 0, -1 with no two neighbours non-zero, which is the form
// with the fewest non-zero digits. Each non-zero digit costs one adder:
//
//   129 = 128 + 1          two terms
//   191 = 256 - 64 - 1     three terms (plain binary 10111111 would take seven)
//    70 =  64 + 8 - 2      three terms
//
// The recoding uses the identity n = P - M with
//   h = n >> 1,  t = n + h (= 3n/2),  c = h ^ t,  P = t & c,  M = h & c.
// SHIFT_ADD = 0 writes a plain "*" instead, for comparison in synthesis; both
// give the same product bit for bit.
`default_nettype none

module nc_const_mult #(
    parameter int W = 20,  // width of the input, already sign-extended, and of the product
    parameter int COEF = 129,  // signed integer constant, |COEF| < 2^29
    parameter bit SHIFT_ADD = 1'b1
) (
    input  wire signed  [W-1:0] x,
    output logic signed [W-1:0] p
);
  localparam int MAG = (COEF < 0) ? -COEF : COEF;
  localparam int HALF = MAG >> 1;
  localparam int THREE_HALVES = MAG + HALF;
  localparam int CARRY = HALF ^ THREE_HALVES;
  localparam int PLUS_MAG = THREE_HALVES & CARRY;
  localparam int MINUS_MAG = HALF & CARRY;
  // A negative coefficient swaps the roles of the two digit sets.
  localparam int PLUS = (COEF < 0) ? MINUS_MAG : PLUS_MAG;
  localparam int MINUS = (COEF < 0) ? PLUS_MAG : MINUS_MAG;
  localparam logic signed [W-1:0] COEF_W = COEF[W-1:0];

  generate
    if (SHIFT_ADD) begin : g_shift_add
      always_comb begin
        p = '0;
        for (int i = 0; i < 31; i++) begin
          if (PLUS[i]) p = p + (x <<< i);
          if (MINUS[i]) p = p - (x <<< i);
        end
      end
    end else begin : g_star
      assign p = COEF_W * x;
    end
  endgenerate
endmodule

`default_nettype wire
