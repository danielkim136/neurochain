// Simple dual-port RAM with a registered read: one write port, one read port.
//
// This is the form every FPGA block RAM and every compiled ASIC SRAM offers:
// the address is sampled on the clock edge and the data appears after it. The
// chain never reads and writes the same address on the same edge (see the
// spacing rule in nc_sbp_chain.sv), so the read-during-write behaviour, which
// differs between RAM technologies, is never exercised.
//
// Contents are not reset. nc_sbp_chain clears the words that need it.
//
// (* no_rw_check *) tells Yosys the same thing in its own terms: do not add
// bypass registers and comparators to emulate a particular collision result.
// Without it an early ECP5 build spent about 200 extra flip-flops on a case
// that cannot happen.
`default_nettype none

module nc_sdp_ram #(
    parameter int DEPTH = 96,
    parameter int WIDTH = 8,
    localparam int A_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  wire              clk,
    input  wire              re,
    input  wire  [  A_W-1:0] raddr,
    output logic [WIDTH-1:0] rdata,
    input  wire              we,
    input  wire  [  A_W-1:0] waddr,
    input  wire  [WIDTH-1:0] wdata
);
  (* no_rw_check *)
  logic [WIDTH-1:0] mem[0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    if (re) rdata <= mem[raddr];
  end
endmodule

`default_nettype wire
