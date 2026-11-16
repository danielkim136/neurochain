# U7 design: the SBP chain in SystemVerilog

U3 chose a feature path, U5 put an artifact canceller in front of it, and both
exist as integer Python. U7 turns that Python into hardware: SystemVerilog that
a synthesiser accepts, proven equal to the Python on every bit of ten minutes
of real recordings, built for a Lattice ECP5, and costed in µW per channel.

**Done-criterion:** bit-exact on 10 minutes of real data; LUTs, fmax and
estimated µW per channel reported. **Status:** met. Zero mismatches in
347,904,000 compared values; 489 LUTs, 49.8 MHz, 0.012 µW per channel (an
estimate that rests on stated assumptions, section 8).

Code: `rtl/*.sv` (the hardware), `rtl/tb/tb_chain.cpp` (the Verilator
testbench), `src/neurochain/rtl/` (specification, stimulus, tool drivers,
energy model), `src/neurochain/analysis/u7.py` (the experiment). One command:
`make u7` (about two minutes). Numbers: `results/u7_metrics.json`. Everything
the tools write goes to `build/u7`, which git ignores.

## 1. What U7 has to answer

1. Does the hardware compute exactly what the Python computes? Not "close":
   the same integers, on every sample.
2. How big is it (LUTs, flip-flops, memory) and how fast can it be clocked?
3. What does it cost in power, and does that change U3's 0.48 µW budget? U3
   charged 1 pJ per multiply-accumulate and called it a placeholder "that U7
   replaces with a synthesised estimate".

The chain, per channel:

```
8-bit code, 2 kS/s
   -> template canceller   e = x - T[k];  T[k] += (x - T[k]) >> 4       (U5)
   -> 300 Hz high-pass     y = (129 s - (-191) y1 - 70 y2 + 128) >> 8   (U2)
   -> |y|, summed over 100 samples (50 ms), with a count of unblanked samples
```

## 2. Toolchain

| Tool | Version | How it was installed |
|---|---|---|
| Verilator | 5.054 | `brew install verilator` |
| Yosys | 0.71 | `brew install yosys` |
| Project Trellis (ECP5 database, `ecppack`) | 1.4 | `brew install prjtrellis` |
| nextpnr-ecp5 | 0.11.1 | built from source, see below |

Homebrew has no `nextpnr-ecp5` formula, only `nextpnr-ice40`. The ECP5 build
is the same source with a different switch, so it was built the way that
formula builds the iCE40 one: the `nextpnr-0.11.1` source archive (its SHA-256
matches the one in Homebrew's formula), `cmake -DARCH=ecp5
-DTRELLIS_INSTALL_PREFIX=/opt/homebrew`, installed to `~/.local/bin`. The
binary built from an archive does not report its version, so
`results/u7_metrics.json` records it as unknown. `neurochain.rtl.tools.find_tool`
looks on `PATH`, then in `~/.local/bin`, and `NEUROCHAIN_NEXTPNR_ECP5` overrides
both. The alternative was the prebuilt OSS CAD Suite: one download, but a
second copy of Yosys and Verilator beside Homebrew's.

Three tool quirks cost time and are worth knowing:

- Yosys 0.71's `hierarchy -chparam` trips an internal assertion on this
  design. `chparam -set`, then `hierarchy -auto-top`, then `rename -top` does
  the same job (`tools._read_sources`).
- Yosys does not accept a negative decimal as a parameter value. `A1 = -191`
  is passed as 32 bits of two's complement.
- Verilator does not mask a value written to a port narrower than its C++
  type. The testbench has to (section 7).

## 3. Decision 1: which real data gives ten minutes

| Candidate | Length | Why not, or why |
|---|---|---|
| Nason in-vivo broadband | 10 s, one channel | Sixty times too short |
| MEArec | 60 s, 32 channels | A simulation, and too short |
| NoMAD, Nason online data | long | Already features (bins); no samples to filter |
| **Neuralink clips** | 743 clips of about 5 s, 62.6 minutes | **Chosen** |

The Neuralink clips have no voltage scale, which is why U1 onward could not
use them. Here it does not matter. Digital logic sees integers, and
bit-exactness is a statement about integers: the proof is as strong in LSB as
it would be in microvolts. What does matter is that the codes look like what
the 0.48 µW chain would produce, so that the comparison exercises realistic
values. `neurochain.rtl.stimulus` builds them like this:

1. **Band.** Each clip's 10-bit codes pass a causal second-order Butterworth
   band-pass, 300 Hz to 1 kHz: a stand-in for U3's narrowband amplifier. The
   filter starts in steady state for the clip's first sample, so the clip's DC
   level causes no start-up transient.
2. **Sampling.** The result is sampled at 2 kS/s by linear interpolation
   between the 19.5 kS/s samples. Nothing else limits the band, so whatever
   the band-pass leaves above 1 kHz aliases, as it would in the implant.
3. **Scale.** Divide by 4: the same converter range digitised with 8 bits
   instead of 10. No gain is invented. The result has an RMS of 1.55 LSB
   outside stimulation, close to what U5 measured for the real chain's signal
   (a mean SBP of about 1.3 LSB there, 1.06 LSB here).
4. **Channels.** Each of 96 channels is its own seeded shuffle of the clips,
   joined end to end (about 119 clips per channel). 96 channels times ten
   minutes is 16 channel-hours made from 62.6 minutes of recordings, so each
   clip appears in about 15 channels, at different times.
5. **Stimulation.** U5's schedule (trains of 0.5 to 1.5 s at 100 pulses per
   second, pauses of the same length, pulses on the ADC clock) and U5's pulse
   responses of the same front end, converted to LSB with the chain's nominal
   7.8 µV step. They are added before rounding and the sum is clipped to the
   converter's range, so the 10 mV and larger artifacts rail the ADC as they
   would. Every train draws a new artifact size for every channel from U5's
   grid (0.1 to 100 mV).

Step 5 is deliberately harsher than U5's analysis, where a channel's artifact
never changed. A template that has to re-adapt at every train produces large
canceller outputs, which is what tests word widths. It is a stress, not a
claim about how stimulation is modulated.

The record: 30,105 pulses in 298 trains, stimulation on 50 % of the time,
3.20 % of samples on a rail (and therefore blanked). Seed 0 fixes everything.

*Alternatives for the resampling.* A polyphase resampler
(`scipy.signal.resample_poly`) would add a near-ideal anti-alias filter the
implant does not have. Keeping every tenth sample would give 1953 S/s, not
2000. Interpolating the band-limited signal mirrors a sample-and-hold after
the amplifier.

*Course link, ELEC3104:* sampling and aliasing; the band-pass is the only
anti-alias filter, as in the analog chain.

## 4. Decision 2: one shared datapath, state in RAM

### 4.1 Shared against per channel

At 2 kS/s a channel needs its arithmetic for a few nanoseconds every 500 µs.
Giving each channel its own adders leaves them idle 99.99 % of the time.

| Option | Measured on the ECP5 | Verdict |
|---|---|---|
| A datapath per channel | the one-channel build is 377 LUTs and 276 flip-flops; 96 of them is 36,192 LUTs | Does not fit the LFE5U-25F (24,288) |
| **One datapath, channels in turn** | 489 LUTs, 137 flip-flops, 5 block RAMs for 96 channels | **Chosen** |

The shared datapath barely grows with the channel count: 449 LUTs for 8
channels, 489 for 96, 471 for 1024, 489 for 2048. Only the memory grows.

### 4.2 Registers against RAM for the state

A channel's state is 307 bits:

| Memory | Per channel | Contents |
|---|---|---|
| control | 5 bits | taps left in the pulse window (4), template loaded (1) |
| template | 224 bits | 16 taps of 14 bits |
| filter | 46 bits | `x1`, `x2` (11 bits each), `y1`, `y2` (12 bits each) |
| bin | 32 bits | sum (18), valid count (7), sample count (7) |

For 96 channels that is 29,472 bits. As flip-flops it would be more than the
FPGA has (24,288), each with a multiplexer in front. As RAM it is 5 of the
ECP5's 56 block RAMs (the control word is small enough that Yosys puts it in
LUT RAM). Four separate memories rather than one 307-bit word, because each
stage needs only its own part, and the template is addressed by
`{channel, tap}` so that one tap is read per sample, not sixteen.

### 4.3 The pipeline

One channel's sample enters per clock cycle. Each stage reads the word it
needs from its RAM and writes it back one cycle later:

```
edge 0  sample accepted          read control[ch]
edge 1  S1  tap index k          write control, read template[ch, k]
edge 2  S2  canceller            write template, read filter[ch]
edge 3  S3  high-pass            write filter, read bin[ch]
edge 4  S4  |y|, accumulate      write bin; per-sample outputs valid
edge 5  bin outputs valid (every 100th sample of a channel)
```

**Spacing rule.** A RAM read and the write of the same word are one clock edge
apart, so the same channel must not be presented on two consecutive edges.
With two or more channels taken in turn that never happens, and no cycle is
idle: the clock can be exactly channels × 2 kHz, 192 kHz for 96 channels. A
simulation-only assertion in `nc_sbp_chain.sv` checks the rule. Because of it
the RAM never reads and writes one address on the same edge, which matters:
what a RAM returns in that case differs between technologies. Telling Yosys so
(`(* no_rw_check *)`) removed about 200 flip-flops of bypass logic it had
added to emulate one particular answer (352 flip-flops against 137 in an early
build).

*Alternative: a sequential machine* with one adder used eight or ten times per
sample. Smaller still, but it needs a clock ten times faster, and clock power
is paid per cycle.

**Reset.** RAM cannot be reset in one cycle. After reset a counter writes zero
to the control, filter and bin words, one channel per cycle, and `ready` stays
low meanwhile. Zero is the right starting state for all three (no window open,
template not loaded, filter at rest, empty bin). The template RAM is not
cleared: the "loaded" flag forces every tap to be written before it is read.

*Course link, ELEC2141:* datapath and control, pipelining, synchronous RAM,
resource sharing by time multiplexing.

## 5. Decision 3: word widths, and why nothing overflows

Python integers grow as needed; a register does not. Every width is derived in
`neurochain.rtl.spec.ChainSpec` from the parameters, as a worst-case bound, and
passed to the RTL as a parameter.

| Node | Bound | Width | Peak on the real data | Peak on the worst-case stimulus |
|---|---|---|---|---|
| ADC code | −128 to 127 | 8 signed | 127 (8 bits) | 127 |
| template tap `T` | −8192 to 8128 | 14 signed | internal | internal |
| canceller output | ±1016 | 11 signed | 650 (11 bits) | 1016 |
| filter accumulator | ±393,344 | 20 signed | 157,497 (19 bits) | 393,066 |
| filter output `y` | ±1536 | 12 signed | 615 (11 bits) | 1535 |
| bin sum | 153,600 | 18 unsigned | 9121 (14 bits) | 27,074 |
| bin counts | 100 | 7 unsigned | | |

The arguments, in order:

- **Template.** `T ← T + ((X − T) >> 4)` moves `T` a sixteenth of the way
  toward `X = x·64` and, because the shift floors, never past it. `T` starts
  as some `X`, so it stays inside the range of `X`: 14 bits.
- **Canceller output.** `(X − T + 8) >> 4` with `T` anywhere in range and `x`
  one step inside the rails (a railed code is blanked): at most
  `(254·64 + 8) >> 4 = 1016`. Outside a window the output is `x·4`, at most
  512.
- **Filter output.** For a linear filter with impulse response `h`, a bounded
  input `|x| ≤ X` gives `|y| ≤ X·Σ|h[n]|`, and the input that follows the signs
  of `h` backwards reaches it. For the quantised filter `Σ|h| = 1.511`.
  Rounding adds an error of at most half a step inside the recursion every
  sample, which the recursion `1/A(z)` can amplify by `Σ|g| = 2.199`. So
  `|y| ≤ 1.511·1016 + 2.199/2 = 1536`: 12 bits.
- **Accumulator.** `acc = 256·y` before rounding, so 20 bits. The three
  products are individually larger (`129·s` alone can reach 524,000), but
  two's-complement addition is exact modulo 2²⁰: an overflow in a partial sum
  cancels when the final sum fits. Nothing wider than 20 bits is built.
- **Bin sum.** 100 samples of at most 1536: 18 bits.

Two things came out of this:

1. **U2's "18-bit accumulator" is not enough any more.** U2 measured 18 bits
   on noise with no canceller in front. With the canceller re-adapting to a new
   artifact size the real data reaches 19 bits, and the worst case needs 20.
   Prediction check 1 builds the 18-bit version and counts what breaks.
2. **A worst-case test is cheap and worth having.**
   `stimulus.worst_case_stimulus` loads the template with one rail and then
   presents the other, with signs following the filter's impulse response
   backwards. The Python reference reaches 1016 at the canceller (the bound)
   and 1535 at the filter (bound 1536), and the RTL matches it on all 185,472
   values.

*Course link, ELEC2141:* two's complement, sign extension, modular addition.
*ELEC3104:* BIBO stability as `Σ|h| < ∞`; the same sum sizes the register.
*MATH2099:* the triangle inequality behind `|Σ h·x| ≤ Σ|h|·max|x|`.

## 6. Decision 4: three multiplications without a multiplier

The coefficients are constants, so each product is a fixed set of shifts and
adds. `rtl/nc_const_mult.sv` recodes the constant at elaboration into
canonical signed digits (digits +1, 0, −1, no two neighbours non-zero), the
form with the fewest non-zero digits:

```
129 = 128 + 1          two terms
191 = 256 - 64 - 1     three terms   (binary 10111111 would take seven)
 70 =  64 + 8 - 2      three terms
```

Eight shifted terms and the rounding constant, summed. The recoding is four
lines of constant arithmetic (`h = n >> 1; t = n + h; c = h ^ t; P = t & c;
M = h & c`), mirrored in Python by `spec.csd_digits`, which a test checks for
every integer up to 3000.

| How the products are written | LUT4 | Hard multipliers | fmax (MHz) |
|---|---|---|---|
| **shifts and adds (chosen)** | 489 | 0 | 49.8 |
| `*`, mapped to LUTs | 762 | 0 | 48.1 |
| `*`, mapped to the ECP5's 18×18 multipliers | 290 | 3 | 44.0 |

The hard multipliers give the smallest LUT count, and are the wrong thing to
report: an implant has none, and a LUT count that hides three multipliers says
little about the logic. The synthesis therefore runs with `-nodsp`. All three
versions pass the bit-exactness tests.

## 7. Decision 5: what "bit-exact" compares

**The specification is the Python.** `ChainSpec.reference` is three existing
functions in a row: `stim.cancel.cancel_template`, `fixedpoint.fixed_highpass`
with no guard bits (the canceller's two fractional bits take their place, as
U5 arranged), and a new `fixedpoint.fixed_masked_sbp`, the integer form of
U5's `masked_mean_abs`: a sum and a count instead of a mean, so the implant
needs no divider. A test ties it to the floating-point version.

**What is compared.** For every sample of every channel: the canceller's
output, its "not blanked" flag and the filter's output. For every bin: the sum
and the count. The testbench also checks that every output carries the channel
number it should. The criterion is zero differences, not a small error.

| Comparison | Values | Mismatches |
|---|---|---|
| RTL against Python, real data, 96 channels × 10 minutes | 347,904,000 | 0 |
| RTL against Python, worst-case stimulus | 185,472 | 0 |
| Gate-level netlist against RTL, first 20 s | 11,596,800 | 0 |

The simulation runs 115,200,108 clock cycles in about ten seconds; the Python
reference takes about as long on eight processes. The third row is a second
kind of proof: Yosys's NAND/NOR netlist, simulated again, gives the same
outputs as the RTL, so synthesis did not change the function.

**Where RTL and Python disagreed, and which was wrong:**

- *Overlapping pulse windows.* If a pulse arrives before the previous window
  has ended, the Python reprocesses the overlap and the RTL restarts the
  window. Neither is "right": `cancel_template` already says windows must not
  overlap. The specification now refuses such input (`ChainSpec.check_onsets`),
  and the RTL header states the rule (pulses at least 16 samples apart; U5
  uses 20).
- *Six-bit codes.* A parameter test with a 6-bit ADC failed on 1809 values.
  The RTL was right. The testbench wrote an 8-bit value into a 6-bit port and
  Verilator kept the two stray bits. Fixed in `tb_chain.cpp`.
- *One channel.* The first RTL sized the template RAM's address wrongly when
  the channel count was 1. Verilator's lint caught it before any simulation.

Tests in `tests/test_rtl.py` run the comparison on synthetic noise and
artifacts, on the worst case, with a deliberately narrow accumulator (which
must fail), and with four other parameter sets (one channel, 12 taps, other
fractional widths, 6 bits with `*`).

## 8. Decision 6: the power estimate

**An FPGA's power is not the implant's.** An ECP5 draws tens of milliwatts
standing still, in configuration memory, routing switches and I/O that a
custom chip does not have. No free tool estimates ECP5 power either. So the
ECP5 build answers "does it fit and how fast", and the power figure is an
estimate for an ASIC, built from what the free tools can measure:

| Measured | Value | Tool |
|---|---|---|
| Gates (NAND, NOR, NOT) | 2682 | Yosys, technology-neutral netlist |
| Flip-flops | 132 | Yosys |
| Node toggles per sample of one channel | 922 over 3329 nets (activity 0.28) | Verilator on that netlist, 20 s of the real data, SAIF |
| Memory bits read plus written per sample | 177 | counted from the RTL |
| State bits | 29,472 | counted from the RTL |

times energies that are **assumptions**, for a 65 nm low-power process:

| Assumption | Value | Where it comes from |
|---|---|---|
| Supply | 1.2 V | nominal for 65 nm; no voltage scaling assumed |
| Capacitance per node | 2 fF, so 1.44 fJ per toggle (`C·V²/2`) | order of magnitude for a small cell with a fan-out of two or three |
| Energy per memory bit accessed | 20 fJ | Meinerzhagen et al. (2012) measured 14 fJ per accessed bit on a 4 kbit standard-cell memory in 65 nm at 0.5 V |
| Leakage per gate and per bit | 5 pW | order of magnitude for high-threshold cells; the same paper measured 0.5 pW per bit at a 0.22 V retention supply |

Result, per sample of one channel: logic 1.33 pJ, clock 0.38 pJ (two edges
per flip-flop), state memory 3.54 pJ; 5.25 pJ in all. At 2 kS/s that is
0.0105 µW, plus 0.0017 µW of leakage: **0.012 µW per channel**.

Two checks on the constants:

- An independent count: the datapath does 18 additions per sample. At 0.1 pJ
  each (Horowitz, 2014, for a 32-bit add at 45 nm; quoted from memory) that is
  1.8 pJ, against 1.71 pJ for logic plus clock here.
- Every constant a tenth and ten times its value (U3's rule):

| Constant varied | Digital power, µW per channel |
|---|---|
| node capacitance | 0.009 to 0.043 |
| memory energy per bit | 0.006 to 0.076 |
| leakage | 0.011 to 0.028 |
| all three together | 0.0012 to 0.122 |

*Alternatives considered.* Scaling an FPGA power figure by the FPGA-to-ASIC
gap of Kuon and Rose (2007, about 14 times in dynamic power): there is no ECP5
power figure to scale. Mapping to an open cell library (SkyWater 130 nm) and
running a power tool: a real number, but for a process no implant would use,
and two more tools. Counting operations and multiplying by a published energy
per operation: used above as the cross-check.

*Course link, ELEC2141 and ELEC2133:* `P = a·C·V²·f`; here `a·f` is measured
and `C·V²` is assumed.

## 9. What the numbers say

### 9.1 The hardware computes what the Python computes

Zero mismatches in 347,904,000 values, on real recordings with stimulation
artifacts, clipped samples and a template that keeps re-adapting; zero on a
stimulus built to reach the registers' bounds; zero between the gate-level
netlist and the RTL. `figures/u7_bit_exact.png` shows 400 ms of one channel.

### 9.2 It is small, and timing is not the constraint

489 LUT4 (733 of the device's 24,288 logic cells once carry chains and
multiplexers are counted), 137 flip-flops and 5 of 56 block RAMs for 96
channels. The fastest clock is 49.8 MHz (48.3 to 50.9 over three placements);
96 channels need 192 kHz. That is 252 times more than needed. By timing alone
one datapath could serve about 24,000 channels; block RAM runs out first
(2048 channels use 39 of 56).

The slowest path starts at the filter-state RAM, goes through the high-pass
and returns to the same RAM: 19.6 ns, of which 5.8 ns is the block RAM's own
clock-to-output delay before any arithmetic. Asking nextpnr for 25, 100 or
200 MHz instead of 50 gave the same three results (tried once by calling
`tools.synthesise_ecp5` with `freq_mhz` changed; not part of `make u7`). A
register after the RAM would buy speed nobody needs.

In an ASIC that headroom is not wasted: it is what allows a lower supply
voltage, where logic is slower and each toggle costs less.

### 9.3 Storage, not arithmetic, is the digital cost

Of 5.25 pJ per sample, state memory is 3.54 pJ (67 %), logic 1.33 pJ and the
clock 0.38 pJ. U3 and U5 counted multiplies and adds; the filter's three
multiplications turned into eight adders, and reading and writing 177 bits of
state per sample costs more than all the arithmetic. This is the price of
sharing one datapath: the state has to be fetched and put back every sample.
It is also the least certain number (0.006 to 0.076 µW across the tenfold
range).

### 9.4 The 0.48 µW budget barely moves

| Stage | U3 + U5 model, µW | With U7's estimate, µW |
|---|---|---|
| amplifier | 0.450 | 0.450 |
| ADC | 0.026 | 0.026 |
| digital | 0.010 (5 operations × 1 pJ) | 0.012 |
| radio | 0.0024 (12 bits per bin) | 0.0050 (25 bits per bin) |
| **total** | **0.488** | **0.493** |

The placeholder was right to within a quarter, for the wrong reason: it
charged 5 pJ for arithmetic, and the estimate is 5.25 pJ, mostly memory. With
every assumed constant between a tenth and ten times its value the chain is
between 0.482 and 0.603 µW, and the amplifier stays 91 % of the central
figure. U3's conclusion survives: further savings need a better amplifier.

### 9.5 The radio word is wider than U3 assumed

U3 sends 12 bits per SBP value. The RTL produces an 18-bit sum and a 7-bit
count. On this data the mean sum is 579 (in quarter LSB), so 12 bits is
usually plenty, but 1.05 % of bins exceed 4095: the bins in which the template
is re-adapting. Sending all 25 bits costs 0.0026 µW more. What to send
(saturate, drop low bits, send the count only when it is not 100) is a design
choice U7 leaves open.

### What was surprising

- The accumulator width U2 handed over (18 bits) was a measurement on noise,
  and it is wrong once the canceller sits in front. Bounds, not measurements,
  size registers.
- The fastest clock is set by a block RAM's output delay, not by arithmetic.
- The digital power came out where the placeholder put it, with a completely
  different breakdown.

## 10. Limitations

- **No hardware.** Nothing ran on an FPGA. The comparison covers the RTL and
  Yosys's generic gate-level netlist; the placed-and-routed ECP5 netlist was
  not simulated. Timing comes from nextpnr's model of the device (speed grade
  6), with no pin constraints.
- **The front end is a stand-in.** The Neuralink clips went through a digital
  band-pass, not U1's amplifier model (they have no voltage scale). That
  affects how realistic the codes are, not the proof.
- **Clips are joined** about every 5 s, and each clip is used in about 15 of
  the 96 channels. The channels are not 96 independent electrodes.
- **Artifacts are simulated** (U5's model), and their size changes from train
  to train as a stress, not as a model of amplitude-modulated stimulation.
- **The power figure is an estimate.** All four energy constants are
  assumptions. The simulation has no gate delays, so glitches are not counted;
  in adder trees they can add tens of percent. The memory energy per bit is
  taken as independent of the memory's size, and is a 0.5 V measurement placed
  beside logic assumed at 1.2 V. The clock tree above the flip-flop pins, the
  ADC interface and the radio's framing are not included.
- **The Horowitz (2014) energies are quoted from memory.** The paper could not
  be opened here to check the table. They are used only as a cross-check.
- **One template per channel.** A stimulator that changes amplitude leaves a
  residual until the template has re-adapted (visible in
  `figures/u7_bit_exact.png`). U5 listed this; U7 did not change it.
- **Pulses must be at least 16 samples apart** (8 ms: at most 125 pulses per
  second with this template length).
- **U6's stabiliser is not built.** It has no integer reference yet; see
  "carried forward".
- **CI does not have the tools.** The tool tests skip there, as the ngspice
  tests would without ngspice; they have run only on this machine.

## 11. Prediction checks

Each changes one setting, rebuilds, and prints the new numbers. Changed runs
use 60 s of data per channel unless `--seconds` is given, and take one to
three minutes. Run them one at a time. Write down your prediction before
opening the answer.

For reference, the committed numbers: 0 mismatches in 347,904,000 values; 489
LUT4, 137 flip-flops, 5 block RAMs, 49.8 MHz; 922 node toggles per sample;
logic 1.33 pJ, clock 0.38 pJ, memory 3.54 pJ; 0.0122 µW per channel.

### Check 1: U2's 18-bit accumulator

```
make u7-key ARGS="--acc-bits 18 --seconds 600"
```

U2 measured that the 2 kS/s filter's accumulator needs 18 bits. The RTL uses
20. Build it with 18 (the filter's output then has 10 bits). The real data's
noise is 1.55 LSB RMS, far below any limit. Does the 18-bit version pass on
the ten minutes? If not, how many of the 347,904,000 values are wrong, in
which outputs, and how much hardware did the two bits cost?

<details>
<summary>Answer</summary>

**It fails: 32,474 values are wrong**, in all 96 channels. 31,321 filter
outputs and 1,153 bin sums differ; the canceller's outputs, the blanking flags
and the counts are all still correct, because the overflow is inside the
filter. The first wrong sample is on channel 0 at sample 32,203, 16.1 s in. On
the worst-case stimulus 48,698 of 185,472 values are wrong.

The noise is not what overflows. When a train starts with a different artifact
size than the template last learnt, the canceller's output jumps to hundreds
of LSB for a few pulses (the peak on this data is 650 quarter-LSB), and the
accumulator reaches 157,497, which needs 19 bits. U2's 18 bits were measured
without a canceller and without artifacts.

The saving: 440 LUT4 instead of 489, 135 flip-flops instead of 137, 303 state
bits per channel instead of 307. Ten percent of the logic for an error rate of
one value in ten thousand, concentrated exactly where stimulation starts:
a bad trade, and one that a test on noise alone would never show.

</details>

### Check 2: ten times the channels

```
make u7-key ARGS="--channels 1024"
```

One datapath, 1024 channels (a Neuralink-sized array). What happens to the
LUT count, the flip-flops, the block RAMs, the fastest clock, and the power
per channel?

<details>
<summary>Answer</summary>

**Almost nothing happens to the logic.** 471 LUT4 (from 489: slightly fewer,
because Yosys no longer builds the control word from LUT RAM), 150 flip-flops
(from 137: the channel number is three bits wider in each pipeline stage).
**The memory grows in proportion:** 20 block RAMs instead of 5, for 314,368
state bits. **The clock does not change:** 50.8 MHz (49.4 to 51.9), still set
by the block RAM's output delay; 1024 channels need 2.05 MHz.

Still bit-exact: 0 mismatches in 371,097,600 values (60 s per channel).

Power per channel in the model: 0.0120 µW, against 0.0122. The model charges
the same energy per memory bit whatever the memory's size, which is optimistic
for a memory ten times larger (longer bit lines); that is listed in the
limitations.

</details>

### Check 3: write `*` and let the tools decide

```
make u7-key ARGS="--star"
```

Replace the hand-recoded shifts and adds with a plain `*` for each of the
three products. The constants are still constants. Does the synthesiser find
the same adders? What happens to LUTs, gates and switching activity?

<details>
<summary>Answer</summary>

**No.** With `*` the ECP5 build needs **762 LUT4** instead of 489 (56 % more),
or 290 LUT4 plus 3 of the device's hard multipliers. The technology-neutral
netlist grows from 2682 to 3460 gates, and the node toggles per sample from
922 to 1124, so the logic energy rises from 1.33 to 1.62 pJ per sample.
Digital power in the model: 0.0128 µW instead of 0.0122.

It is still bit-exact (0 mismatches in 34,790,400 values on 60 s): the two
forms are the same function. Yosys builds a general multiplier and then
removes what the constant makes redundant, which does not recover the
signed-digit form: 191 stays seven partial products instead of three. On the
power budget the difference is invisible (memory dominates); on area it is
the largest single saving in the design.

</details>

### Check 4 (bonus): feed it silence

```
make u7-key ARGS="--silent"
```

Measure the activity with every code zero and no pulses. With nothing to
compute, how far does the digital power fall?

<details>
<summary>Answer</summary>

Node toggles fall from 922 to **22 per sample** (the channel counter, the bin
counters), and logic energy from 1.33 to 0.03 pJ. **Digital power in the model
falls only from 0.0122 to 0.0096 µW**, 21 %.

The clock (0.38 pJ) still reaches every flip-flop, the state is still read and
written (3.53 pJ: the model charges an access whatever the data), and leakage
does not care. Data-dependent switching is a quarter of the total. Saving the
rest needs gating: not clocking a stage that has nothing to do, and not
touching memory for a channel that is switched off. Neither is in this RTL.

</details>

## 12. What to read next

- `rtl/nc_sbp_chain.sv`: read the header comment, then follow one sample
  through S1 to S4 with `src/neurochain/stim/cancel.py` and
  `src/neurochain/fixedpoint.py` open beside it. Each stage is a few lines.
- `src/neurochain/rtl/spec.py`: every width in the RTL is a property here.
  Change `code_bits` to 10 and see which widths move.
- `rtl/nc_const_mult.sv`: fifteen lines that replace three multipliers.
- `tests/test_rtl.py::test_a_narrow_accumulator_is_caught`: a test that proves
  the comparison can fail.
- Horowitz (2014) for where the energy goes in digital systems, and
  Meinerzhagen et al. (2012) for memories built for implants.

## Carried forward

1. **U6's running stabiliser in RTL.** `neurochain.stabilise.normalise` is
   floating point. It needs an integer reference (fixed-point mean and
   mean-absolute-deviation registers, a shift for the gain) and a comparison
   with the floating-point version before any RTL. Not started.
2. **The radio word.** The RTL emits 18 + 7 bits per bin; U3's model sends 12.
   1.05 % of bins on this data exceed 12 bits. Choose a format and check that
   decoding (U4) does not change.
3. **The memory energy.** Replace the 20 fJ per bit assumption with a
   characterised memory (an OpenRAM macro, a foundry SRAM datasheet, or a
   standard-cell memory synthesised at the hardware phase), at one supply
   voltage for logic and memory.
4. **U2's handover says "18-bit accumulator" for the 2 kS/s path.** With the
   canceller in front it is 20 bits (`TODO.md`, U2 carried-forward list).
5. **Clock and memory gating**, which check 4 shows is where three quarters of
   the digital power is.
6. **Simulate the placed-and-routed netlist and run on a board** when one
   exists (U8). `ecppack` is installed; no bitstream was made.
7. **CI has no Verilator, Yosys or nextpnr.** The tool tests skip there.
   Ubuntu's packaged Verilator may be too old for `--trace-saif`.
8. **Verify the Horowitz (2014) table** (0.1 pJ for a 32-bit add at 45 nm).
9. **One template per stimulation amplitude** (U5's carried-forward item): the
   residual at the start of each train in `figures/u7_bit_exact.png` is what
   it would remove.

## Papers used

Verified means the title, venue and the number quoted were confirmed from a
publisher or repository page during this unit. "From memory" means not
checked.

1. Meinerzhagen, P., Sherazi, S. M. Y., Burg, A. and Rodrigues, J. N. (2012).
   A 500 fW/bit 14 fJ/bit-access 4kb standard-cell based sub-VT memory in
   65nm CMOS. *Proc. ESSCIRC*, 321–324.
   https://doi.org/10.1109/ESSCIRC.2012.6341319. Use: the energy per memory
   bit accessed (14 fJ at 0.5 V) and the leakage per bit (0.5 pW at 0.22 V)
   behind two constants of `neurochain.rtl.energy.EnergyModel`. Title, venue,
   DOI and both numbers verified from repository abstracts; the full text was
   not read, and the author list is from memory.
2. Horowitz, M. (2014). Computing's energy problem (and what we can do about
   it). *IEEE ISSCC Digest of Technical Papers*, 10–14.
   https://doi.org/10.1109/ISSCC.2014.6757323. Use: 0.1 pJ for a 32-bit
   addition at 45 nm and 0.9 V, as a cross-check on the logic energy. Title,
   venue and DOI verified; **the energy figure and the page range are from
   memory** (the PDF could not be opened here).
3. Kuon, I. and Rose, J. (2007). Measuring the gap between FPGAs and ASICs.
   *IEEE Transactions on Computer-Aided Design of Integrated Circuits and
   Systems* 26(2), 203–215. https://doi.org/10.1109/TCAD.2006.884574. Use:
   why an FPGA's power is not scaled to an ASIC's here (about 14 times in
   dynamic power on average; an alternative considered and not used). Title,
   venue, volume, pages and the factor verified from the abstract; the DOI is
   from memory.
4. Widrow, B. et al. (1975), Nason, S. R. et al. (2020) and the U5 sources for
   the pulse shape, all already in `docs/references.md`. Use: the canceller is
   U5's, the feature is Nason's SBP; nothing new is taken from them.
5. Hartley, R. I. (1996). Subexpression sharing in filters using canonic
   signed digit multipliers. *IEEE Transactions on Circuits and Systems II*
   43(10), 677–688. https://doi.org/10.1109/82.539000. Use: background for
   canonical signed-digit constant multiplication (section 6); the recoding
   identity in `nc_const_mult.sv` is the standard non-adjacent form and was
   checked numerically, not taken from this paper. **Citation from memory;
   verify before quoting.**

Tools, not papers: Verilator (https://verilator.org), Yosys
(https://yosyshq.net/yosys/), nextpnr (https://github.com/YosysHQ/nextpnr),
Project Trellis (https://github.com/YosysHQ/prjtrellis).
