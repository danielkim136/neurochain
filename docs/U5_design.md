# U5 design: stimulation artifacts

U1 to U4 asked how cheap a sensing chain can be while the brain is the only
thing driving the electrodes. A bidirectional implant also writes: it injects
current to evoke sensation. That current reaches the recording electrodes as a
voltage hundreds of times larger than a spike. U5 asks what that does to the
chain U3 chose, and what it costs to undo.

Code: `src/neurochain/stim/` (pulse, propagation, cancellation),
`src/neurochain/frontend/saturation.py` (the amplifier's large-signal model),
`neurochain.frontend.spice.transient_netlist` (the same circuit in ngspice),
`src/neurochain/analysis/u5.py` (the experiment). One command: `make u5`
(about a minute). Numbers: `results/u5_metrics.json`.

## 1. What U5 has to answer

1. How large is the artifact at a recording electrode, and what sets it?
2. How long is each front end blind after a pulse, and why?
3. What is left in the features after blanking, and after cancelling?
4. What does that do to decoding, for U3's 0.48 µW chain and for the
   conventional chain?
5. What does the canceller cost in the µW terms of U3?

Done-criterion (from `TODO.md`): decoding loss with and without cancellation,
plus recovery time.

## 2. The artifact

### 2.1 The pulse (`stim/pulse.py`)

A biphasic, cathodic-first current pulse: 60 µA for 200 µs, a 100 µs gap, then
30 µA the other way for 400 µs. This is the shape Flesher et al. (2016) used in
human somatosensory cortex; each phase carries 60 µA × 200 µs = 12 nC. Pulses
come in trains at 100 per second, 0.5 to 1.5 s long with pauses of the same
length, as feedback from a sensor would.

The second phase exists to take the charge back out (Merrill et al., 2005). Two
current sources never match exactly, so the pulse has an `imbalance`: the
second phase returns 99 % of the charge and the remaining 1 % (0.12 nC) stays
on the stimulating electrode's double layer. The stimulator then shorts the
electrode to its return, and that charge flows back through the tissue as a
decaying exponential. Its time constant is the electrode's capacitance times
the resistance in the loop. I assume 1 ms: about 20 nF (12 to 20 nC moves a
microelectrode by roughly a volt) through about 50 kΩ. Both the 1 % and the
1 ms are assumptions, and check 4 below shows how much rides on them.

### 2.2 Propagation (`stim/propagation.py`)

Below a few kilohertz cortex is a resistor: Logothetis et al. (2007) measured an
impedance that does not depend on frequency and barely on direction. So the
tissue does not filter the pulse. The potential is the current waveform times
one number per electrode, and that number comes from the resistive version of
Coulomb's law. For a point source of current `I` in a medium of conductivity
`σ`:

```
V = I / (4 π σ r)
```

Hand calculation, 60 µA, σ = 0.3 S/m, the neighbouring electrode at 400 µm:
`60e-6 / (4π × 0.3 × 400e-6) = 39.8 mV`. The code gives 39.79 mV.

Two geometries matter (figure `u5_artifact.png`, left):

- **Monopolar**: the current returns through a distant electrode. The artifact
  falls as `1/r`. This is how human ICMS is delivered and it is the case used
  everywhere below.
- **Bipolar**: the current returns through a neighbour 400 µm away. Source and
  sink nearly cancel at a distance, and the artifact falls as `1/r²`: at 20 mm
  it is 50 times smaller than monopolar. The price is a stimulation field that
  reaches fewer neurons, which is why it is not the default. It is in the
  figure as the cheapest fix that is not electronics.

The recording reference is taken to be distant, so none of the artifact is
common to electrode and reference. That is the worst case. A reference on the
array would subtract the common part (`tests/test_stim.py` shows a factor of
more than 6 at 20 mm).

| Where the stimulating electrode is | Artifact on the 96 channels |
|---|---|
| On the recording array | 6.2 to 39.8 mV, median 9.9 mV |
| 5 mm from its centre | 2.3 to 5.0 mV, median 3.1 mV |
| 20 mm from its centre | 0.73 to 0.87 mV, median 0.79 mV |

Against that: spikes are 50 to 300 µV, the ADC takes ±1 mV, and the amplifier's
output reaches its rails when the input passes ±8 mV (0.8 V of swing divided by
a gain of 100). On its own array the pulse is past the ADC's range on every
channel and past the amplifier's on 72 % of them.

## 3. The front end in saturation

### 3.1 Why the hard clip of U1 is not enough

U1's emulator clips the output at the rails and carries on. That is right while
the amplifier is linear, and wrong once it is not, because a saturated
amplifier comes back in a different state from the one a clip assumes.

The model (`frontend/saturation.py`) is U1's circuit with two nonlinearities:

- **The input pair.** In weak inversion a differential pair steers its tail
  current as `I_max tanh(gm v / I_max)`. For small `v` this is `gm v`, the
  small-signal model of ELEC2133. For large `v` the current stops at `I_max`:
  that is slew rate.
- **The rails.** A stiff conductance that only conducts once the output passes
  ±0.8 V.

Written as state equations it has two states, the charge on the summing node
and the output voltage (the docstring has the algebra). The same two
behavioural elements go into ngspice as `B` sources
(`spice/u5_saturation.cir`).

### 3.2 Why saturation has memory

A charge-balanced pulse has no DC content. A linear high-pass therefore has
nothing to remember: the charge the first phase pushes through the feedback
resistor `R2`, the second phase pulls back. Measured on the 1 Hz chain, a 3 mV
pulse leaves less than 0.05 µV behind (`tests/test_saturation.py`).

A rail breaks the cancellation. While the output is pinned, the feedback loop is
open and the current through `R2` is set by the rail, not by the input. A 60 mV
phase and a 30 mV phase both give 0.8 V at the output, so the phase that lasts
twice as long now moves twice the charge. When the pulse ends the difference is
still on the summing node. After a 30 mV pulse the 1 Hz chain is left with a
10.0 µV offset that decays with `R2 C2` = 160 ms, which is sixteen pulse
periods: in a 100 Hz train the offsets pile up.

### 3.3 The cheap chain rings

U3's winner has a 300 Hz high-pass and a 1 kHz low-pass. Those corners are so
close that the closed loop is no longer two separate real poles. The
characteristic equation
`(s C_s + G2)(s (CL + C2) + Go + G2) − (s C2 + G2)(s C2 + G2 − gm) = 0`
has a complex pair: natural frequency 473 Hz, damping ratio 0.82, ringing at
271 Hz under an envelope that decays with a time constant of 0.41 ms
(`SaturatingAmplifier.poles`). The conventional chain's poles are real, at 1 Hz
and 7.5 kHz.

That one fact predicts the recovery time. The output leaves the pulse at about
its rail, 12 mV referred to the input, and has to fall to 13 µV:
`0.41 ms × ln(12 000 / 13) = 2.8 ms`. Measured: 2.95 ms.

### 3.4 Measured recovery

Recovery time is counted from the end of the second phase until the output,
referred to the input, stays inside ±13 µV (the in-vivo noise level).

| Artifact (mV) | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|
| 0.48 µW chain (ms) | 0.26 | 1.35 | 1.74 | 1.98 | 2.14 | 2.95 | 2.23 |
| Conventional chain (ms) | 0.03 | 0.05 | 0.08 | 0.11 | 0.35 | 1.01 | 2.17 |

Three things to read from it:

- **The narrow band that made the chain cheap makes it slow.** At 3 mV, well
  inside both amplifiers' linear range, the 0.48 µW chain needs 1.98 ms and the
  conventional chain 0.11 ms. This is plain linear filtering, the time-domain
  face of a narrow band, and U3 did not charge for it.
- **The conventional chain's time at large amplitudes is the discharge tail,**
  not the amplifier: 1 % of the charge returning over 1 ms is a 60 µV tail on a
  30 mV artifact, and the wide band passes it untouched.
- **The cheap chain is not monotonic** (2.95 ms at 30 mV, 2.23 ms at 100 mV).
  Its tail is an oscillation and the discharge tail adds to it; which lobe is
  the last above 13 µV changes with amplitude.

At 30 mV the cheap chain's output is outside the ADC's range until 1.30 ms
after the pulse ends, which is 2.0 ms after it starts: three of the first four
samples at 2 kS/s are clipped. The conventional chain is back in range 0.05 ms
after the pulse.

### 3.5 Model against ngspice

The Python model and ngspice solve the same equations and agree on every
recovery time, both chains, all seven amplitudes, within 4 µs. After the pulse
the waveforms differ by at most 0.9 µV up to 30 mV.

The Python model drives the amplifier from an ideal source. ngspice with the
electrode's impedance in series gives 2.87 ms instead of 2.95 ms for the cheap
chain at 30 mV, and 1.23 ms instead of 1.01 ms for the conventional chain
(0.69 ms instead of 0.35 ms at 10 mV). So the model is good to 3 % on the chain
U5 is about, and understates the conventional chain's recovery by up to a half
where it is shortest.

## 4. Injecting the artifact into recordings

No dataset here has stimulation in it, so the artifact is added in simulation.
Two approximations make that cheap, and both are tested.

- **Neural signal and artifact superpose** wherever the sum is inside the
  ADC's range. The amplifier's linear range (±8 mV) is wider than the ADC's
  (±1 mV), so any sample where the amplifier is nonlinear is a sample the ADC
  has clipped anyway. The chain's output is therefore `clip(linear response to
  the neural signal + nonlinear response to the artifact alone)`.
- **Pulses superpose** once the previous pulse's fast states have decayed. Two
  saturating pulses 5 ms apart give the sum of two single responses to within
  1 µV (`test_pulses_superpose_when_the_tail_has_died`). So the response to one
  pulse is computed once per amplitude (a *kernel*) and a train is a
  convolution.

**The stimulator runs off the ADC's clock.** Pulses start on a sample edge and
the period is a whole number of samples. The implant generates both, so this
costs nothing, and it means every pulse is sampled at the same instants and
looks identical in the ADC's output. Check 2 shows what happens without it.

The recordings: the MEArec simulation (32 channels, 60 s, 13 µV background,
known spike times) for statistics, and the Nason in-vivo channel (10 s of real
broadband) as a check that nothing depends on the recording being simulated.

## 5. Removing it (`stim/cancel.py`)

The implant fires the stimulator, so it knows when each pulse starts. Both
methods use that and nothing else.

### 5.1 Blanking

Discard a window after each pulse (Weiss et al., 2019, describe blanked and
filtered recordings in people). No arithmetic, no residue. The window has to outlast the recovery, and
everything in it is lost. Here each amplitude gets the shortest window that
covers its own recovery, which is the most favourable case for blanking: 1.0 ms
at 0.1 mV, 4.0 ms at 30 mV on the cheap chain; 0.7 to 2.9 ms on the
conventional chain.

### 5.2 Adaptive template subtraction

Because of the shared clock the artifact is the same after every pulse. Keep a
running average of it, one value per sample position `k` after the trigger, and
subtract:

```
e[k]  = x[k] − T[k]                 output
T[k] += (x[k] − T[k]) >> shift      update
```

This is the LMS algorithm of ELEC3104 at its smallest. An LMS filter updates
each weight by `μ × error × reference`. Here the reference is an impulse at the
trigger, so weight `k` only ever sees sample `k`, the reference is 1, and the
update is `μ × error` with `μ = 2^−shift`: a shift and an add, no multiplier.
Along the pulse index the update is a one-pole low-pass with a memory of about
`2^shift` pulses (Widrow et al., 1975, is the general idea), which is what lets
the template follow an artifact that changes as electrodes age.

Three details, each of which showed up as a number:

- **The template contains a little neural signal**, so subtracting it adds
  noise. An exponentially weighted average of past noise has `μ / (2 − μ)` of
  its variance and is independent of the present sample, so noise power rises
  by `1 + μ / (2 − μ)`. For `shift = 4` that is 3.2 % in power, 1.6 % in
  amplitude. Measured SBP shift at small artifacts: +1.7 % of the noise floor.
- **A clipped sample cannot be recovered.** Samples at either end of the ADC's
  range are blanked individually, by looking at the code.
- **Rounding matters when the signal is one LSB.** With 8 bits over ±1 mV the
  LSB is 7.8 µV and the mean SBP is about 1.3 LSB. Rounding the template to a
  whole LSB before subtracting puts a rounding error of up to half an LSB on a
  signal that small (`test_integer_canceller_matches_floating_point` compares
  the two). The high-pass filter of U2 already shifts its input left by two
  guard bits, so the canceller hands it a quarter-LSB result and fills those
  bits with information. Integer and floating-point cancellers then
  differ by at most 0.29 LSB and the SBP they produce correlates at 0.9984.

On the conventional chain the template has to cover the whole pulse period
(300 samples), because of the offset from section 3.2. With a template that
stopped at 8 ms, the output stepped by the accumulated offset at every pulse
and the detector fired on each step. That was a bug in my first version, found
from the false-positive count.

### 5.3 Alternatives considered

| Option | Why not |
|---|---|
| An ADC with enough range to digitise the artifact (Zhou et al., 2019) | 40 mV at an 8 µV step is 14 bits. In U3's model the ADC goes from 0.026 µW to `50 fJ × 2^14 × 2 kS/s` = 1.6 µW, three times the whole chain, and the amplifier must also stay linear |
| LMS with the stimulus as reference through an FIR filter (Mendrela et al., 2016) | A multiplier per tap, and at 2 kS/s the pulse is shorter than a sample |
| Cross-channel methods such as ERAASR (O'Shea and Shenoy, 2018) | Matrix decompositions across channels; an offline tool |
| A switch that resets the amplifier during the pulse | Would shorten recovery; needs a circuit model of the switch's own charge injection. Left for the hardware phase |
| Bipolar stimulation | 50 times less artifact at 20 mm, but a different stimulation field: not the sensing chain's decision |

## 6. From features to decoding

As in U3, no dataset has both broadband and behaviour, so the effect is
measured where the truth is known and carried across.

On MEArec, for each amplitude and treatment, every stimulated 50 ms bin is
compared with the same bin without stimulation:

- **SBP**: mean and spread of `(SBP − clean SBP) / noise floor`.
- **Crossings**: the share of the unstimulated chain's true detections that
  survive, and false positives added per second.

On the Nason decoding data a stimulation schedule marks 51 % of bins as
stimulated. Each channel gets its artifact amplitude from the geometry of
section 2.2, and in stimulated bins its feature is changed by what was measured
at that amplitude. The stimulating electrode's own channel reads a constant.

Decoders are U4's (Wiener filter and GRU), scored on the stimulated bins of
held-out folds. Two protocols: fitted on unstimulated data and then used with
stimulation on (a decoder calibrated before feedback is switched on), and
refitted with stimulation on. Losses are paired differences against the same
decoder on the same bins without stimulation.

## 7. What the numbers say

### 7.1 Untreated, the chain is unusable

A 30 mV artifact raises SBP by 16.6 noise floors (14.8 on the real recording)
and adds 87 false crossings per second. With the stimulating electrode on the
array the GRU scores R² −0.25 on either chain; the Wiener filter, being linear,
scores −2296 on SBP. Even 20 mm away, where no sample clips, both decoders fail.
Refitting does not rescue it (R² 0.005 for the cheap chain).

### 7.2 Blanking costs the cheap chain 25 to 40 % of its samples

From 0.3 mV up, because of section 3.3. At 30 mV the cheap chain keeps 60 % of its samples in a
100 Hz train and the conventional chain 83 %. Both are unbiased to within 0.04
noise floors.

### 7.3 The template canceller gets most of them back

It keeps 86 % at 30 mV (the three clipped samples per pulse are the loss) and
all of them up to 1 mV, with a residual of 3.3 to 4.0 µV RMS that is the
template's own noise. On the real recording: 86 % kept, SBP shift +0.003 noise
floors.

### 7.4 Decoding, stimulating electrode on the recording array

R² on stimulated bins, GRU, with the paired change against no stimulation
(0.718 for the cheap chain, 0.724 for the conventional one):

| Chain and treatment | Fitted without stimulation | Refitted with stimulation |
|---|---|---|
| 0.48 µW, blanking | 0.579, −0.139 [−0.168, −0.111] | 0.679, −0.039 [−0.064, −0.014] |
| 0.48 µW, template | 0.627, −0.091 [−0.115, −0.067] | 0.707, −0.011 [−0.032, +0.009] |
| Conventional, blanking | 0.614, −0.110 [−0.146, −0.079] | 0.699, −0.025 [−0.044, −0.006] |
| Conventional, template | 0.351, −0.374 [−0.434, −0.313] | 0.692, −0.033 [−0.054, −0.010] |

Read with the Wiener-filter rows of the README table, this says four things.

- **With the canceller and a refitted GRU the cheap chain loses nothing
  measurable**: −0.011, interval spanning zero. It ends level with the
  conventional chain (0.707 against 0.699) at one fortieth of the power.
- **A decoder calibrated without stimulation does not transfer.** Every
  treatment leaves a small shift that is the same sign on all 96 channels
  (−0.04 noise floors after blanking, +0.02 or −0.04 after cancelling, +1 false
  crossing per second on the conventional chain). A shift common to every input
  is exactly what a decoder that has never seen it cannot tell from a change in
  behaviour. The GRU is the more brittle: the conventional chain with the
  canceller loses 0.374 unrefitted and 0.033 refitted. Its Wiener filter loses
  0.035 and 0.026.
- **With the Wiener filter the cheap chain stays behind.** Refitted, it loses
  0.072 with the canceller and 0.080 with blanking, where the conventional
  chain with blanking loses nothing (+0.002). SBP is a small modulation on top
  of a noise pedestal; anything that perturbs the pedestal by a few percent
  competes with the signal. Crossing counts lose a share of their events and
  are otherwise unchanged. U3's "within 5 %" does not survive stimulation with
  a linear decoder.
- **Farther away, the canceller's advantage is clearer.** At 5 mm (3 mV,
  nothing saturates) the unrefitted GRU loses 0.023 with the canceller and
  0.147 with blanking.

### 7.5 Cost

Two additions per sample and no multiplier. Charging each addition as a full
1 pJ multiply, the canceller is 0.004 µW per channel, 0.8 % of the 0.48 µW
chain, plus 224 bits of template (16 taps of 14 bits). On the conventional
chain the same canceller is 0.060 µW and 4800 bits, because both scale with the
sample rate. The figure is an upper bound that U7 replaces with a synthesised
one.

So the answer to the research question changes in one place: the cheapest chain
that still decodes while stimulating is U3's 0.48 µW chain plus a 0.004 µW
canceller, 0.488 µW, **provided** the decoder is nonlinear and is calibrated
with stimulation on.

### Limitations

- **Tissue is a uniform resistor with point sources.** No skull, no fluid
  layer, no shielding by neighbouring shanks. Amplitudes scale as `1/σ` and
  0.3 S/m is assumed.
- **Coupling inside the implant is not modelled.** In real systems a share of
  the artifact arrives through capacitance between stimulation and recording
  leads, not through tissue. It would add a component with a different shape.
- **`R2` is a linear resistor.** A real MOS pseudo-resistor conducts hard once
  the summing node moves by a few hundred millivolts, which changes the charge
  moved during saturation. There are no input protection diodes either.
- **Stimulation does not change the brain here.** Real pulses evoke spikes
  time-locked to the pulse. A template would learn those too and subtract
  them.
- **One pulse amplitude.** Feedback that modulates amplitude needs one template
  per level, and the memory multiplies.
- **The bridge carries a mean and a spread per bin**, drawn independently
  across channels. The mean is common to all channels, as it should be; the
  random part of a real residual would also be partly shared, since every
  channel sees the same pulse times.
- **Blanking was given per-channel windows**, the best it can do. One window
  for the whole array would cost it more.
- **Nason's electrode map is not published.** Channels are assumed to run row
  by row.
- **The pulse parameters are quoted from memory** of Flesher et al. (2016) and
  the imbalance and discharge time are assumptions. Check them against the
  paper before quoting this outside the project.
- **The refitted protocol assumes test-time stimulation looks like
  training-time stimulation.**

## 8. Prediction checks

Each changes one setting, reruns the signal-level measurement on MEArec and
the decoding with the stimulating electrode on the array, and prints the new
numbers under the committed ones. Each takes about a minute and a half. Run
them one at a time: each loads the 0.9 GB recording and starts eight workers.
Write down your prediction before opening the answer.

For reference, the committed numbers at 30 mV for the 0.48 µW chain with the
template canceller: 86 % of samples kept, residual 4.0 µV, SBP shift −0.039
noise floors; GRU change −0.091 unrefitted and −0.011 refitted.

### Check 1: a template that adapts in two pulses

```
make u5-key ARGS="--adapt-shift 1"
```

The template's step size goes from 1/16 to 1/2, so it follows the artifact
almost instantly. Is that better? What happens to the residual and to SBP?

<details>
<summary>Answer</summary>

Worse on every count. The residual rises from 4.0 to **8.8 µV**, the SBP shift
goes from −0.039 to **+0.072** noise floors, and the conventional chain gains
**8.1** false crossings per second instead of 1.1. Decoding with the GRU:
**−0.394** [−0.453, −0.340] unrefitted (from −0.091) and −0.043 [−0.073,
−0.017] refitted (from −0.011).

A template with a memory of two pulses is half made of the previous pulse's
neural signal. Subtracting it adds noise of power `μ / (2 − μ)` = 1/3 of the
signal's own, against 1/31 at `shift = 4`. The canceller has turned into a
noise source.

The other direction buys almost nothing: `--adapt-shift 6` gives −0.103
unrefitted and −0.015 refitted. The template's noise was already small at 4;
what remains is the dent the blanked samples leave in the high-pass filter,
which no step size touches. So 4 it stays: enough memory to be quiet, short
enough (16 pulses, 160 ms) to follow drift.
</details>

### Check 2: a stimulator on its own clock

```
make u5-key ARGS="--jitter-us 100"
```

Each pulse now starts up to 100 µs early or late relative to the ADC's sample
clock: a fifth of a sample at 2 kS/s. The artifact has the same shape every
time. Does the template canceller still work?

<details>
<summary>Answer</summary>

It fails completely. The residual goes from 4.0 to **122 µV**, the SBP shift
from −0.039 to **+3.48** noise floors, and the refitted GRU scores R²
**0.000** [change −0.718]. Blanking is barely affected: it keeps 55 % of the
samples instead of 60 % and loses 0.051 refitted.

The artifact is identical in continuous time, but the ADC only sees it at its
sample instants, and near the pulse it changes by millivolts within one
sample. Moving the pulse by a fifth of a sample changes every sampled value by
far more than a spike. The template is an average of samples that no longer
share a shape. This is aliasing: the artifact has content far above the 1 kHz
Nyquist frequency, and a sampled record of it is only repeatable if the
sampling phase is.

It is the strongest design rule in the unit and it costs nothing: derive the
stimulator's timing from the ADC clock. It is also a warning for any test with
an external stimulator.
</details>

### Check 3: stimulating at 300 pulses per second

```
make u5-key ARGS="--rate-hz 300"
```

The period drops to 3.5 ms (the nearest whole number of samples; 286 per
second). The 0.48 µW chain needs 2.95 ms after a 0.7 ms pulse to recover. What
is left for blanking, and for the canceller?

<details>
<summary>Answer</summary>

Blanking keeps **0 %** of the cheap chain's samples: the window is as long as
the period, and the chain is blind for as long as a train lasts. The refitted
GRU still scores 0.530 (change −0.188), but only from the bins of history
before the train. The conventional chain with blanking keeps 60 % of its
crossings and loses 0.093 refitted.

The canceller keeps **59 %** of the samples, since only the clipped ones are
lost, and the refitted GRU loses **0.025** [−0.052, −0.000]: R² 0.693.
Unrefitted it loses 0.565, because the SBP shift has grown to −0.196 noise
floors: three of every seven samples are now zeros going into the high-pass
filter.

So the canceller is not just an improvement on blanking at high rates, it is
the only one of the two that works on this chain. And the shift a decoder has
to be calibrated for grows with the stimulation rate.
</details>

### Check 4 (bonus): a stimulator with 5 % charge imbalance, and a perfect one

```
make u5-key ARGS="--imbalance 0.05"
make u5-key ARGS="--imbalance 0"
```

The committed pulse leaves 1 % of its charge to return over 1 ms. Which chain's
recovery time depends on that, and which treatment cares?

<details>
<summary>Answer</summary>

The conventional chain's recovery at 30 mV is **0.13 ms** with perfect balance,
1.01 ms at 1 % and **2.56 ms** at 5 %. Its wide band passes the discharge tail
untouched, so its recovery time is the stimulator's, not the amplifier's. With
blanking it keeps 95 %, 88 % and 72 % of its crossings.

The cheap chain goes the other way: 3.08 ms, 2.95 ms, **2.44 ms**. Its own
ringing dominates, and the tail happens to oppose the last lobe.

The template canceller does not care: the tail repeats, so it is subtracted.
Refitted GRU change at 5 %: −0.004 [−0.026, +0.017] on the cheap chain. This is
the general point about subtraction against blanking: blanking pays for every
slow component of the artifact, subtraction only for the unrepeatable ones.
</details>


## 9. What to read next

- Widrow et al. (1975), sections I to III: adaptive noise cancelling, with the
  two-weight sine canceller that is the nearest relative of the template.
- Merrill, Bikson and Jefferys (2005), section 3: why pulses are biphasic and
  what "charge balanced" does and does not guarantee.
- Weiss et al. (2019): what blanking plus filtering achieves in a person, and
  the artifact amplitudes measured there, to hold against section 2.2.
- Zhou et al. (2019), the WAND front end: the opposite design choice, an input
  range wide enough that nothing saturates.
- In this repo: `tests/test_cancel.py::test_noise_gain_formula` for the
  `1 + μ/(2−μ)` result, and `tests/test_saturation.py` for the pole pair.

U6 takes the other half of the research question: the same chain while its
electrodes age. The adaptive template is the one part of U5 built to follow
that drift (`test_adaptation_follows_a_drifting_artifact`).
