# U4 design: decoders and evaluation

**Done-criterion:** a Kalman filter and a small GRU; session-held-out splits;
R² per configuration with confidence intervals.
**Status:** met. Four decoders are scored on fifteen dataset-and-chain
combinations (seven chains on Nason, four on each NoMAD session), every R² has
a 95 % interval, and the comparison that matters
has a paired interval on the difference. "Held-out" is reported two ways:
held-out blocks of trials within a session, and a whole held-out session.
`make u4` reproduces everything in about three minutes.

Every number below comes from `results/u4_metrics.json` or from
`scripts/u4_key_number.py`.

## 1. What U4 has to answer

U3 chose a sensing chain (SBP, 0.25 µA, 8 bits, 2 kS/s: 0.48 µW per channel)
using one decoder: a ridge readout over a quarter of a second of history. That
leaves three doubts.

1. **Does the choice depend on the decoder?** A cheap chain that only matches
   the expensive one under a weak decoder has not really matched it.
2. **How sure are the numbers?** U3 gave R² without intervals. A 0.02 gap
   between two chains means nothing until its uncertainty is known.
3. **Would a better decoder change the design?** If the decoder moves accuracy
   more than the front end does, power is better spent there.

The short answers: no, it does not depend on the decoder; the gaps are mostly
inside their intervals; and yes, on finger movements the decoder matters far
more than the front end.

## 2. The four decoders

All four are causal: the estimate at bin `t` uses features up to bin `t` only.
In ELEC3104 terms they are three kinds of filter from the feature vector `y[t]`
to the behaviour `x[t]`.

| Decoder | Filter type | Memory | Fitted by |
|---|---|---|---|
| U3's ridge readout | FIR, 4 or 5 taps per channel | 0.24 or 0.25 s | least squares with a fixed penalty |
| Wiener filter | FIR, 8 or 10 taps per channel | 0.5 s | least squares, penalty by cross-validation |
| Kalman filter | first-order IIR with matrix coefficients | infinite, decaying | two least-squares fits and a Riccati iteration |
| GRU | nonlinear IIR whose poles depend on the input | infinite, gated | gradient descent |

### 2.1 U3's ridge readout (`decode/ridge.py`, `RidgeDecoder`)

Unchanged from U3, as the reference point. Each output is a weighted sum of the
last few bins of every channel: an FIR filter per channel, summed. The weights
solve `(X'X + αnI) w = X'y` (MATH2099 least squares with a penalty), with
`α = 0.01`.

### 2.2 Wiener filter (`decode/ridge.py`, `WienerDecoder`)

The same linear map, tuned the way the other decoders are. It was added during
U4 because the comparison was unfair without it: the Kalman filter chooses its
delay by cross-validation and the GRU stops on validation loss, while U3's
readout had a fixed history and a fixed penalty. On NoMAD session 1 that
readout scores 0.305 and the Wiener filter 0.539, from the same linear model.
Without this row, most of that gap would have been credited to the Kalman
filter.

Two changes: half a second of history, and the penalty chosen from
`(0.01, 0.1, 1, 10)` on three contiguous blocks of the training trials. More
taps need a larger penalty because there are more weights to fit noise with
(prediction check 4 shows what happens otherwise). The selection costs one
eigendecomposition per block: with `X'X = V diag(s) V'`, the solution for every
penalty is `V diag(1/(s + αn)) V'X'y`.

### 2.3 Kalman filter (`decode/kalman.py`)

The Kalman filter assumes a model of how the data were generated (Wu et al.,
2006):

```
behaviour has momentum:     x[t] = A x[t-1] + w,   w ~ N(0, W)
features encode behaviour:  y[t] = C x[t]   + q,   q ~ N(0, Q)
```

`A` and `C` are least-squares fits; `W` and `Q` are the covariances of their
residuals (MATH2099: regression, covariance matrices). At each step the filter
predicts `x[t]` from `x[t-1]`, then corrects by a gain `K` times the part of
`y[t]` the prediction did not explain.

Three design choices, each with the alternative it beat:

- **Steady-state gain.** The model does not change over time, so `K` converges
  within a few steps. It is computed once by iterating the Riccati equation,
  and the filter becomes `x[t] = M x[t-1] + K y[t]` with `M = (I - KC)A`. That
  is a first-order IIR filter; the eigenvalues of `M` are its poles and lie
  inside the unit circle. The alternative, updating `K` every step, gives the
  same answer after the first few bins of each trial and costs a matrix
  inversion per step, which an implant cannot afford (Malik et al., 2011).
- **Displacement in the state.** The state is `[velocity, displacement since
  the trial began]`. Neurons are tuned to position as well as velocity, so the
  extra state explains more of `y`. Only velocity is scored. Because
  displacement is an exact sum of velocity, `W` is singular; the Riccati step
  is written as `P = (I + P_pred C'Q⁻¹C)⁻¹ P_pred` so that nothing singular is
  inverted.
- **A delay chosen by cross-validation.** Cortex leads movement, and the
  observation model is instantaneous, so the features are delayed by 0 to 4
  bins before filtering. The delay is chosen on four contiguous blocks of the
  training trials. The first version used a single validation block (the last
  fifth of the training trials) and chose badly on the Nason data, because the
  end of that session behaves differently from the rest. Cross-validation over
  the whole training set does not have that weakness. The chosen delays are in
  the results file: 1 bin (50 ms) on Nason, 1 bin (60 ms) on NoMAD session 2,
  4 bins (240 ms) on NoMAD session 1.

### 2.4 GRU (`decode/gru.py`)

A gated recurrent unit (Cho et al., 2014) keeps a hidden vector `h` and updates
it as

```
z = σ(W_z [y, h])            how much to overwrite
r = σ(W_r [y, h])            how much of the past to consult
h̃ = tanh(W [y, r·h])         the candidate new state
h = (1 - z)·h + z·h̃
```

Read `h = (1 - z)·h + z·h̃` as a leaky integrator, the same first-order IIR as
the Kalman filter's `x = Mx + Ky`, except that the leak `z` is recomputed from
the input at every step. The filter can hold still when nothing is happening
and respond fast when something is. A fixed-gain filter has to pick one.

The network is deliberately small: dropout on the input, one GRU layer of 64
units, one linear output layer; 30 784 multiplies per bin on 96 channels.
Training: Adam, learning rate 3·10⁻³, weight decay 10⁻⁴, dropout 0.3, sequences
of 100 bins, gradient clipping at 1, and early stopping on the last 15 % of the
training trials. It trains on the CPU; for a network this small the M-series
GPU was slower in a quick test, so MPS is not used.

Alternatives not taken: an LSTM (a third more weights, no evidence it would
help at this size); a transformer (needs far more data than 409 trials); a
bidirectional GRU (not causal, so not realisable in an implant).

## 3. How the scores are made honest

**Held-out predictions.** Trials are cut into five contiguous blocks. Each
block is predicted by a decoder fitted on the other four, so every bin is
predicted exactly once by a decoder that never saw it. Feature scaling uses the
training blocks only. Blocks are contiguous because neighbouring trials share
slow drifts; shuffled folds would leak them.

**One R² from all bins.** `R² = 1 - Σ(x - x̂)² / Σ(x - x̄)²` over all held-out
bins, averaged over the behaviour's dimensions. U3 reported the mean of the
per-fold R² over several random draws of the bridge, so its numbers differ
slightly: 0.462 there, 0.471 here, for the same chain and readout.

**Intervals from resampling trials.** A thousand times, draw as many trials as
the session has, with replacement, and recompute R². The 2.5th and 97.5th
percentiles are the interval (MATH2099: sampling distributions, percentile
intervals). Trials are resampled, not bins, because neighbouring bins are
strongly correlated: ten thousand bins are nowhere near ten thousand
independent observations, and treating them so would give intervals several
times too narrow.

**Paired differences.** To compare two chains or two decoders, both are scored
on the *same* resampled trials and the difference is taken inside each
resample. Hard trials are hard for both, so that shared variation cancels. On
NoMAD session 2 the two chains' separate intervals are each about ±0.1 wide,
yet the interval on their difference is about ±0.006.

**What the intervals do not cover.** They describe which trials happened to be
recorded. They do not include the GRU's random initialisation (a second seed
moved its two headline scores by 0.003 and 0.010; `--seed 1`), the bridge's
random draw, or the fact that there is one monkey and one person.

## 4. What the numbers say

Figures: `figures/u4_decoders.png`, `figures/u4_traces.png`,
`figures/u4_cross_session.png`.

### 4.1 U3's choice survives every decoder

R² on the conventional chain (crossings, 10 µA, 10 bits, 30 kS/s, 19.63 µW) and
on U3's choice, with the paired difference:

| Dataset | Decoder | Conventional | U3's choice | Difference |
|---|---|---|---|---|
| Nason (0.48 µW) | U3's ridge | 0.471 [0.446, 0.495] | 0.438 [0.412, 0.463] | -0.033 [-0.057, -0.010] |
| | Wiener | 0.478 [0.458, 0.497] | 0.472 [0.446, 0.496] | -0.006 [-0.027, +0.015] |
| | Kalman | 0.363 [0.337, 0.389] | 0.342 [0.314, 0.368] | -0.021 [-0.045, +0.003] |
| | GRU | 0.713 [0.689, 0.737] | 0.720 [0.695, 0.744] | +0.007 [-0.015, +0.032] |
| NoMAD session 2 (3.43 µW) | U3's ridge | 0.538 [0.413, 0.632] | 0.510 [0.393, 0.598] | -0.028 [-0.035, -0.020] |
| | Wiener | 0.650 [0.547, 0.726] | 0.632 [0.535, 0.703] | -0.018 [-0.023, -0.012] |
| | Kalman | 0.653 [0.540, 0.737] | 0.637 [0.526, 0.718] | -0.015 [-0.020, -0.011] |
| | GRU | 0.655 [0.514, 0.746] | 0.643 [0.521, 0.725] | -0.012 [-0.026, +0.006] |
| NoMAD session 1 (4.07 µW) | U3's ridge | 0.305 [0.271, 0.347] | 0.294 [0.260, 0.335] | -0.011 [-0.016, -0.007] |
| | Wiener | 0.539 [0.517, 0.566] | 0.532 [0.510, 0.560] | -0.007 [-0.008, -0.005] |
| | Kalman | 0.623 [0.591, 0.643] | 0.619 [0.588, 0.638] | -0.004 [-0.006, -0.002] |
| | GRU | 0.565 [0.547, 0.591] | 0.569 [0.550, 0.594] | +0.003 [-0.002, +0.009] |

- On Nason the 0.48 µW chain keeps 93 to 101 % of the conventional chain's R².
  With the three tuned decoders the difference interval includes zero. Only
  under U3's own readout is the loss distinguishable from nothing, and there it
  is 0.033.
- On NoMAD the cheaper chains keep 95 to 101 %. The losses are real (most
  intervals exclude zero) and small (at most 0.028). They are real by
  construction: the bridge removes a share of the spikes, which can only remove
  information.
- **The better the decoder, the less the cheap chain loses.** On every dataset
  the loss is largest under U3's readout and smallest under the GRU. A decoder with memory averages
  away part of the noise a cheap front end adds.

### 4.2 On finger movements, the decoder matters more than the front end

On Nason the GRU scores 0.72 where the best linear decoder scores 0.47, on the
same 0.48 µW chain: +0.282 over U3's readout. Going from the conventional
chain to U3's choice moved R² by 0.03; changing the decoder moved it by 0.28. Even the 0.12 µW chain (SBP at 0.05 µA) under the GRU
scores 0.643 [0.608, 0.677], above the 19.63 µW chain under any linear decoder
(0.478 at best).

The traces figure shows why. Finger movement in this task is short bursts
between holds at exactly zero velocity. The linear decoders cannot be quiet
during holds and fast during bursts, so they wobble in the holds and undershoot
the peaks. The GRU sits at zero and then follows the burst. This is the gated
leak of section 2.4 doing its job. That reading is an interpretation of the
traces, not something the analysis tests directly.

### 4.3 The Kalman filter is the best decoder on one dataset and the worst on another

- **NoMAD session 1:** Kalman 0.623, GRU 0.565, Wiener 0.539.
- **NoMAD session 2:** all three tie (0.653, 0.655, 0.650), with wide intervals
  because that session has only 16 recording segments to resample.
- **Nason:** Kalman 0.363, below even U3's readout (0.471).

The filter is optimal only when its model is right. NoMAD's cursor velocity is
smooth and close to the linear-Gaussian momentum model. The Nason finger
velocity is not: a Gaussian `w` cannot describe "nothing, then a burst". A
model-based decoder is a bet on the model.

### 4.4 A decoder moved to another session mostly fails, whichever it is

Fitted on one NoMAD session and tested on the other, 274 days apart, with no
recalibration:

| Decoder | Fitted on 2, tested on 1 | Fitted on 1, tested on 2 | Within session |
|---|---|---|---|
| U3's ridge | -0.018 [-0.058, 0.006] | -0.070 [-0.143, -0.010] | 0.305 and 0.538 |
| Wiener | 0.239 [0.227, 0.248] | 0.245 [0.184, 0.292] | 0.539 and 0.650 |
| Kalman | 0.119 [0.094, 0.140] | 0.092 [-0.039, 0.188] | 0.623 and 0.653 |
| GRU | 0.131 [0.121, 0.141] | 0.079 [0.063, 0.094] | 0.565 and 0.655 |

The best keeps under half of its within-session R². The surprise is the order:
the plainest tuned decoder transfers best, and the two that were best within a
session transfer worst. A heavily penalised linear map spreads its weight over
many channels, so it leans less on any one channel's tuning staying put. That
is a hypothesis, and U6 is where it gets tested, together with a stabiliser.

### 4.5 The decoder is nearly free in power

Under U3's 1 pJ per multiply, on 96 channels at 50 ms bins:

| Decoder | Multiplies per bin | µW per channel |
|---|---|---|
| Kalman | 196 | 0.00004 |
| U3's ridge | 480 | 0.0001 |
| Wiener | 960 | 0.0002 |
| GRU | 30 784 | 0.0064 |

The GRU costs 150 times more than the Kalman filter and still adds only 1.3 %
to the 0.48 µW chain. A decoder runs once per 50 ms bin; the front end runs
2000 times a second. Under this model, even running the GRU on the implant is
affordable. The model counts multiplies only: it leaves out the memory that
holds 31 thousand weights and its leakage.

### Limitations

- **Offline, not closed-loop.** These are open-loop reconstructions of recorded
  behaviour. In closed loop the user corrects for the decoder, and offline R²
  is known to rank decoders imperfectly. A GRU that wins offline by 0.28 will
  probably still win online, but not necessarily by that much.
- **The GRU result depends on its size.** With 8 hidden units instead of 64,
  U3's choice loses 0.098 against the conventional chain (prediction check 2).
  "Survives every decoder" is true of the decoders as configured here.
- **One draw of the bridge.** Each configuration is decoded on the first of
  U3's random draws of the degraded features.
- **NoMAD session 2 has 16 segments.** Its absolute intervals are about ±0.1.
  Only its paired differences are tight.
- **U3's readout was under-tuned on NoMAD.** That readout scores 0.31 and 0.54
  on the two sessions, which understates what a linear decoder can do there
  (0.54 and 0.65). U3's *choice* of
  chain is unaffected (section 4.1), but its absolute NoMAD accuracies should
  be read with this in mind.
- **Steady-state Kalman gain.** The first few bins of each trial use the
  converged gain rather than the larger transient one.

## 5. Prediction checks

Each changes one decoder setting and refits. Write down your prediction before
opening the answer.

### Check 1: a Kalman filter with no delay

```
make u4-key ARGS="--kalman-delays 0 --dataset nomad_session1"
```

On NoMAD session 1 the filter chose a delay of 4 bins (240 ms) and scored
0.623. Force the delay to zero. How much does it lose?

<details>
<summary>Answer</summary>

R² falls from 0.623 to **0.403** [0.367, 0.431]: a third of the accuracy, from
one integer.

The observation model `y[t] = C x[t]` says the features *now* reflect the
behaviour *now*. If the neural activity actually reflects the behaviour 240 ms
later, `C` is fitted between mismatched pairs and comes out weak, and the gain
that follows from it is small. The FIR decoders never have this problem: their
taps span the delay and the fit puts the weight where it belongs. An IIR
decoder with an instantaneous observation model has to be told.

Session 1 needs four times the delay of session 2. This fits the cursor having
been under closed-loop control in parts of session 1, where the cursor command
is itself a smoothed, and therefore delayed, function of neural activity. The
data file does not let this be confirmed per bin.
</details>

### Check 2: a GRU with 8 hidden units instead of 64

```
make u4-key ARGS="--gru-hidden 8 --dataset nason"
```

On Nason the 64-unit GRU scores 0.713 on the conventional chain and 0.720 on
U3's choice. Does an 8-unit GRU still beat the linear decoders? Does U3's
choice still match the conventional chain?

<details>
<summary>Answer</summary>

It still beats them, and U3's choice no longer matches:

- Conventional chain (crossings): 0.713 → **0.669** [0.644, 0.693].
- U3's choice (SBP at 0.25 µA): 0.720 → **0.571** [0.541, 0.601].
- Paired difference: **-0.098** [-0.128, -0.071].

Eight units are enough to learn the gating that beats the linear decoders
(0.669 against 0.478). They are not enough for the noisier feature: SBP at
0.25 µA carries added amplifier noise in all 96 channels, and averaging it away
takes hidden units as well. With 64 units there is room for both jobs.

The lesson for the co-design: decoder capacity and front-end noise trade
against each other. A cheap front end assumes a decoder with room to clean up
after it. Since section 4.5 shows the 64-unit decoder is nearly free, that is
the right trade, but it is a dependency and should be stated.
</details>

### Check 3: a Wiener filter with a full second of history

```
make u4-key ARGS="--wiener-history 1.0 --dataset nomad_session2"
```

On NoMAD session 2 the Wiener filter, with 0.5 s of history, ties the Kalman
filter and the GRU at 0.650. Double its history. Does it pull ahead?

<details>
<summary>Answer</summary>

Yes: 0.650 → **0.686** [0.591, 0.752], above the Kalman filter (0.653) and the
GRU (0.655), though all three intervals overlap heavily.

A plain linear decoder with enough history, properly penalised, is a hard
baseline to beat on smooth cursor data. The default stays at 0.5 s because the
fit time grows with the cube of the number of weights (192 channels × 17 taps
is 3264 weights per output, and `make u4` would take several times longer),
and because the conclusion of section 4.1 does not depend on
it: at 1 s the cheap chain loses 0.010 [0.004, 0.016].
</details>

### Check 4 (bonus): long history without retuning the penalty

```
make u4-key ARGS="--ridge-lags 20 --dataset nomad_session1"
```

Give U3's ridge readout 20 bins of history instead of 4, keeping its penalty at
0.01. It scored 0.305. More information can only help, surely?

<details>
<summary>Answer</summary>

R² goes from 0.305 to **-0.754** [-0.861, -0.513]: far worse than predicting
the mean.

With 20 taps on 192 channels the readout has 3840 weights per output, and
neighbouring taps of the same channel are almost copies of each other. The
least-squares problem is then nearly singular: enormous positive and negative
weights on neighbouring taps cancel on the training data and explode on new
data. The penalty is what forbids that, and 0.01 is a hundred times too small
for this many correlated inputs (cross-validation chooses 1). The Wiener filter
is this same readout with the penalty chosen properly, and it scores 0.539.

MATH2099 name for the problem: an ill-conditioned normal-equations matrix.
</details>

## 6. What to read next

- Wu et al. (2006), sections 2 and 3: the Kalman filter as a decoder, and the
  delay between neural activity and movement.
- Malik et al. (2011): why the steady-state gain loses almost nothing, with
  timing measurements.
- Cho et al. (2014), section 2.3: the GRU's two gates, in the paper that
  introduced it.
- Nason et al. (2020), the closed-loop experiments: what the offline numbers
  here cannot tell you.

Full citations are in [references.md](references.md).
