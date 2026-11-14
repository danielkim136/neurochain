# U3 design: power model and co-design

**Done-criterion:** accuracy-versus-µW Pareto plot on held-out sessions.
**Status:** met, with one honest qualification. "Held-out" here means held-out
blocks of trials within a session. Decoding across the two NoMAD sessions with
a fixed decoder fails outright (R² about zero), which is reported below and is
the subject of U6. `make u3` reproduces everything in about four minutes.

Every number below comes from `results/u3_metrics.json` or from
`scripts/u3_key_number.py`.

## 1. What U3 has to answer

The project's question, minus stimulation and aging: **how few microwatts per
channel can the sensing chain use and still decode?** U1 and U2 answered this
one stage at a time. U3 chooses the amplifier's current, the ADC's bits and
sample rate, and the feature together, because the stages interact: a feature
that needs a tenth of the bandwidth changes what the amplifier and the ADC have
to be.

## 2. The power model

Four stages, one line each.

| Stage | Model | Constant | Source |
|---|---|---|---|
| Amplifier | `P = V_DD · I` | 1.8 V; noise from NEF = 4.0 | Harrison and Charles (2003); U1 |
| ADC | `P = FoM · 2^bits · fs` | 50 fJ per conversion step | ten times van Elzakker et al. (2010), 4.4 fJ |
| Digital logic | `P = E_mult · multiplies/sample · fs` | 1 pJ per multiply | an order-of-magnitude assumption; U7 replaces it |
| Radio | `P = E_bit · bit rate` | 10 pJ per bit | near-term target in Sen et al. (2026) |

The amplifier's current buys noise through `e_n = NEF·√(4kT·U_T/I)` (U1). The
ADC's Walden figure of merit says each extra bit doubles the power, and so does
doubling the sample rate. The radio sends 4 bits per 50 ms for a crossing count
(80 bit/s), 12 bits for an SBP value (240 bit/s), or every sample when
streaming raw (300 kbit/s).

All four constants are assumptions. The test of a conclusion is whether it
survives changing them tenfold; section 6 does that.

*Course link, ELEC2133:* power as V·I, the noise-current trade.
*ELEC2141:* switching energy per operation.

## 3. The design space

| Knob | Values |
|---|---|
| Feature | threshold crossings; spiking-band power (SBP); both from one chain; raw stream |
| Amplifier current | 25 nA to 20 µA, ten steps |
| ADC bits | 6, 8, 10 |
| ADC rate | 10, 15, 30 kS/s for crossings; 2 kS/s for SBP |

151 combinations. 36 are infeasible: **the wideband amplifier cannot reach
7.5 kHz below 0.25 µA at any load.** Its transconductance is set by its current
(`gm = κ·I/4U_T`), and the feedback network alone loads it enough that at 25 nA
the bandwidth tops out at 1.2 kHz. The narrowband chain, which only needs
1 kHz, has no such floor in this range. This is a second way, besides noise,
that bandwidth costs current.

## 4. The bridge: from a chain to a change in recorded features

No dataset here has both broadband and behaviour (U1's finding). So each
configuration's effect on features is measured on MEArec, where truth is known,
and then applied to the recorded features.

**Threshold crossings.** Each wideband configuration runs on MEArec at 13 µV of
background noise. Against ground truth it gives two rates: detections that land
on a real spike, and detections that land on nothing. Relative to the reference
chain (10 µA, 10 bits, 30 kS/s, taken to be what made the recordings), a
configuration keeps a share `p_keep` of the real detections. Recorded counts
are thinned by `p_keep`, and any false detections beyond those that survive the
thinning are added as Poisson noise.

**Spiking-band power.** SBP is the mean absolute value of a band signal, and
mean-square values add. A worse chain adds noise power, so recorded SBP is
raised in quadrature by a multiple of the channel's own noise floor, and each
bin's average is made noisier by the amount averaging 70 independent samples
leaves. The noise ratio counts everything the narrowband chain passes: its
first-order skirts let through amplifier noise well outside 300 Hz to 1 kHz,
and the 2 kS/s sampler folds all of it back.

### Three bridge designs, two of them wrong

This took three attempts, and the failures are worth keeping.

1. **First:** thin by the ratio of real-detection rates; raise SBP
   deterministically. *Wrong for SBP:* raising a feature by a fixed formula adds
   no randomness, so SBP's accuracy did not fall at all, even at 25 nA.
2. **Second:** measure each chain's correlation with the reference chain's
   crossings and thin to match. *Wrong for crossings:* this penalises a chain
   for being different, not for being worse. An 8-bit ADC looked 13 % worse
   although U2 measured no change in F1, and doubling the amplifier current
   looked harmful. Two equally good chains with independent noise disagree
   about marginal events; that is not information loss.
3. **Third (kept):** ground truth for crossings, stochastic noise for SBP, as
   described above.

### Checking the bridge where truth is known

On MEArec the cheaper chain can be run directly. At 0.5 µA, across all 32
channels:

| | Simulated directly | Bridge prediction |
|---|---|---|
| Crossings: correlation with true spike counts | 0.28 | 0.25 |
| Crossings: mean count per 50 ms bin | 0.39 | 0.45 |
| SBP: correlation with the reference chain's SBP | 0.968 | 0.979 |
| SBP: spread across bins | 1.39 µV | 1.37 µV |

The bridge is slightly pessimistic for crossings (random thinning loses more
information than losing the smallest spikes does) and slightly optimistic for
SBP. Neither error is large enough to change which feature wins.

*Course link, MATH2099:* binomial thinning, Poisson sums, variance of a mean.

## 5. What the numbers say

Decoding uses the ridge readout from U2 and 5-fold cross-validation over
contiguous blocks of trials.

| Dataset | Conventional chain | Cheapest within 5 % | Saving |
|---|---|---|---|
| Nason, finger velocity (96 ch) | R² 0.462 at 19.63 µW | **SBP, 0.25 µA, 8 b, 2 kS/s: R² 0.444 at 0.48 µW** | 40.6× |
| NoMAD session 1, cursor velocity (192 ch) | R² 0.336 at 19.63 µW | crossings, 2 µA, 8 b, 30 kS/s: R² 0.324 at 4.07 µW | 4.8× |
| NoMAD session 2, cursor velocity (192 ch) | R² 0.571 at 19.63 µW | crossings, 1 µA, 10 b, 30 kS/s: R² 0.543 at 3.43 µW | 5.7× |

1. **0.48 µW per channel.** A narrowband amplifier at 250 nA, an 8-bit ADC at
   2 kS/s, and SBP decode finger velocity within 5 % of a 19.6 µW conventional
   chain. That is the project's answer so far, before stimulation and aging.
2. **The saving comes from the feature, not from trimming.** On the same Nason
   data, the cheapest crossings-only match is 9.19 µW (5 µA, 6 bits, 30 kS/s):
   lower current and fewer bits buy a factor of 2.1. Switching to SBP buys
   another 19, because it needs a seventh of the bandwidth and a fifteenth of
   the sample rate. NoMAD has no recorded SBP, so there only crossings could be
   scored, and the saving stops at about 5×.
3. **The amplifier is 93 % of the winner's power.** ADC 5 %, logic 1 %, radio
   0.5 %. Everything after the amplifier has been made nearly free; the next
   gain has to come from the amplifier itself (a lower NEF).
4. **Most accurate is not cheapest.** Both features from one wideband chain
   reach R² 0.519, but at 19.8 µW. Sending both costs 41 times more than SBP
   alone for 17 % more R².
5. **Raw streaming is not much worse than the conventional chain here**:
   22.5 µW, of which the radio is 3.0 µW. At 10 pJ/bit the radio is not the
   bottleneck per channel. It becomes one when the constant is realistic for
   today's implants (check 2).
6. **A real channel does not fit the wideband chain's ADC.** On the Nason
   in-vivo broadband, 6.4 % of samples fall outside ±1 mV after the 1 Hz
   high-pass, because of LFP. Through the narrowband amplifier none do. The
   wideband chain would need more range, hence more bits, which this sweep does
   not charge it for. The comparison is therefore generous to crossings.
7. **A fixed decoder does not survive nine months.** Trained on one NoMAD
   session and tested on the other, 274 days apart, the same readout scores R²
   -0.01 and -0.07. No choice of front end fixes that. It is U6's problem.

### Limitations

- **The bridge is a model.** It carries a chain's measured effect on simulated
  features to recorded ones. It was checked on MEArec at one current, not on
  real data, because no real data permits the check.
- **MEArec at 13 µV stands for every dataset.** The Nason and NoMAD channels
  have their own noise and unit amplitudes, which are unknown.
- **The recording is assumed to come from the reference chain.** A configuration
  can only be made worse than the recording, never better.
- **NEF is held at 4.0 across three decades of current and two bandwidths.** A
  real amplifier's NEF moves with both. At 250 nA the model is an
  extrapolation.
- **Linear readout only.** U4 repeats the comparison with a Kalman filter and a
  GRU.
- **The 5 % tolerance is a choice.** At 2 % the Nason winner is SBP at 0.5 µA
  (0.93 µW, 21× less); at 10 % it is unchanged.
- **One Nason session.** The 40× rests on 8.9 minutes from one monkey.
- **NoMAD's decoding target is the cursor velocity of a closed-loop BCI**, which
  is that system's own decoder output. It measures how well features keep what
  that decoder used, not ground-truth intention.

## 6. Prediction checks

Each changes one constant of the power model and recomputes the cheapest match
without rerunning the sweep. Baseline: conventional 19.63 µW; cheapest match
SBP at 0.25 µA, 8 bits, 2 kS/s, 0.484 µW, 40.6× less.

### Check 1: an ADC ten times less efficient

```
make u3-key ARGS="--adc-fom-fj 500"
```

At 500 fJ per step instead of 50, does the winner change? Which chain suffers
more?

<details>
<summary>Answer</summary>

The winner is unchanged: SBP at 0.25 µA, 8 bits, 2 kS/s, now 0.714 µW. The
conventional chain goes from 19.63 to 33.45 µW, so SBP's advantage **grows**,
from 40.6× to 46.8×.

ADC power is `FoM · 2^bits · fs`. The conventional chain converts 10 bits at
30 kS/s: 1.5 µW at 50 fJ, 15.4 µW at 500 fJ. The SBP chain converts 8 bits at
2 kS/s, sixty times fewer conversion steps per second: 0.026 µW becomes
0.26 µW.

The lesson: a chain designed around a low sample rate is insured against a bad
ADC. The conventional chain is not.
</details>

### Check 2: a radio a hundred times less efficient

```
make u3-key ARGS="--pj-per-bit 1000"
```

At 1 nJ per bit, typical of a narrowband implant link today, what happens to
raw streaming, and to the feature chains?

<details>
<summary>Answer</summary>

Raw streaming goes from 22.5 to 319.5 µW per channel. The conventional
crossing chain barely moves (19.63 to 19.71 µW). The SBP winner goes from 0.484
to 0.722 µW, because its radio share rises from 0.5 % to a third.

Raw streaming sends 300 kbit/s per channel; a crossing count sends 80 bit/s and
an SBP value 240 bit/s. A thousand channels of raw data at 1 nJ/bit would be
320 mW, far past the roughly 10 mW a link can dissipate in tissue.

The lesson: on-implant feature extraction is not an optimisation but a
requirement, and once it is done the radio drops out of the budget. Note that
SBP sends three times the bits of a crossing count; at a costly enough radio
that starts to matter.
</details>

### Check 3: a better amplifier, NEF 2 instead of 4

```
make u3-key ARGS="--nef 2"
```

Halving the NEF means the same noise for a quarter of the current. What is the
cheapest match now, and does the ranking of features change?

<details>
<summary>Answer</summary>

The conventional chain falls from 19.63 to 6.13 µW, and the SBP winner from
0.484 to 0.146 µW per channel. The ratio stays at about 42×.

Amplifier power scales with NEF². Because the amplifier is 93 % of the winner's
power, a 4× cut in amplifier power is nearly a 4× cut in total (3.3×, the rest
being the ADC's now-larger share).

The lesson: the feature and the amplifier are independent levers, and they
multiply. This sweep exhausted the first. The second is circuit design.
</details>

### Check 4 (bonus): how much does the 5 % tolerance matter?

```
make u3-key ARGS="--tolerance 0.02"
make u3-key ARGS="--tolerance 0.10"
```

<details>
<summary>Answer</summary>

At 2 % the Nason winner is SBP at 0.5 µA, 0.934 µW (21× less). At 10 % it is
the same configuration as at 5 %, because the next step down in current (0.1 µA,
R² 0.405) loses 12 %.

On NoMAD, where only crossings can be scored, 10 % allows 2.0 to 2.3 µW (about
9× less) and 2 % needs 9.2 µW (2.1× less).

The lesson: SBP's accuracy has a flat top and then a cliff, between 0.25 and
0.1 µA, so its operating point is well defined. Crossings degrade steadily, so
their "cheapest match" slides with the tolerance. A design with a knee is easier
to commit to hardware.
</details>

## 7. What to read next

- van Elzakker et al. (2010), sections I and II: why a SAR converter's energy
  scales as `2^bits`, and what limits it.
- Sen et al. (2026): why the link's energy per bit decides how much an implant
  can send.
- Nason et al. (2020), the discussion: the authors' own power argument for SBP,
  made with a different amplifier.

Full citations are in [references.md](references.md).
