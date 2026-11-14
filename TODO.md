# NeuroChain build units

Units are done in order. Each ends at its done-criterion, not on a date.
After each unit: commit, update README.md and this file, write the summary.

Every unit also ships its learning artifacts: `docs/Ux_design.md`, a script or
test that prints the key number, and three prediction checks with measured
answers.

## U0: repo and data — DONE

Done-criterion: spectra and spike-amplitude figures reproduce from one command.

- [x] Package layout (`src/neurochain`), uv environment, Makefile
- [x] ruff, mypy, pytest, pre-commit, GitHub Actions workflow
- [x] Dataset registry and verified downloader
- [x] Neuralink loader (int16 to 10-bit code, saturation flag)
- [x] Seeded synthetic recording with ground truth (test fixture)
- [x] Band-split: LFP below 300 Hz, spikes 300 Hz to 7.5 kHz
- [x] Welch spectra, MAD noise estimate, threshold spike detection
- [x] `make u0` writes `figures/u0_spectra.png`, `figures/u0_spike_amplitudes.png`,
      `results/u0_metrics.json` and the README table
- [x] `docs/U0_design.md` with four prediction checks; `make u0-key`

Key number: spike-band noise σ = 7.01 LSB; median detected spike 43.7 LSB = 5.55σ.

Carried forward from U0:

- [x] **Nason data:** files downloaded, SHA-256 pinned in `registry.py`, loader
      in `neurochain.data.nason` with tests (`tests/test_nason.py`, and
      `tests/test_nason_data.py` on the real files). Ready for U2.
- [x] **NoMAD data:** files downloaded and verified, loader in
      `neurochain.data.nomad` with tests (`tests/test_nomad.py`, and
      `tests/test_nomad_data.py` on the real files). Ready for U4.
- [x] **NoMAD description checked:** the plan was right. Both files record one
      human participant (BrainGate T11, *Homo sapiens*) with 192 channels.
      `make data-describe` prints this. The monkey wording comes from the
      paper's abstract, which Dryad reuses; this release is the human data.

What opening the files showed, for the units that use them:

- [ ] **U6 cannot age NoMAD waveforms, because there are none.** NoMAD holds
      threshold crossings in 20 ms bins: no broadband, no spike shapes, no
      electrode table, no impedance. Impedance, noise and amplitude drift
      therefore have to be modelled as changes to crossing counts (for example
      through an amplitude-distribution model fitted on Neuralink or Nason
      broadband), or the aging model has to be validated on broadband data and
      only its effect carried to NoMAD. Decide this in `docs/U6_design.md`.
- [ ] **NoMAD is two sessions 273.9 days apart, and `session2` is the earlier
      one.** Held-out-session evaluation means train on one, test on the other.
      U6's "R² vs simulated weeks" comes from the aging model; the real pair
      gives one measured point at about 39 weeks to compare it with.
- [ ] **NoMAD time is not continuous:** 255 contiguous segments in `session1`
      (some one bin long) and 16 in `session2`. Decoders and any filtering must
      respect `NomadSession.segment_starts`. Only about 37 % of bins fall
      inside a trial.
- [ ] **NoMAD `cursor_vel` is not the derivative of `cursor_pos`** (correlation
      about 0.1 to 0.2 with the finite difference). Work out what it is before
      choosing the decoding target in U4.
- [x] **Nason online SBP is a sum over a varying number of samples** (0 to 4 per
      1 ms bin, mean 2.0). `neurochain.analysis.u2.online_features` divides by
      the sample count in each 50 ms bin. Its unit is not stated.
- [x] **Nason in-vivo broadband is the only µV-scaled real recording so far**
      (10 s, one channel, 30 kHz, three sorted units). Used in U1: its
      spike-band noise is 13.2 µV, 4.4 times the front-end model's floor.
- [x] **CI on GitHub:** green on `github.com/dvntekim/neurochain`, including the
      U1 commit (d248678), where the ngspice tests ran on Linux with ngspice 45
      and passed.
- [ ] **CI housekeeping:** GitHub warns that `actions/checkout@v4` and
      `astral-sh/setup-uv@v6` target the deprecated Node 20 runtime. Bump both
      when newer majors are confirmed.
- [x] **Causal filtering:** U0's figures use zero-phase filters. A causal filter
      of the same order loses 60 % of threshold crossings. From U2 on every
      implant path is causal. U2 found the loss belonged to that 4th-order
      filter: with a 2nd-order high-pass, causal detection scores F1 0.63
      against 0.66 zero-phase.

## U1: electrode and front-end model — DONE

Done-criterion: simulated noise within 10 % of hand calculation.

- [x] ngspice installed (version 47); CI installs it too
- [x] Randles electrode in ngspice, with the measured constant-phase double
      layer as an RC ladder; impedance sweep against the analytic model
- [x] Capacitive-feedback amplifier model; noise budget and NEF
- [x] Hand calculation, analytic network model and ngspice agree within 0.3 %
      for three electrode models (`tests/test_frontend.py`)
- [x] AC versus DC coupling: offset tolerance, ADC bits, settling after a step
- [x] Causal, seeded front-end emulator; checked on silence and on the Nason
      in-vivo recording
- [x] `make u1` writes four figures, `results/u1_metrics.json`, the README
      table and the reference netlists in `spice/`
- [x] `docs/U1_design.md` with the worked hand calculation and four prediction
      checks; `make u1-key`

Key number: front-end noise floor 3.41 µV rms in the spike band (electrode
2.51 µV, amplifier 2.31 µV at 18 µW).

Carried forward from U1:

- [x] **No public dataset here has both broadband and behaviour.** (Resolved in
      U3 by the feature-level bridge, `neurochain.bridge`.) The emulator
      needs a signal in microvolts. Neuralink is broadband without a voltage
      scale or behaviour; Nason's behaviour file and NoMAD hold features, not
      broadband. So the front end cannot simply be swept in front of a decoder.
      U3 needs a deliberate bridge: measure how front-end settings change the
      features (threshold crossings, SBP) on broadband with ground truth
      (MEArec, Nason in vivo), then apply that change to the feature datasets.
      Decide this at the start of U3, or find a dataset that has both.
- [x] **U3 must include neural background noise.** (Done: the bridge is
      measured at 13 µV of background.) In vivo the noise is four
      times the front end's floor, and a quarter of the bias current costs 4 %
      in σ. A power model that sees only front-end noise would overspend on the
      amplifier.
- [x] **Aliasing belongs to U2's ADC model.** Done: the emulator's
      `alias_zones` folds amplifier noise from above the Nyquist frequency back
      in, and the ADC samples with no digital filter in front.
- [ ] **No 1/f noise in the amplifier model.** Small in the spike band (about
      4 % of noise power for a 100 Hz corner, hand estimate), large below
      300 Hz. Add it before any LFP-band result.
- [x] **Rails are a hard clip.** U5 needs saturation recovery.
- [ ] **Harrison's supply voltage is unverified.** The 5 V that gives 16 µA and
      NEF 4.0 is from memory; the full text was not reachable. Check the paper.

## U2: ADC and features — DONE

Done-criteria: detection F1 on MEArec across noise levels; SBP versus threshold
crossings reproduced on Nason data.

- [x] MEArec recording with ground-truth spikes (`make mearec`; 32 channels,
      16 units, 60 s; the far-neuron background is kept separate so its level
      can be set)
- [x] ADC model: sample rate, bits and input range referred to the electrode;
      sampling aliases, noise from above Nyquist is folded in
- [x] Bit-depth and sample-rate sweeps, with bootstrap intervals over channels
- [x] Threshold and NEO detectors and spiking-band power, all causal
- [x] Fixed-point versions of each path, matching floating point
- [x] SBP versus threshold crossings reproduced on the Nason data (simulation,
      in vivo, and decoding from the online data)
- [x] `make u2` writes four figures, `results/u2_metrics.json` and the README
      table; `docs/U2_design.md` with four prediction checks; `make u2-key`

Key number: causal threshold detection at in-vivo noise (13 µV) scores F1 0.63
at 30 kS/s and 10 bits, and still 0.61 at 6 bits or at 10 kS/s.

Carried forward from U2:

- [x] **The ADC's input range has to be chosen on real data.** (U3 measured it:
      6.4 % of the Nason channel's samples fall outside ±1 mV on the wideband
      chain, none on the narrowband one.) MEArec has almost
      no LFP. The Nason broadband has an RMS of 972 µV, which would clip the
      ±1 mV range used here. U3 must choose range and analog high-pass corner
      together: U1 showed a 300 Hz analog corner costs 12 % in amplifier noise.
- [ ] **NEO was not retuned per ADC setting**, which is why its sweep curves are
      irregular. If U3 keeps NEO as a candidate, tune its factor at each sample
      rate and resolution. On F1 it has not beaten the plain threshold.
- [x] **U3's bridge from front end to decoder** (see U1's carried-forward item):
      U2 gives the first half. Detection F1 and feature tracking are now known
      as functions of noise, bits and sample rate. The second half is turning
      those into a change in the Nason and NoMAD features.
- [ ] **RTL parameters are fixed** for U7: 14 fractional bits and 6 guard bits
      at 30 kS/s (29-bit accumulator); 8 and 2 at 2 kS/s (18-bit accumulator);
      round half up; numerator `b0*(1 - z^-1)^2`. The Python in
      `neurochain.fixedpoint` is the reference the RTL must match bit for bit.
- [ ] **MEArec with NEURON 9:** two legacy synapse mechanisms are patched in a
      copy under `data/mearec/` by `scripts/make_mearec.py`. If MEArec or NEURON
      change version, check the patch still applies.

## U3: power model and co-design (A) — DONE

Done-criterion: accuracy-versus-µW Pareto plot on held-out sessions.

- [x] Per-stage power model: amplifier, ADC, digital logic, radio, with stated
      constants (`neurochain.power`)
- [x] Bridge from a chain configuration to a change in recorded features,
      measured on MEArec and checked there (`neurochain.bridge`)
- [x] Sweep of 151 configurations (36 infeasible), decoded on the Nason online
      data and both NoMAD sessions with blocked cross-validation
- [x] Pareto plot, power breakdown and bridge figures; `results/u3_metrics.json`;
      README table; `docs/U3_design.md` with four prediction checks; `make u3-key`

Key number: 0.48 µW per channel (SBP, 250 nA amplifier, 8 bits, 2 kS/s) decodes
finger velocity within 5 % of a 19.6 µW conventional chain.

Qualification on the done-criterion: "held-out" is held-out blocks of trials
within a session. Across the two NoMAD sessions a fixed decoder scores R² of
about zero.

Carried forward from U3:

- [ ] **The winning path for RTL (U7) is SBP:** narrowband amplifier, 8-bit ADC
      at 2 kS/s, high-pass, absolute value, sum over 50 ms. Its fixed-point
      parameters are in U2's carried-forward list.
- [ ] **The amplifier is 93 % of the winner's power.** Further savings need a
      lower NEF, not a better ADC or radio. U5 and U6 should report their costs
      in the same µW terms so they can be added to this budget.
- [ ] **NEF is assumed constant from 25 nA to 20 µA.** Check the 250 nA operating
      point against a published sub-microwatt amplifier before quoting 0.48 µW
      outside this project.
- [ ] **The bridge was checked at one current (0.5 µA) on simulated data.** It is
      slightly pessimistic for crossings and slightly optimistic for SBP.
- [x] **U4 must repeat the SBP-versus-crossings comparison with the Kalman
      filter and GRU** (done in U4: the choice survives every decoder), at least at the conventional chain and the winner.
- [ ] **U6: a fixed decoder does not transfer across NoMAD's sessions**
      (R² -0.01 and -0.07). No front-end choice fixes that.
- [ ] **The wideband chain's ADC range is too small for real LFP**, which the
      sweep did not charge it for. A fair crossings chain needs either more bits
      or a higher analog corner.

## U4: decoders and evaluation — DONE

Done-criterion: R² per configuration with confidence intervals.

- [x] Kalman filter decoder (steady-state gain, displacement in the state,
      delay chosen by cross-validation; `decode/kalman.py`)
- [x] Small GRU decoder (64 units; trains on the CPU, which was faster than MPS
      at this size; `decode/gru.py`)
- [x] Wiener filter added as the tuned linear baseline (`decode/ridge.py`)
- [x] Held-out splits: contiguous blocks of trials within a session, and a whole
      held-out NoMAD session; 95 % intervals from resampling trials, paired
      intervals on differences (`decode/evaluate.py`)
- [x] `make u4` writes three figures, `results/u4_metrics.json` and the README
      table; `docs/U4_design.md` with four prediction checks; `make u4-key`

Key number: on Nason the 0.48 µW chain keeps 93 to 101 % of the conventional
chain's R² under every decoder; with the GRU it scores 0.720 [0.695, 0.744]
against 0.713 [0.689, 0.737].

Carried forward from U4:

- [ ] **Offline decoding only, not closed-loop.** Offline R² ranks decoders
      imperfectly; the GRU's 0.28 lead over linear decoders on Nason may shrink
      online.
- [ ] **The GRU result depends on its size.** With 8 hidden units instead of 64,
      U3's choice loses 0.098 against the conventional chain (prediction
      check 2). "Survives every decoder" holds for the decoders as configured.
- [ ] **U6: cross-session decoding.** Fitted on one NoMAD session and tested on
      the other, the best decoder (Wiener) scores R² 0.24; the GRU and Kalman
      filter transfer worse. U6 tests why, together with a stabiliser.

## U5: stimulation artifacts (B) — DONE

Done-criterion: decoding loss with and without cancellation, plus recovery time.

- [x] Artifact model: point-source propagation in resistive tissue (`stim/propagation.py`), charge-balanced pulses (`stim/pulse.py`), front-end saturation and recovery in ngspice and a matching Python model (`frontend/saturation.py`)
- [x] Injection into real recordings: the Nason in-vivo channel, MEArec and the Nason SBP decoding data, three stimulation geometries
- [x] Fixed-point adaptive cancellation: multiplier-free template canceller and blanking baseline (`stim/cancel.py`), integer version within 0.29 LSB of floating point
- [x] `docs/U5_design.md` with prediction checks, `make u5-key`, `make u5` writes four figures, `results/u5_metrics.json` and the README table

Key number: with the stimulating electrode on the recording array, the 0.48 µW chain
plus a 0.004 µW template canceller decodes at R² 0.707 (GRU refitted with stimulation
on) against 0.718 without stimulation; recovery from a 30 mV artifact takes 2.95 ms
(conventional chain 1.01 ms).

Carried forward from U5:

- [ ] **Decoders must be calibrated with stimulation on**; a GRU fitted without it loses 0.09
      to 0.37. U6 should test whether its stabiliser also absorbs this shift.
- [ ] **With a Wiener filter the 0.48 µW chain loses 0.072 R²** under same-array stimulation
      even refitted; the conventional chain loses nothing.
- [ ] **U7: the RTL reference for the canceller** is `neurochain.stim.cancel.cancel_template`
      (shift 4, 6 fractional template bits, 2 fractional output bits that replace the
      high-pass filter's guard bits, 16 taps at 2 kS/s). The stimulator must share the ADC
      clock: ±100 µs of jitter breaks the canceller.
- [ ] **The pulse shape is quoted from memory** of Flesher et al. (2016); the 1 % imbalance,
      1 ms discharge and 0.3 S/m conductivity are assumptions. Verify before quoting.
- [ ] **Not modelled:** lead-to-lead coupling inside the implant, a nonlinear pseudo-resistor,
      stimulus-evoked spikes, amplitude-modulated stimulation (one template per level).
- [ ] **The new ngspice transient tests have run only on macOS ngspice 47**; confirm CI
      (ngspice 45, Linux).

## U6: aging stress test (C)

Done-criterion: R² versus simulated weeks.

- [ ] Encapsulation and impedance drift model applied to NoMAD data
- [ ] Comparison with a simple stabiliser
- [ ] `docs/U6_design.md`, key-number script, three prediction checks

## U7: RTL

Done-criterion: bit-exact on 10 minutes of real data; LUTs, fmax and estimated
µW per channel reported.

- [ ] Install verilator, yosys, nextpnr-ecp5
- [ ] SBP extractor and artifact canceller in SystemVerilog
- [ ] Verilator tests, bit-exact against the fixed-point Python
- [ ] Yosys and nextpnr for ECP5 (LFE5U-25F)
- [ ] `docs/U7_design.md`, key-number script, three prediction checks

## U8: hardware design (for later)

Done-criterion: schematic, layout, BOM sorted by cost, bring-up test plan.

- [ ] KiCad design: ADS1299 board with an STM32 over SPI
- [ ] Saline phantom with a DAC-driven playback electrode
- [ ] Charge-balanced current stimulator
- [ ] Saline only, never a person. Battery or isolated USB for any on-body recording
- [ ] `docs/U8_design.md`, three prediction checks
