# NeuroChain: working rules for Claude sessions

NeuroChain is a portfolio-grade neurotech engineering project by a 2nd-year
Electrical Engineering student at UNSW. Build each unit to a finished, polished
standard. The owner learns by reverse-engineering complete work afterwards, so
the reasoning has to be written down, not just the code.

Plan, primer, skills matrix and frontier analysis:
https://claude.ai/artifact/JApV8T8qvqvoyqATbQ3LVF (a Claude Doc; read it with the
Docs connector, not a web fetch).

## Research question

What is the lowest-power neural sensing chain (µW per channel) that still decodes
well while the implant is stimulating and its electrodes are aging?

- **A (deep, the spine):** co-design front end + ADC + features + decoder under a
  power model; produce an accuracy-vs-µW Pareto surface; implement the winning
  feature path in RTL.
- **B (deep):** record while stimulating: artifact propagation, front-end
  saturation and recovery, charge-balanced stimulation, fixed-point adaptive
  artifact cancellation, then RTL.
- **C (thin evaluation module):** simulated electrode aging (impedance, noise,
  amplitude drift) plus held-out-session tests on the public NoMAD dataset. A
  stress test only.

## Owner context

- Courses done: ELEC2133 (analog), ELEC3104 (DSP), ELEC2141 (digital), MATH2099
  (linear algebra / statistics). Explain new ideas in terms of these.
- Target roles: neurotech startups (Neuralink, Paradromics, Synchron, Precision,
  Axo).
- Machine: MacBook M5 Pro, Apple Silicon. Use PyTorch MPS where useful.
- Prefer open-source, free tools.
- No FPGA board yet: synthesise for Lattice ECP5 (LFE5U-25F) with Yosys +
  nextpnr, no hardware in the loop.
- The hardware phase happens later at a makerspace. Always choose the cheapest
  viable parts.
- Safety: saline only, never a person. Battery or isolated USB for any on-body
  recording.

## Non-negotiable rules

1. **Never invent results.** Every number in the README comes from code in this
   repo. README tables are generated from `results/*.json` by
   `scripts/update_readme.py`; do not hand-edit text between the
   `<!-- BEGIN/END ... -->` markers. If something fails or underperforms, report
   it honestly, in the README and in the unit summary.
2. **Never commit data.** `data/` is git-ignored. Datasets arrive through
   `neurochain.data.download` (`make data`), verified by SHA-256.
3. **One command per figure.** Every figure is reproducible from a `make`
   target. Each figure states its dataset, split and units on the figure itself.
4. **Fixed random seeds.** Anything stochastic takes an explicit `seed` and uses
   `numpy.random.default_rng(seed)`. No global RNG state.
5. **Work unit by unit, in order** (U0 to U8, see `TODO.md`). A unit ends at its
   done-criterion, not on a date.

## Quality bar

- `src/` layout package `neurochain`, type hints and docstrings on every public
  function, pytest for every module.
- `ruff` (lint + format) through pre-commit, `mypy` on `src/`, GitHub Actions CI
  running lint, types and tests.
- Tests must not need downloaded data. Use `neurochain.data.synthetic` or tiny
  fixtures built in `tests/`. Tests that need real data are marked
  `@pytest.mark.data` and skip when the data is absent.
- Units go in names or docstrings (`fs_hz`, `_uv`, `_lsb`). Never leave a
  physical quantity unitless.

## Learning artifacts (required for every unit)

1. `docs/Ux_design.md`: the EE reasoning, alternatives considered, why each
   choice won, and which UNSW course concept it uses.
2. A test or script that prints the unit's key number (`make uX-key`).
3. Three prediction checks: a parameter to change plus the question "what do you
   expect to happen?", with the answer in a collapsed `<details>` section. The
   answers must be numbers measured by running the code, not guesses.

## After each unit

1. Commit (conventional message, e.g. `U0: repo, loaders, band-split`).
2. Update `README.md` (results table, figures) and `TODO.md`.
3. Give the owner a short summary: key number, what was surprising, what to read
   next.
4. Cite every paper relied on in `docs/references.md`, with a link.

## Session economy

Cost follows the number of tool calls times the context each one re-reads, so:

- One unit per session. Start a fresh chat for the next unit; the state lives in
  `TODO.md`, the design notes, `results/*.json` and the session memory, not in
  the old transcript.
- Batch: write the files, then lint and test in one command.
- Look at a figure through a downscaled copy
  (`sips -Z 900 figures/x.png --out <scratchpad>/x.png`), once per layout change.
- Grep for what is needed; do not print whole files or whole result tables.
- README prose, TODO ticks and other housekeeping go to a cheaper model as one
  aggregated prompt. Generated tables (`make readme`) stay with the build.

None of this relaxes the rules above.

## Commands

```
make setup     # uv sync + pre-commit hooks
make data      # download / verify datasets
make data-describe  # print what the Nason and NoMAD files contain
make u0        # data + U0 figures + key number + README table
make u1        # U1 figures, metrics, netlists, README table (needs ngspice + Nason data)
make u1-key    # U1 key numbers; ARGS="--bias-scale 0.5" changes one parameter
make mearec    # generate the MEArec ground-truth recording (optional `sim` dependency group)
make u2        # U2 figures, metrics, README table (needs `make mearec` + Nason data)
make u2-key    # U2 key numbers; ARGS="--bits 6" changes one setting
make u3        # U3 sweep, figures, metrics, README table (about four minutes)
make u3-key    # U3 key numbers; ARGS="--nef 2" changes a power-model constant
make u4        # U4 decoders, figures, metrics, README table (about three minutes)
make u4-key    # U4 key numbers; ARGS="--gru-hidden 8" changes one decoder setting
make u5        # U5 figures, metrics, netlist, README table; about a minute; needs ngspice, `make mearec`, Nason data
make u5-key    # U5 key numbers; ARGS="--adapt-shift 1" changes one setting; run one at a time, 16 GB RAM
make u6        # U6 aging sweep, figures, metrics, README table; about four minutes; needs `make mearec`, Nason data, both NoMAD sessions
make u6-key    # U6 key numbers; ARGS="--shift 6" changes one setting; run one at a time, 16 GB RAM
make test      # pytest
make lint      # ruff + mypy
```

Run Python through uv: `uv run python ...`, `uv run pytest`.

## Data access facts (checked 2026-11-13)

- **Neuralink compression challenge:** the original bucket
  (`content.neuralink.com/compression-challenge/`) returns 404. The downloader
  falls back to the Internet Archive snapshot of `data.zip` and pins its
  SHA-256. 743 mono 16-bit WAV clips, 19 531 Hz, about 5 s each. No voltage
  scale is published, so amplitudes are reported in ADC LSB and in multiples of
  the noise σ, never in µV.
- **NoMAD (Dryad, doi:10.5061/dryad.q83bk3jtp):** two NWB files (`session1.nwb`,
  `session2.nwb`, 247 MB total). File downloads need a Dryad bearer token
  (`DRYAD_TOKEN`) or a manual browser download; checksums come from the Dryad
  API. Verified by opening the files: one human participant (BrainGate T11,
  *Homo sapiens*), 192 channels, threshold crossings in 20 ms bins plus cursor
  kinematics. No broadband and no electrode table. `session2` was recorded
  first; `session1` follows 273.9 days later. Loader: `neurochain.data.nomad`.
- **Nason SBP data (Deep Blue, doi:10.7302/wwya-5q86):** three MATLAB v5 `.mat`
  files plus a README, 74 MB, CC BY-NC 4.0. Deep Blue sits behind a Cloudflare
  challenge, so it is a manual browser download, verified against hashes pinned
  from the first copy. Do not try to script around the challenge. Only
  `MultiUnitInVivo.mat` has broadband (10 s, one channel, 30 kHz, in µV); the
  online file is 96 channels of 1 ms SBP and threshold crossings with finger
  position. Loader: `neurochain.data.nason`.
- **MEArec:** generated locally by `make mearec` into `data/mearec/` (seeded,
  about a minute). MEArec, NEURON and LFPy are in the optional `sim` dependency
  group, so CI and the default environment do not install them. Two legacy
  NEURON mechanisms are patched in a copy by the script. The loader reads a
  plain `.npz` and needs none of them.

## Modelling conventions (set in U1)

- **Units:** from U1 on, work in real microvolts. The Neuralink data stays in
  ADC LSB and cannot go through the front-end emulator, which refuses anything
  not in µV.
- **"Front-end noise" means** noise integrated over 300 Hz to 7.5 kHz, referred
  to the tissue signal (SPICE's `inoise`). The NEF uses a different number
  (all frequencies), and a filtered recording gives a third. `docs/U1_design.md`
  section 3.3 has the table. Say which one a number is.
- **Electrode:** Randles with a constant-phase double layer (Straka et al.,
  2018), never the ideal-capacitor version, which understates noise 2.6 times.
- **Filters in anything implant-realistic are causal.** The emulator's are.
- **SPICE:** netlists are generated by `neurochain.frontend.spice`; the copies
  in `spice/` are written by `make u1` and a test keeps them in sync. Tests that
  need ngspice skip without it.

## Evaluation conventions (set in U2)

- **Noise level** means the spike-band RMS of the neural background, in µV.
  13 µV is the in-vivo reference (the Nason channel).
- **Detection scoring:** a target spike comes from a unit at least 50 µV deep on
  that channel; detections of smaller real units are set aside; matches within
  1 ms; thresholds calibrated on the first 10 s and scored on the rest; F1
  pooled over channels with a bootstrap interval over channels.
- **Baseline ADC:** 30 kS/s, 10 bits, ±1 mV at the electrode.
- **Heavy sweeps run in worker processes** (`U2Config.max_workers`); tests set
  it to 1.

## Co-design conventions (set in U3)

- **Power per channel** comes from `neurochain.power`: amplifier `V·I`, ADC
  `FoM·2^bits·fs`, logic per multiply, radio per bit. The constants are
  assumptions; conclusions must survive tenfold changes (`make u3-key`).
- **Reference chain:** crossings, 10 µA, 10 bits, 30 kS/s (19.6 µW). Recorded
  feature datasets are assumed to come from it.
- **The bridge** (`neurochain.bridge`) carries a chain's effect onto recorded
  features: crossings are thinned by the share of real spikes kept (measured on
  MEArec against ground truth); SBP gets added noise power and jitter. Do not
  judge a chain by its correlation with another chain's features: that
  penalises being different, not being worse.
- **Winning path:** SBP from a 300 Hz to 1 kHz amplifier at 250 nA, 8 bits,
  2 kS/s: 0.48 µW per channel. This is what U7 implements.
- **Held-out** means contiguous blocks of trials within a session. NoMAD's two
  sessions do not support cross-session evaluation with a fixed decoder.

## Decoding conventions (set in U4)

- **Four causal decoders** in `neurochain.decode`, sharing one fit/predict
  interface: U3's ridge readout and the Wiener filter (`ridge.py`), a
  steady-state Kalman filter (`kalman.py`), a 64-unit GRU (`gru.py`).
  Evaluation lives in `evaluate.py`.
- **Held-out** means five contiguous blocks of trials, each predicted by a
  decoder fitted on the other four; feature scaling uses the training blocks
  only. Cross-session (one NoMAD session to the other) is reported separately.
- **R² is pooled** over all held-out bins, averaged over the behaviour's
  dimensions, not a mean of per-fold R².
- **Intervals come from resampling trials** (1000 resamples, 95 %), not bins:
  neighbouring bins are correlated. NoMAD resamples recording segments.
- **Comparisons use paired differences**: both sides scored on the same
  resampled trials, difference taken inside each resample.
- **The Wiener filter, not U3's ridge readout, is the linear baseline**: U3's
  readout has a fixed history and penalty and is under-tuned on NoMAD.
- **The GRU trains on the CPU.** At this size MPS was slower.

## Stimulation conventions (set in U5)

- **Artifact amplitude** means the peak potential of the first phase at the
  electrode, in mV. Reference pulse: 60 µA, 200 µs cathodic, 100 µs gap, 400 µs
  anodic at half current, 100 pulses per second, monopolar, distant reference.
- **Recovery time** runs from the end of the second phase until the output stays
  within ±13 µV referred to the input.
- **The stimulator runs off the ADC clock**; pulses start on a sample edge.
- **Large-signal front-end behaviour** comes from `neurochain.frontend.saturation`,
  which a test keeps in agreement with ngspice; `spice/u5_saturation.cir` is
  written by `make u5`.
- **Treatments** are "none", "blank" and "template"; scores are R² on stimulated
  bins, with paired differences against no stimulation, and say whether the
  decoder was fitted with or without stimulation.
- **Report canceller cost** in µW per channel (0.004 µW at 2 kS/s).

## Aging conventions (set in U6)

- **Aging model:** a series sheath on the Randles electrode solved for a 2.48-fold
  rise of |Z| at 1 kHz within two weeks, plus a 2.4 % per month mean amplitude
  decline with a per-channel rate. All parameters are assumptions; conclusions
  must survive the nine variations in `u6.variations`.
- **"Simulated weeks"** means a decoder fitted at week 0 and tested on held-out
  folds aged by the model. Week 39 is the key week (NoMAD's real gap); week 0
  equals U4's number.
- **Three treatments are always reported together:** left alone, running
  z-score, refitted; `loss` is paired against week 0, `gain` against left alone.
- **The stabiliser** is `stabilise.normalise`: `shift = 10` (1024 bins), gain
  limit 4, cold start from the fit-time constants, decoder not refitted. Cost in
  µW per channel (0.00034).
- **Thresholds are recalibrated each session** as part of the baseline chain;
  fixed thresholds are a variation.
- **Simulated and real gaps are reported separately.** The real NoMAD pair is the
  only measured point; the latent alignment (10 axes, first 300 s unlabelled) is
  only scored there.

## Toolchain status

Installed: uv, git, gh, make, ngspice 47. Not yet installed (install when the
unit needs them, via Homebrew): verilator, yosys, nextpnr-ecp5 (U7), KiCad (U8).
