"""U6 analysis: does decoding survive an aging electrode, and what does a stabiliser buy?

Five steps:

1. **The model.** An aging model (:mod:`neurochain.aging`) changes the Randles
   electrode of U1 and shrinks the spikes, week by week.
2. **Response.** On the MEArec recording, where the spikes are known, the aged
   electrode goes through the front-end emulator. For each electrode state and
   each spike amplitude this measures how many of the week-0 detections
   survive. The SBP path is checked the same way.
3. **Decoding against weeks.** The measured change is carried onto the Nason
   and NoMAD features through the bridge of U3 (neither dataset has broadband
   with behaviour; NoMAD has no waveforms at all). U4's decoders are fitted at
   week 0 and tested on aged features, without help, with a running
   stabiliser (:mod:`neurochain.stabilise`), and refitted.
4. **The real pair.** NoMAD's two sessions lie 273.9 days apart. A decoder
   fitted on the earlier one is tested on the later one, with and without the
   stabiliser: one measured point to hold the simulation against.
5. **Stimulation.** Whether the same stabiliser absorbs the feature shift U5
   found when stimulation is switched on after the decoder was fitted.

Scoring follows U4: five contiguous folds of trials, R² pooled over held-out
bins, intervals from 1000 resamples of trials, and paired differences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import torch

from neurochain.adc import Adc, convert
from neurochain.aging import AgingModel, encapsulated
from neurochain.align import aligned_axes, principal_axes
from neurochain.analysis import u2, u3, u4, u5
from neurochain.bridge import age_sbp, degrade_crossings, sbp_floor, sbp_noise_power_ratio
from neurochain.data.mearec import MearecRecording
from neurochain.decode.evaluate import (
    Decoder,
    bootstrap_r2,
    bootstrap_r2_difference,
    held_out_predictions,
)
from neurochain.dsp.spikes import mad_sigma
from neurochain.features import detect as det
from neurochain.frontend.electrode import UTAH_PT_IN_VITRO, Electrode
from neurochain.frontend.emulator import emulate, reference_gain
from neurochain.frontend.model import SPIKE_BAND_HZ, FrontEnd
from neurochain.power import NARROWBAND_HZ, SBP_FS_HZ, ChainConfig, PowerModel, chain_power
from neurochain.recording import FloatArray, IntArray
from neurochain.stabilise import MODES, Mode, normalise, operations_per_bin

REFERENCE = u3.REFERENCE
BIN_S = u3.BIN_S
CHAINS = ("sbp", "crossings")
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

# One electrode state: (|Z| factor at 1 kHz, background factor, where the rise is put).
State = tuple[float, float, str]


@dataclass(frozen=True)
class U6Config:
    """Parameters of the U6 analysis.

    Attributes:
        seed: Seed for every stochastic step.
        weeks: Simulated weeks after implantation at which decoding is scored.
        key_week: The week quoted as the key number (39: NoMAD's real gap).
        aging: The aging model.
        amplitude_grid: Spike amplitudes, relative to week 0, at which the
            MEArec response is measured; values between are interpolated.
        noise_uv: Spike-band RMS of the neural background at week 0.
        rethreshold: Recalibrate each channel's detection threshold on the
            session's own first seconds, as U2's detector does. If false the
            week-0 threshold is kept forever.
        shift: Time constant of the stabiliser's registers, ``2**shift`` bins.
        max_gain: Largest amplification the running scale may apply.
        warm_start: Start the stabiliser from the aged session's own
            statistics, as if the implant had been used the day before. By
            default it starts from the week-0 constants.
        refit: Also refit every decoder at every week (the supervised
            recalibration a stabiliser tries to avoid).
        n_latents: Dimensions of the latent space in the alignment test. Ten
            is a customary size for motor-cortex population activity and an
            assumption here; the result is sensitive to it (prediction check 2
            in ``docs/U6_design.md``).
        align_calibration_s: Unlabelled seconds at the start of the later
            session from which its principal axes are estimated.
        sbp_current_a: Supply current of the SBP chain's amplifier (U3's winner).
        sbp_bits: Resolution of the SBP chain's ADC.
        check_amplitudes: Amplitudes at which the bridge is checked on MEArec.
        decoding: Decoder and scoring settings, as in U4.
        power: Technology constants of the power model.
    """

    seed: int = 0
    weeks: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0, 13.0, 26.0, 39.0, 52.0, 104.0)
    key_week: float = 39.0
    aging: AgingModel = AgingModel()  # noqa: RUF009 - frozen dataclass, safe as a default
    amplitude_grid: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
    noise_uv: float = 13.0
    rethreshold: bool = True
    shift: int = 10
    max_gain: float = 4.0
    warm_start: bool = False
    refit: bool = True
    n_latents: int = 10
    align_calibration_s: float = 300.0
    sbp_current_a: float = 0.25e-6
    sbp_bits: int = 8
    check_amplitudes: tuple[float, ...] = (1.0, 0.8, 0.5)
    decoding: u4.U4Config = u4.U4Config()  # noqa: RUF009
    power: PowerModel = PowerModel()  # noqa: RUF009


def variations(config: U6Config) -> dict[str, U6Config]:
    """Return the changed assumptions the conclusions have to survive."""
    model = config.aging
    rate = model.amplitude_decline_per_month

    def aged(**changes: Any) -> U6Config:
        return replace(config, aging=replace(model, **changes))

    return {
        "amplitude decline halved": aged(amplitude_decline_per_month=rate / 2.0),
        "amplitude decline doubled": aged(amplitude_decline_per_month=rate * 2.0),
        "every channel declines alike": aged(decline_spread=0.0),
        "spread between channels doubled": aged(decline_spread=2.0 * model.decline_spread),
        "no impedance rise": aged(impedance_factor_final=1.0),
        "impedance rises 8 times": aged(impedance_factor_final=8.0),
        "rise in the interface, not a sheath": aged(encapsulation="interface"),
        "background noise up 1.53 times": aged(background_factor_final=1.53),
        "thresholds fixed at week 0": replace(config, rethreshold=False),
    }


# ----------------------------------------------------------------------------- the model


def state_of(model: AgingModel, weeks: float) -> State:
    """Return the electrode state after ``weeks`` (rounded, so settled weeks coincide)."""
    return (
        round(model.impedance_factor(weeks), 2),
        round(model.background_factor(weeks), 2),
        model.encapsulation,
    )


def aged_electrode(state: State, fresh: Electrode = UTAH_PT_IN_VITRO) -> Electrode:
    """Return the electrode in a given state."""
    return encapsulated(fresh, state[0], "interface" if state[2] == "interface" else "series")


def frontends(config: U6Config, state: State) -> dict[str, FrontEnd]:
    """Return both chains' front ends with the electrode in a given state.

    The amplifiers are the ones designed for the fresh electrode; only the
    electrode in front of them changes.
    """
    electrode = aged_electrode(state)
    wide = u3.wideband_frontend(REFERENCE.i_total_a, config.power)
    narrow = u3.narrowband_frontend(config.sbp_current_a, config.power)
    return {
        "sbp": replace(narrow, electrode=electrode),
        "crossings": replace(wide, electrode=electrode),
    }


FRESH: State = (1.0, 1.0, "series")


def chain_state(
    config: U6Config, state: State, background: tuple[FloatArray, FloatArray]
) -> dict[str, float]:
    """Return what one electrode state does to each chain, from the U1 model.

    Returns:
        ``sbp_gain`` and ``crossings_gain``: gain at 1 kHz relative to the fresh
        electrode. ``sbp_noise_ratio``: noise power in the SBP signal relative
        to the reference chain with a fresh electrode, the quantity the bridge
        needs. ``electrode_noise_uv``, ``amplifier_noise_uv`` and
        ``frontend_noise_uv``: the conventional chain's noise over 300 Hz to
        7.5 kHz referred to the tissue signal. ``z_1khz_kohm`` and ``rs_kohm``
        describe the electrode.
    """
    aged, fresh = frontends(config, state), frontends(config, FRESH)
    freqs, psd = background
    factor = state[1] ** 2
    absolute = factor * sbp_noise_power_ratio(
        aged["sbp"], Adc(fs_hz=SBP_FS_HZ, bits=config.sbp_bits), freqs, psd * factor
    )
    reference = sbp_noise_power_ratio(
        u3.narrowband_frontend(REFERENCE.i_total_a, config.power),
        Adc(fs_hz=SBP_FS_HZ, bits=REFERENCE.bits),
        freqs,
        psd,
    )
    noise = aged["crossings"].band_noise(*SPIKE_BAND_HZ)
    electrode = aged_electrode(state)
    return {
        "sbp_gain": reference_gain(aged["sbp"]) / reference_gain(fresh["sbp"]),
        "crossings_gain": reference_gain(aged["crossings"]) / reference_gain(fresh["crossings"]),
        "sbp_noise_ratio": max(1.0, absolute / reference),
        "electrode_noise_uv": noise.electrode_v * 1e6,
        "amplifier_noise_uv": noise.lna_v * 1e6,
        "frontend_noise_uv": noise.total_v * 1e6,
        "z_1khz_kohm": electrode.magnitude_ohm(1000.0) / 1e3,
        "rs_kohm": electrode.rs_ohm / 1e3,
    }


def model_summary(
    config: U6Config, background: tuple[FloatArray, FloatArray], n_channels: int = 192
) -> dict[str, Any]:
    """Describe the aging model: impedance, noise and amplitude against weeks."""
    model = config.aging
    fine = np.linspace(0.0, max(config.weeks), 105)
    scale = model.amplitude_scale(fine, n_channels, _rate_seed(config))
    freqs = np.geomspace(10.0, 1.0e5, 81)
    settled = state_of(model, max(config.weeks))
    return {
        "weeks": list(config.weeks),
        "states": [
            {"week": week, **chain_state(config, state_of(model, week), background)}
            for week in config.weeks
        ],
        "fine_weeks": fine.tolist(),
        "impedance_factor": [model.impedance_factor(float(w)) for w in fine],
        "amplitude": {
            "mean": scale.mean(axis=1).tolist(),
            "p10": np.percentile(scale, 10, axis=1).tolist(),
            "p50": np.percentile(scale, 50, axis=1).tolist(),
            "p90": np.percentile(scale, 90, axis=1).tolist(),
            "n_channels": n_channels,
        },
        "spectrum": {
            "freqs_hz": freqs.tolist(),
            "fresh_ohm": np.abs(aged_electrode(FRESH).impedance(freqs)).tolist(),
            "settled_ohm": np.abs(aged_electrode(settled).impedance(freqs)).tolist(),
        },
    }


# ---------------------------------------------------------------------- MEArec response


@dataclass(frozen=True)
class Response:
    """What each electrode state and spike amplitude does to crossings on MEArec.

    Attributes:
        states: The electrode states measured; the first is the fresh one.
        amplitude_grid: Spike amplitudes relative to week 0.
        real: ``(2, n_channels, n_states, n_amplitudes)`` detections that land
            on a real spike. Index 0 of the first axis is with thresholds
            recalibrated, index 1 with the week-0 threshold kept.
        false: The same shape, detections that land on nothing.
        seconds: Scored seconds per channel.
        checks: Per-channel bridge checks (see :func:`_response_job`).
    """

    states: tuple[State, ...]
    amplitude_grid: tuple[float, ...]
    real: FloatArray
    false: FloatArray
    seconds: float
    checks: tuple[dict[str, float], ...]

    def tables(self, state: State, *, rethreshold: bool) -> tuple[FloatArray, FloatArray]:
        """Return per-channel ``(p_keep, extra_rate_hz)`` against amplitude.

        ``p_keep`` is the share of the fresh chain's real detections that the
        aged chain still makes; ``extra_rate_hz`` is the false detections it
        makes beyond those that survive the thinning (as in U3).
        """
        which, s = (0 if rethreshold else 1), self.states.index(state)
        fresh_s, fresh_a = self.states.index(FRESH), self.amplitude_grid.index(1.0)
        real0 = np.maximum(self.real[0, :, fresh_s, fresh_a], 1.0)[:, np.newaxis]
        false0 = self.false[0, :, fresh_s, fresh_a][:, np.newaxis]
        p_keep = np.clip(self.real[which, :, s, :] / real0, 0.0, 1.0)
        extra = np.maximum(self.false[which, :, s, :] - p_keep * false0, 0.0) / self.seconds
        return p_keep, extra

    def summary(self, *, rethreshold: bool) -> dict[str, Any]:
        """Return the tables pooled over channels, for the metrics file."""
        out: dict[str, Any] = {
            "amplitude_grid": list(self.amplitude_grid),
            "n_channels": int(self.real.shape[1]),
            "scored_s_per_channel": self.seconds,
            "states": [],
        }
        which = 0 if rethreshold else 1
        fresh_s, fresh_a = self.states.index(FRESH), self.amplitude_grid.index(1.0)
        for s, state in enumerate(self.states):
            p_keep, extra = self.tables(state, rethreshold=rethreshold)
            pooled = self.real[which, :, s, :].sum(axis=0) / self.real[0, :, fresh_s, fresh_a].sum()
            out["states"].append(
                {
                    "impedance_factor": state[0],
                    "background_factor": state[1],
                    "encapsulation": state[2],
                    "p_keep": np.minimum(pooled, 1.0).tolist(),
                    "p_keep_p10": np.percentile(p_keep, 10, axis=0).tolist(),
                    "p_keep_p90": np.percentile(p_keep, 90, axis=0).tolist(),
                    "extra_rate_hz": extra.mean(axis=0).tolist(),
                }
            )
        return out


ResponseJob = tuple[u2.ChannelJob, U6Config, tuple[State, ...], dict[State, dict[str, float]]]


def _stats(series: FloatArray, truth: FloatArray) -> dict[str, float]:
    constant = series.std() == 0.0 or truth.std() == 0.0
    return {
        "mean": float(series.mean()),
        "sd": float(series.std()),
        "r": 0.0 if constant else float(np.corrcoef(series, truth)[0, 1]),
    }


def _response_job(args: ResponseJob) -> dict[str, Any]:
    """Measure one MEArec channel in every electrode state and at every amplitude.

    The front end is linear below its rails, so the spikes and the background
    are passed through it separately and added afterwards at each amplitude.
    Outputs are referred to the input with the *fresh* chain's gain: the
    implant does not know its electrode has changed.

    Returns:
        ``real`` and ``false`` as in :class:`Response` (without the channel
        axis), and ``checks``: for the settled state and each check amplitude,
        the mean, spread and correlation with the true spike count of the
        directly simulated feature and of the bridge's prediction from the
        week-0 feature.
    """
    job, config, states, chain_states = args
    cfg, fs = job.config, job.fs_hz
    adc = Adc(fs_hz=REFERENCE.fs_hz, bits=REFERENCE.bits)
    sbp_adc = Adc(fs_hz=SBP_FS_HZ, bits=config.sbp_bits)
    seed = cfg.seed * 100_000 + job.channel * 100
    calibration = round(cfg.calibration_s * adc.fs_hz)
    fresh = frontends(config, FRESH)
    gain0 = {name: reference_gain(frontend) for name, frontend in fresh.items()}
    grid = config.amplitude_grid

    def parts(kind: str, state: State, offset: int) -> tuple[FloatArray, FloatArray]:
        frontend = frontends(config, state)[kind]
        spikes = emulate(job.clean_uv, fs, frontend, seed=0, add_noise=False)
        rest = emulate(
            config.noise_uv * state[1] * job.background_uv,
            fs,
            frontend,
            seed=seed + offset,
            alias_zones=cfg.alias_zones,
        )
        scale = 1.0e6 / gain0[kind]
        return spikes.output_v * scale, rest.output_v * scale

    def score_of(analog_uv: FloatArray) -> FloatArray:
        codes = convert(analog_uv, fs, adc).codes.astype(np.float64)
        return det.score_signal(det.causal_highpass(codes, adc.fs_hz), adc.fs_hz, "threshold")

    def threshold_of(score: FloatArray) -> float:
        return det.calibrate_threshold(score[:calibration], "threshold", cfg.threshold_factor)

    def sbp_of(analog_uv: FloatArray) -> FloatArray:
        sampled = convert(analog_uv, fs, sbp_adc).microvolts
        per_bin = round(BIN_S * SBP_FS_HZ)
        n = sampled.size // per_bin
        binned = np.abs(sampled[: n * per_bin]).reshape(n, per_bin).mean(axis=1)
        return np.asarray(binned[round(cfg.calibration_s / BIN_S) :], dtype=np.float64)

    real = np.zeros((2, len(states), len(grid)))
    false = np.zeros_like(real)
    fresh_parts = parts("crossings", FRESH, 77)
    fresh_score = score_of(fresh_parts[0] + fresh_parts[1])
    threshold0 = threshold_of(fresh_score)
    events: dict[tuple[State, float], IntArray] = {}
    for s, state in enumerate(states):
        spikes, rest = fresh_parts if state == FRESH else parts("crossings", state, 77)
        for a, amplitude in enumerate(grid):
            score = score_of(amplitude * spikes + rest)
            for which, threshold in enumerate((threshold_of(score), threshold0)):
                found = det.detect_events(score, threshold, adc.fs_hz)
                result = u2._score(job, found, adc.fs_hz)
                real[which, s, a] = result.true_positives + result.ignored
                false[which, s, a] = result.false_positives
                if which == (0 if config.rethreshold else 1):
                    events[(state, amplitude)] = found

    # Bridge check, at the settled state: does the bridge's prediction from the
    # week-0 feature look like the feature simulated directly?
    settled = state_of(config.aging, config.key_week)
    rng = np.random.default_rng([cfg.seed, job.channel, 6])
    truth = u3._binned_events(np.sort(np.concatenate((job.targets, job.ignore))), fs, job, BIN_S)
    which = 0 if config.rethreshold else 1
    s = states.index(settled)
    counts0 = u3._binned_events(events[(FRESH, 1.0)], adc.fs_hz, job, BIN_S)
    narrow0 = parts("sbp", FRESH, 50)
    narrow = narrow0 if settled == FRESH else parts("sbp", settled, 50)
    sbp0 = sbp_of(narrow0[0] + narrow0[1])
    n_effective = 2.0 * (NARROWBAND_HZ[1] - NARROWBAND_HZ[0]) * BIN_S
    floor0 = sbp_floor(sbp0[:, np.newaxis], n_effective)
    ratio = chain_states[settled]["sbp_noise_ratio"] / chain_states[FRESH]["sbp_noise_ratio"]
    checks: dict[str, float] = {}
    for amplitude in config.check_amplitudes:
        a = grid.index(amplitude)
        p_keep = min(1.0, real[which, s, a] / max(real[0, states.index(FRESH), grid.index(1.0)], 1))
        fresh_false = false[0, states.index(FRESH), grid.index(1.0)]
        extra = max(0.0, false[which, s, a] - p_keep * fresh_false) / (truth.size * BIN_S)
        series = {
            "crossings_actual": u3._binned_events(
                events[(settled, amplitude)], adc.fs_hz, job, BIN_S
            ),
            "crossings_predicted": degrade_crossings(
                counts0[:, np.newaxis], p_keep, extra, BIN_S, rng
            )[:, 0],
            "sbp_actual": sbp_of(amplitude * narrow[0] + narrow[1]),
            "sbp_predicted": age_sbp(
                sbp0[:, np.newaxis],
                amplitude,
                max(1.0, ratio),
                floor0,
                gain=chain_states[settled]["sbp_gain"],
                n_effective=n_effective,
                rng=rng,
            )[:, 0],
        }
        for name, values in series.items():
            for stat, value in _stats(values, truth).items():
                checks[f"{name}_{stat}_{amplitude:g}"] = value
    return {"real": real, "false": false, "checks": checks}


def aging_response(
    recording: MearecRecording,
    config: U6Config,
    states: tuple[State, ...],
    background: tuple[FloatArray, FloatArray],
    detection: u2.U2Config | None = None,
) -> Response:
    """Measure the MEArec response of the crossings chain in each electrode state."""
    detection = detection or u2.U2Config()
    ordered = (FRESH, *sorted(set(states) - {FRESH}))
    settled = state_of(config.aging, config.key_week)
    if settled not in ordered:
        raise ValueError("the settled state of the baseline model must be measured")
    chain_states = {state: chain_state(config, state, background) for state in ordered}
    jobs: list[ResponseJob] = [
        (job, config, ordered, chain_states) for job in u2.channel_jobs(recording, detection)
    ]
    results: list[dict[str, Any]] = u2._map(_response_job, jobs, detection)
    return Response(
        states=ordered,
        amplitude_grid=config.amplitude_grid,
        real=np.stack([r["real"] for r in results], axis=1),
        false=np.stack([r["false"] for r in results], axis=1),
        seconds=recording.duration_s - detection.calibration_s,
        checks=tuple(r["checks"] for r in results),
    )


def bridge_check(response: Response, config: U6Config) -> dict[str, Any]:
    """Average the per-channel bridge checks over channels."""
    out: dict[str, Any] = {
        "state": {
            "impedance_factor": state_of(config.aging, config.key_week)[0],
            "background_factor": state_of(config.aging, config.key_week)[1],
        },
        "n_channels": len(response.checks),
        "amplitudes": {},
    }
    for amplitude in config.check_amplitudes:
        entry: dict[str, dict[str, float]] = {}
        for feature in CHAINS:
            for source in ("actual", "predicted"):
                entry[f"{feature}_{source}"] = {
                    stat: float(
                        np.mean(
                            [c[f"{feature}_{source}_{stat}_{amplitude:g}"] for c in response.checks]
                        )
                    )
                    for stat in ("mean", "sd", "r")
                }
        out["amplitudes"][f"{amplitude:g}"] = entry
    return out


# --------------------------------------------------------------- aging recorded features


def _rate_seed(config: U6Config) -> int:
    return config.seed * 1000 + 11


@dataclass(frozen=True)
class Ager:
    """Everything needed to age one dataset's features on one chain.

    Attributes:
        kind: ``"sbp"`` or ``"crossings"``.
        weeks: The weeks this ager can produce.
        amplitude: ``(n_weeks, n_channels)`` spike amplitude relative to week 0.
        p_keep: ``(n_weeks, n_channels)`` share of crossings kept.
        extra_rate_hz: ``(n_weeks, n_channels)`` false crossings added.
        sbp_ratio: ``(n_weeks,)`` SBP noise power relative to the recording.
        gain: ``(n_weeks,)`` feature gain relative to the fresh electrode.
        seed: Seed of the bridge's random draws.
    """

    kind: str
    weeks: tuple[float, ...]
    amplitude: FloatArray
    p_keep: FloatArray
    extra_rate_hz: FloatArray
    sbp_ratio: FloatArray
    gain: FloatArray
    seed: tuple[int, ...]

    def features(self, data: u3.DecodingSet, index: int) -> FloatArray:
        """Return the dataset's features as the chain would deliver them at one week.

        The same random draws are used at every week, so a curve against weeks
        shows the aging and not the bridge's sampling noise.
        """
        if self.kind == "crossings":
            return degrade_crossings(
                data.crossings,
                self.p_keep[index],
                self.extra_rate_hz[index],
                data.bin_s,
                np.random.default_rng(self.seed),
            )
        if data.sbp is None:
            raise ValueError(f"{data.name} has no recorded SBP")
        n_effective = 2.0 * (NARROWBAND_HZ[1] - NARROWBAND_HZ[0]) * data.bin_s
        return age_sbp(
            data.sbp,
            self.amplitude[index],
            float(self.sbp_ratio[index]),
            sbp_floor(data.sbp, n_effective),
            gain=float(self.gain[index]),
            n_effective=n_effective,
            rng=np.random.default_rng(self.seed),
        )


def _u3_position(chain: ChainConfig) -> int:
    """Return a chain's index in U3's grid, which seeds U3's bridge draw for it."""
    return [c.label for c in u3.grid(u3.U3Config())].index(chain.label)


def chain_config(kind: str, config: U6Config) -> ChainConfig:
    """Return the power-model description of a chain."""
    if kind == "sbp":
        return ChainConfig("sbp", config.sbp_current_a, config.sbp_bits, SBP_FS_HZ)
    return REFERENCE


def make_ager(
    kind: str,
    n_channels: int,
    weeks: tuple[float, ...],
    config: U6Config,
    response: Response,
    background: tuple[FloatArray, FloatArray],
) -> Ager:
    """Give every channel of a dataset an aging trajectory.

    Each channel gets its own amplitude decay rate from the aging model and
    the measured response of one MEArec channel, drawn at random, so channels
    differ in how many spikes they lose for the same loss of amplitude.
    """
    model = config.aging
    amplitude = model.amplitude_scale(np.asarray(weeks), n_channels, _rate_seed(config))
    donor = np.random.default_rng([config.seed, 12]).integers(
        0, response.real.shape[1], size=n_channels
    )
    grid = np.asarray(response.amplitude_grid)[::-1]
    p_keep = np.ones_like(amplitude)
    extra = np.zeros_like(amplitude)
    ratio, gain = np.ones(len(weeks)), np.ones(len(weeks))
    for w, week in enumerate(weeks):
        state = state_of(model, week)
        chain = chain_state(config, state, background)
        ratio[w], gain[w] = chain["sbp_noise_ratio"], chain[f"{kind}_gain"]
        keep_table, extra_table = response.tables(state, rethreshold=config.rethreshold)
        for channel in range(n_channels):
            p_keep[w, channel] = np.interp(
                amplitude[w, channel], grid, keep_table[donor[channel], ::-1]
            )
            extra[w, channel] = np.interp(
                amplitude[w, channel], grid, extra_table[donor[channel], ::-1]
            )
    seed = (u3.U3Config().seed, _u3_position(chain_config(kind, config)))
    return Ager(kind, tuple(weeks), amplitude, p_keep, extra, ratio, gain, seed)


# ------------------------------------------------------------------------------ decoding


@dataclass
class Fold:
    """One fold: its held-out bins, its decoder and the constants stored with it."""

    test: BoolArray
    decoder: Decoder
    mean: FloatArray
    scale: FloatArray
    mad: FloatArray


def fit_folds(
    name: str, data: u3.DecodingSet, features: FloatArray, cfg: u4.U4Config
) -> list[Fold]:
    """Fit one decoder per fold on the other folds, as U4 does, and keep it.

    Each fold stores the training bins' mean, standard deviation and mean
    absolute deviation per channel: the constants the stabiliser starts from.
    """
    make = u4.decoder_factory(name, data, cfg)
    folds = []
    for block in np.array_split(np.unique(data.groups), cfg.n_folds):
        test = np.isin(data.groups, block)
        train = features[~test]
        mean, scale = train.mean(axis=0), train.std(axis=0)
        decoder = make()
        decoder.fit(
            normalise(train, mean, scale, scale, mode="fixed"),
            data.targets[~test],
            data.groups[~test],
        )
        folds.append(Fold(test, decoder, mean, scale, np.abs(train - mean).mean(axis=0)))
    return folds


def predict_folds(
    folds: list[Fold],
    data: u3.DecodingSet,
    features: FloatArray,
    config: U6Config,
    mode: Mode = "fixed",
    gate: IntArray | None = None,
) -> FloatArray:
    """Predict every bin with the decoder that did not see it.

    With a running mode the stabiliser runs causally through the whole session
    in recording order, starting from the fold's stored constants (or, with
    ``config.warm_start``, from the session's own statistics), and the fold's
    held-out bins are read out of that.
    """
    out = np.empty_like(data.targets, dtype=np.float64)
    start = None
    if config.warm_start and mode != "fixed":
        centre = features.mean(axis=0)
        start = (centre, np.abs(features - centre).mean(axis=0))
    for fold in folds:
        scaled = normalise(
            features,
            fold.mean,
            fold.scale,
            fold.mad,
            mode=mode,
            shift=config.shift,
            max_gain=config.max_gain,
            gate=gate,
            start=start,
        )
        out[fold.test] = fold.decoder.predict(scaled[fold.test], data.groups[fold.test])
    return out


def _scored(
    targets: FloatArray,
    groups: IntArray,
    predictions: FloatArray,
    cfg: u4.U4Config,
    **against: FloatArray,
) -> dict[str, float]:
    """Return R² with its interval, and a paired difference per named reference."""
    r2, low, high = bootstrap_r2(targets, predictions, groups, n_boot=cfg.n_boot, seed=cfg.seed)
    out = {"r2": r2, "r2_low": low, "r2_high": high}
    for name, reference in against.items():
        difference, d_low, d_high = bootstrap_r2_difference(
            targets, predictions, reference, groups, n_boot=cfg.n_boot, seed=cfg.seed
        )
        out |= {name: difference, f"{name}_low": d_low, f"{name}_high": d_high}
    return out


DecodeJob = tuple[str, u3.DecodingSet, str, U6Config, dict[str, tuple[U6Config, Ager]]]


def _decode_job(args: DecodeJob) -> list[dict[str, Any]]:
    """Fit one decoder at week 0 and score it on every aged version of one dataset.

    Returns one row per variation, week and mode. ``loss`` is the paired
    difference from the same decoder at week 0; ``gain`` is the paired
    difference from the unstabilised decoder at the same week.
    """
    condition, data, name, config, agers = args
    torch.set_num_threads(1)  # several of these run side by side
    cfg = config.decoding
    _, base = agers["baseline"]
    folds = fit_folds(name, data, base.features(data, 0), cfg)
    week0 = predict_folds(folds, data, base.features(data, 0), config)
    rows: list[dict[str, Any]] = []
    for variation, (varied, ager) in agers.items():
        for index, week in enumerate(ager.weeks):
            features = ager.features(data, index)
            fixed = predict_folds(folds, data, features, varied)
            predictions = {"fixed": fixed} | {
                mode: predict_folds(folds, data, features, varied, mode)
                for mode in MODES
                if mode != "fixed"
            }
            if variation == "baseline" and config.refit:
                refitted = fit_folds(name, data, features, cfg) if week > 0 else folds
                predictions["refit"] = predict_folds(refitted, data, features, varied)
            for mode, predicted in predictions.items():
                rows.append(
                    {
                        "variation": variation,
                        "condition": condition,
                        "decoder": name,
                        "week": week,
                        "mode": mode,
                        **_scored(
                            data.targets, data.groups, predicted, cfg, loss=week0, gain=fixed
                        ),
                    }
                )
    return rows


def conditions(sets: dict[str, u3.DecodingSet]) -> dict[str, tuple[u3.DecodingSet, str]]:
    """Return every dataset on every chain it can be scored for.

    A dataset without recorded SBP (NoMAD) can only carry the crossings chain.
    """
    out = {}
    for name, data in sets.items():
        for kind in CHAINS:
            if kind == "sbp" and data.sbp is None:
                continue
            out[f"{name}_{kind}"] = (data, kind)
    return out


def decoding(
    sets: dict[str, u3.DecodingSet],
    config: U6Config,
    response: Response,
    background: tuple[FloatArray, FloatArray],
    detection: u2.U2Config | None = None,
    varied: dict[str, U6Config] | None = None,
) -> dict[str, Any]:
    """Score every decoder on every dataset and chain against simulated weeks."""
    detection = detection or u2.U2Config()
    varied = varied or {}
    short = tuple(dict.fromkeys((config.key_week, max(config.weeks))))
    jobs: list[DecodeJob] = []
    described = {}
    for condition, (data, kind) in conditions(sets).items():
        n_channels = data.crossings.shape[1]
        agers = {
            "baseline": (
                config,
                make_ager(kind, n_channels, config.weeks, config, response, background),
            )
        }
        for name, other in varied.items():
            agers[name] = (
                other,
                make_ager(kind, n_channels, short, other, response, background),
            )
        base = agers["baseline"][1]
        at = config.weeks.index(config.key_week)
        described[condition] = {
            "dataset": data.name,
            "chain": kind,
            "n_bins": int(data.groups.size),
            "n_channels": int(n_channels),
            "n_groups": int(np.unique(data.groups).size),
            "bin_s": data.bin_s,
            "chain_uw": chain_power(chain_config(kind, config), config.power).total_w * 1e6,
            "key_week_amplitude_median": float(np.median(base.amplitude[at])),
            "key_week_p_keep_median": float(np.median(base.p_keep[at])),
            "key_week_p_keep_p10": float(np.percentile(base.p_keep[at], 10)),
            "key_week_sbp_noise_ratio": float(base.sbp_ratio[at]),
        }
        jobs.extend((condition, data, name, config, agers) for name in config.decoding.decoders)
    results: list[list[dict[str, Any]]] = u2._map(_decode_job, jobs, detection)
    return {"conditions": described, "rows": [row for rows in results for row in rows]}


# ------------------------------------------------------------------------- the real pair


def session_shift(early: u3.DecodingSet, late: u3.DecodingSet) -> dict[str, float]:
    """Describe how per-channel firing statistics moved between two sessions."""
    rate_early = early.crossings.mean(axis=0) / early.bin_s
    rate_late = late.crossings.mean(axis=0) / late.bin_s
    active = (rate_early > 0.5) & (rate_late > 0)
    ratio = rate_late[active] / rate_early[active]
    return {
        "n_channels": int(rate_early.size),
        "n_compared": int(active.sum()),
        "rate_ratio_median": float(np.median(ratio)),
        "rate_ratio_p10": float(np.percentile(ratio, 10)),
        "rate_ratio_p90": float(np.percentile(ratio, 90)),
        "share_rate_more_than_doubled": float(np.mean(ratio > 2.0)),
        "share_rate_more_than_halved": float(np.mean(ratio < 0.5)),
        "share_rate_up": float(np.mean(ratio > 1.0)),
        "log_rate_correlation": float(
            np.corrcoef(np.log(rate_early[active]), np.log(rate_late[active]))[0, 1]
        ),
    }


CROSS_MODES = ("fixed", "mean", "zscore", "batch", "latent", "aligned", "within")


def cross_session(early: u3.DecodingSet, late: u3.DecodingSet, config: U6Config) -> dict[str, Any]:
    """Fit on the earlier NoMAD session and test on the later one.

    Decoders reading the channels:

    * ``fixed``: U4's test, with the earlier session's constants.
    * ``mean`` and ``zscore``: the running stabiliser, started cold from those
      constants.
    * ``batch``: the later session's own mean and spread, known only after the
      session, so not causal: the most any per-channel rescaling could do.
    * ``within``: a decoder fitted on the later session itself (held-out
      folds): what supervised recalibration gives.

    Decoders reading ``config.n_latents`` latent variables (a second decoder,
    fitted on the earlier session's principal components), both behind the
    running z-score:

    * ``latent``: the later session projected on the earlier session's axes.
    * ``aligned``: the later session's own axes, estimated from its first
      ``config.align_calibration_s`` seconds without labels and rotated onto
      the earlier session's (:mod:`neurochain.align`).

    ``gain`` is the paired difference from ``fixed``.
    """
    cfg = config.decoding
    train = early.crossings
    mean, scale = train.mean(axis=0), train.std(axis=0)
    mad = np.abs(train - mean).mean(axis=0)
    own_mean, own_scale = late.crossings.mean(axis=0), late.crossings.std(axis=0)
    train_z = normalise(train, mean, scale, mad, mode="fixed")
    late_z: dict[str, FloatArray] = {
        mode: normalise(
            late.crossings,
            mean,
            scale,
            mad,
            mode=mode,
            shift=config.shift,
            max_gain=config.max_gain,
        )
        for mode in MODES
    }
    late_z["batch"] = normalise(late.crossings, own_mean, own_scale, mad, mode="fixed")
    axes = principal_axes(train_z, config.n_latents)
    latent_scale = (train_z @ axes).std(axis=0)
    n_calibration = min(late.groups.size, round(config.align_calibration_s / late.bin_s))
    late_axes = aligned_axes(late_z["zscore"][:n_calibration], axes)
    rows = []
    for name in cfg.decoders:
        make = u4.decoder_factory(name, early, cfg)
        decoder = make()
        decoder.fit(train_z, early.targets, early.groups)
        predictions: dict[str, FloatArray] = {
            mode: decoder.predict(z, late.groups) for mode, z in late_z.items()
        }
        latent_decoder = make()
        latent_decoder.fit(train_z @ axes / latent_scale, early.targets, early.groups)
        for mode, use in (("latent", axes), ("aligned", late_axes)):
            predictions[mode] = latent_decoder.predict(
                late_z["zscore"] @ use / latent_scale, late.groups
            )
        predictions["within"] = held_out_predictions(
            u4.decoder_factory(name, late, cfg),
            late.crossings,
            late.targets,
            late.groups,
            n_folds=cfg.n_folds,
        )
        for mode in CROSS_MODES:
            rows.append(
                {
                    "decoder": name,
                    "mode": mode,
                    **_scored(
                        late.targets,
                        late.groups,
                        predictions[mode],
                        cfg,
                        gain=predictions["fixed"],
                    ),
                }
            )
    return {
        "train": early.name,
        "test": late.name,
        "n_segments": int(np.unique(late.groups).size),
        "n_latents": config.n_latents,
        "calibration_s": n_calibration * late.bin_s,
        "calibration_share": n_calibration / late.groups.size,
        "shift": session_shift(early, late),
        "rows": rows,
    }


# --------------------------------------------------------------------------- stimulation


def stimulation_shift(
    data: u3.DecodingSet,
    measured: dict[str, Any],
    config: U6Config,
    response: Response,
    background: tuple[FloatArray, FloatArray],
    stim: u5.U5Config | None = None,
    treatment: str = "template",
) -> dict[str, Any]:
    """Test whether the stabiliser absorbs U5's stimulation-induced feature shift.

    U5's protocol on the 0.48 µW chain: the stimulating electrode is on the
    recording array, the template canceller runs, and the decoder was fitted
    before stimulation was switched on. Scores are R² on stimulated bins.

    Modes: ``fixed`` (U5's result), ``zscore`` (one set of running registers),
    ``gated mean`` and ``gated zscore`` (two sets, selected by the stimulator's
    own on/off flag) and ``refit`` (the decoder refitted with stimulation on,
    U5's remedy). ``loss`` is the paired difference from the same decoder on
    the same bins without stimulation, ``gain`` from ``fixed``.

    Args:
        data: The decoding dataset (Nason).
        measured: ``signal`` from ``results/u5_metrics.json``.
        config: U6 parameters.
        response: The MEArec response (only its channel count is used here).
        background: Neural background PSD.
        stim: U5 parameters; the stimulation schedule and geometry come from it.
        treatment: U5's artifact treatment.
    """
    stim = stim or u5.U5Config()
    cfg = replace(config.decoding, decoders=stim.decoding.decoders)
    n_bins, n_channels = data.crossings.shape
    features = make_ager("sbp", n_channels, (0.0,), config, response, background).features(data, 0)
    on = u5._schedule(n_bins * data.bin_s, stim, 0.0, 4).bins_on(n_bins, data.bin_s)
    amplitude, stim_channel = u5.scenario_amplitudes_uv(0.0, stim)
    changed = u5.stimulated_features(
        "sbp",
        features,
        on,
        amplitude[:n_channels],
        stim_channel,
        treatment,
        measured,
        np.random.default_rng([stim.seed, 0, 0, u5.TREATMENTS.index(treatment)]),
    )
    gate = on.astype(np.int64)
    truth, groups = data.targets[on], data.groups[on]
    rows = []
    for name in cfg.decoders:
        folds = fit_folds(name, data, features, cfg)
        clean = predict_folds(folds, data, features, config)
        predictions = {
            "fixed": predict_folds(folds, data, changed, config),
            "zscore": predict_folds(folds, data, changed, config, "zscore"),
            "gated mean": predict_folds(folds, data, changed, config, "mean", gate),
            "gated zscore": predict_folds(folds, data, changed, config, "zscore", gate),
            "refit": predict_folds(fit_folds(name, data, changed, cfg), data, changed, config),
        }
        for mode, predicted in predictions.items():
            rows.append(
                {
                    "decoder": name,
                    "mode": mode,
                    **_scored(
                        truth,
                        groups,
                        predicted[on],
                        cfg,
                        loss=clean[on],
                        gain=predictions["fixed"][on],
                    ),
                }
            )
        r2, low, high = bootstrap_r2(truth, clean[on], groups, n_boot=cfg.n_boot, seed=cfg.seed)
        rows.append(
            {"decoder": name, "mode": "no stimulation", "r2": r2, "r2_low": low, "r2_high": high}
        )
    return {
        "dataset": data.name,
        "chain": "sbp",
        "scenario": u5.scenario_name(0.0),
        "treatment": treatment,
        "stimulated_bins": int(on.sum()),
        "n_bins": int(n_bins),
        "rows": rows,
    }


# ------------------------------------------------------------------- a real recording


def in_vivo(broadband_uv: FloatArray, fs_hz: float, config: U6Config) -> dict[str, Any]:
    """Pass a real recording through the fresh and the settled electrode.

    A real recording cannot have its spikes shrunk separately from its
    background, so this isolates the electrode's part of the model: what the
    impedance rise alone does to the noise level, the crossing rate and SBP.
    """
    settled = state_of(config.aging, config.key_week)
    adc = Adc(fs_hz=REFERENCE.fs_hz, bits=REFERENCE.bits)
    sbp_adc = Adc(fs_hz=SBP_FS_HZ, bits=config.sbp_bits)
    gain0 = {name: reference_gain(fe) for name, fe in frontends(config, FRESH).items()}
    out: dict[str, Any] = {"duration_s": float(broadband_uv.size / fs_hz), "states": {}}
    for label, state in (("fresh", FRESH), ("settled", settled)):
        both = frontends(config, state)
        wide = emulate(broadband_uv, fs_hz, both["crossings"], seed=config.seed + 77, alias_zones=8)
        codes = convert(wide.output_v / gain0["crossings"] * 1e6, fs_hz, adc)
        filtered = det.causal_highpass(codes.codes.astype(np.float64), adc.fs_hz)
        score = det.score_signal(filtered, adc.fs_hz, "threshold")
        events = det.detect_events(
            score, det.calibrate_threshold(score, "threshold", 4.5), adc.fs_hz
        )
        narrow = emulate(broadband_uv, fs_hz, both["sbp"], seed=config.seed + 50, alias_zones=8)
        sampled = convert(narrow.output_v / gain0["sbp"] * 1e6, fs_hz, sbp_adc).microvolts
        out["states"][label] = {
            "impedance_factor": state[0],
            "spike_band_sigma_uv": mad_sigma(filtered) * adc.lsb_uv,
            "crossing_rate_hz": float(events.size / out["duration_s"]),
            "sbp_uv": float(np.abs(sampled).mean()),
        }
    return out


# ---------------------------------------------------------------------------------- cost


def stabiliser_cost(config: U6Config, register_bits: int = 12) -> dict[str, Any]:
    """Return the stabiliser's cost per channel in the terms of U3's power model.

    Every add, subtract or division step is charged as one multiply of the
    power model, at the feature rate (one bin per 50 ms). The registers hold
    ``register_bits`` plus ``shift`` fractional bits each. The latent
    projection of the alignment costs ``n_latents`` real multiplies per
    channel per bin; finding the axes (a singular value decomposition, once
    per session) is assumed to happen outside the implant and is not charged.
    """
    energy = config.power.energy_per_multiply_j
    out: dict[str, Any] = {"bin_s": BIN_S, "shift": config.shift, "modes": {}, "chains": {}}
    for mode in ("mean", "zscore"):
        ops = operations_per_bin(mode, register_bits)
        out["modes"][mode] = {
            "operations_per_bin": ops,
            "uw_per_channel": ops / BIN_S * energy * 1e6,
            "register_bits_per_channel": (1 if mode == "mean" else 2)
            * (register_bits + config.shift),
        }
    out["modes"]["aligned"] = {
        "operations_per_bin": config.n_latents,
        "uw_per_channel": config.n_latents / BIN_S * energy * 1e6,
    }
    for kind in CHAINS:
        chain_uw = chain_power(chain_config(kind, config), config.power).total_w * 1e6
        out["chains"][kind] = {
            "chain_uw": chain_uw,
            "zscore_share_of_chain": out["modes"]["zscore"]["uw_per_channel"] / chain_uw,
        }
    return out


# ------------------------------------------------------------------------------- compute


def _row(rows: list[dict[str, Any]], **match: Any) -> dict[str, Any]:
    found = [r for r in rows if all(r[k] == v for k, v in match.items())]
    if len(found) != 1:
        raise KeyError(f"{len(found)} rows match {match}")
    return found[0]


def _key(summary: dict[str, Any], config: U6Config) -> dict[str, Any]:
    """Collect the unit's headline numbers from the full summary."""
    rows = summary["decoding"]["rows"]
    key: dict[str, Any] = {
        "week": config.key_week,
        "last_week": max(config.weeks),
        "stabiliser_uw": summary["cost"]["modes"]["zscore"]["uw_per_channel"],
        "curves": {},
    }
    modes = [*MODES, "refit"] if config.refit else list(MODES)
    for condition in summary["decoding"]["conditions"]:
        entry: dict[str, Any] = {}
        for name in config.decoding.decoders:
            per_decoder: dict[str, Any] = {
                f"week0{suffix}": _row(
                    rows,
                    variation="baseline",
                    condition=condition,
                    decoder=name,
                    week=0.0,
                    mode=mode,
                )["r2"]
                for suffix, mode in (("", "fixed"), ("_zscore", "zscore"))
            }
            for week, label in ((config.key_week, "key"), (max(config.weeks), "last")):
                for mode in modes:
                    row = _row(
                        rows,
                        variation="baseline",
                        condition=condition,
                        decoder=name,
                        week=week,
                        mode=mode,
                    )
                    per_decoder[f"{label}_{mode}"] = {
                        k: row[k]
                        for k in (
                            "r2",
                            "loss",
                            "loss_low",
                            "loss_high",
                            "gain",
                            "gain_low",
                            "gain_high",
                        )
                    }
            entry[name] = per_decoder
        key["curves"][condition] = entry
    if "cross_session" in summary:
        cross = summary["cross_session"]
        key["cross_session"] = {
            "weeks_apart": cross["weeks_apart"],
            **{
                name: {
                    mode: _row(cross["rows"], decoder=name, mode=mode)["r2"] for mode in CROSS_MODES
                }
                for name in config.decoding.decoders
            },
        }
    return key


def compute_u6(
    recording: MearecRecording,
    *,
    config: U6Config | None = None,
    detection: u2.U2Config | None = None,
    sets: dict[str, u3.DecodingSet] | None = None,
    session_pair: tuple[u3.DecodingSet, u3.DecodingSet] | None = None,
    days_apart: float | None = None,
    stim_data: u3.DecodingSet | None = None,
    u5_signal: dict[str, Any] | None = None,
    broadband: tuple[FloatArray, float] | None = None,
    with_variations: bool = True,
) -> dict[str, Any]:
    """Run the whole U6 analysis and return a JSON-ready summary.

    Args:
        recording: The MEArec recording.
        config: Analysis parameters.
        detection: Detection settings and the worker count.
        sets: Datasets to age, by name. Skipped if ``None``.
        session_pair: ``(earlier, later)`` NoMAD sessions for the real test.
        days_apart: Days between them.
        stim_data: Dataset for the stimulation test (Nason).
        u5_signal: ``signal`` from ``results/u5_metrics.json``, for that test.
        broadband: ``(samples in µV, fs)`` of a real recording.
        with_variations: Also score the changed assumptions of
            :func:`variations`.
    """
    config = config or U6Config()
    detection = detection or u2.U2Config()
    background = u3.background_psd(recording, config.noise_uv)
    varied = variations(config) if with_variations else {}
    short = (config.key_week, max(config.weeks))
    states = {state_of(config.aging, week) for week in config.weeks}
    for other in varied.values():
        states |= {state_of(other.aging, week) for week in short}
    response = aging_response(recording, config, tuple(states), background, detection)
    settled = response.states.index(state_of(config.aging, config.key_week))
    summary: dict[str, Any] = {
        "config": asdict(config),
        "model": model_summary(config, background),
        "response": response.summary(rethreshold=config.rethreshold) | {"settled": settled},
        "response_fixed_threshold": response.summary(rethreshold=False) | {"settled": settled},
        "bridge_check": bridge_check(response, config),
        "cost": stabiliser_cost(config),
        "variations": list(varied),
    }
    if broadband is not None:
        summary["in_vivo"] = in_vivo(*broadband, config)
    if sets is not None:
        summary["decoding"] = decoding(sets, config, response, background, detection, varied)
    if session_pair is not None:
        summary["cross_session"] = cross_session(*session_pair, config) | {
            "days_apart": days_apart,
            "weeks_apart": None if days_apart is None else days_apart / 7.0,
        }
    if stim_data is not None and u5_signal is not None:
        summary["stimulation"] = stimulation_shift(
            stim_data, u5_signal, config, response, background
        )
    if sets is not None:
        summary["key"] = _key(summary, config)
    return summary
