"""U5 analysis: what stimulation does to the sensing chain, and what it costs to undo.

Five steps, each feeding the next:

1. **Propagation.** How large the artifact is at a recording electrode, from
   the point-source model (:mod:`neurochain.stim.propagation`).
2. **Recovery.** How long each front end takes to come back after a pulse, in
   ngspice and in the Python large-signal model, which have to agree
   (:mod:`neurochain.frontend.saturation`).
3. **Signal level.** On the MEArec recording, where the spikes are known: add
   the artifact to every channel, run both chains with and without
   cancellation, and measure what happens to the features.
4. **A real recording.** The same on the Nason in-vivo channel, as a check
   that nothing above depends on the recording being simulated.
5. **Decoding.** Carry the measured change in the features onto the Nason
   decoding dataset (the bridge idea of U3) and score the decoders of U4.

The bridge in step 5 is needed for the same reason as in U3: no public dataset
here has both broadband voltages and behaviour. Step 3 measures, per artifact
amplitude and per treatment, how far a stimulated bin's feature sits from the
unstimulated one (a mean shift and a spread, in units of the channel's noise
floor; for crossings, the share of spikes kept and the false positives added).
Step 5 applies exactly that to the recorded features in the bins where a
stimulation schedule is on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import signal

from neurochain.adc import Adc, quantise, sample
from neurochain.analysis import u2, u3, u4
from neurochain.bridge import sbp_floor
from neurochain.data.mearec import MearecRecording
from neurochain.decode.evaluate import Decoder, bootstrap_r2, bootstrap_r2_difference, r2_score
from neurochain.features import detect as det
from neurochain.features.score import score_detections
from neurochain.fixedpoint import design_fixed_highpass, fixed_highpass
from neurochain.frontend import spice
from neurochain.frontend.emulator import emulate, reference_gain
from neurochain.frontend.model import FrontEnd
from neurochain.frontend.saturation import (
    SaturatingAmplifier,
    pulse_kernel,
    pulse_response_uv,
    recovery_time_s,
)
from neurochain.power import NARROWBAND_HZ, SBP_FS_HZ, ChainConfig, PowerModel, chain_power
from neurochain.recording import FloatArray, IntArray
from neurochain.stim import cancel
from neurochain.stim.propagation import (
    UTAH_PITCH_UM,
    artifact_amplitude_uv,
    monopole_uv,
    potential_uv,
    utah_positions_um,
)
from neurochain.stim.pulse import BoolArray, Pulse, Schedule, pulse_train, random_schedule

BIN_S = u3.BIN_S
TREATMENTS = ("none", "blank", "template")
CHAINS = ("sbp", "crossings")


@dataclass(frozen=True)
class U5Config:
    """Parameters of the U5 analysis.

    Attributes:
        seed: Seed for the schedules, the noise and the bootstrap.
        pulse: The stimulation pulse.
        rate_hz: Pulse rate inside a train.
        jitter_s: Random offset of each pulse from the ADC clock; 0 means the
            stimulator runs off that clock.
        amplitudes_mv: Artifact amplitudes at the electrode at which the
            feature changes are measured.
        noise_uv: Spike-band RMS of the neural background on MEArec.
        recovery_threshold_uv: The output counts as recovered once it stays
            inside this, referred to the input. 13 µV is the in-vivo noise.
        adapt_shift: Adaptation shift of the template canceller.
        template_frac_bits: Fractional bits of the stored template.
        n_taps: Samples after each pulse the SBP chain's template covers, at
            most one pulse period. The crossings chain's covers the period.
        fine_hz: Time resolution of jittered pulse onsets.
        sigma_s_per_m: Tissue conductivity.
        scenarios_mm: Distance from the stimulating electrode to the centre of
            the recording array; 0 puts it on the array itself.
        train_s: Shortest and longest pulse train.
        pause_s: Shortest and longest pause between trains.
        example_mv: Amplitude of the pulse drawn in the figures.
        sbp_current_a: Supply current of the SBP chain's amplifier (U3's choice).
        sbp_bits: ADC resolution of the SBP chain.
        decoding: Decoder and bootstrap settings, shared with U4.
        power: Constants of the power model.
    """

    seed: int = 0
    pulse: Pulse = Pulse()  # noqa: RUF009 - frozen dataclass, safe as a default
    rate_hz: float = 100.0
    jitter_s: float = 0.0
    amplitudes_mv: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    noise_uv: float = 13.0
    recovery_threshold_uv: float = 13.0
    adapt_shift: int = 4
    template_frac_bits: int = 6
    n_taps: int = 16
    fine_hz: float = 240_000.0
    sigma_s_per_m: float = 0.3
    scenarios_mm: tuple[float, ...] = (0.0, 5.0, 20.0)
    train_s: tuple[float, float] = (0.5, 1.5)
    pause_s: tuple[float, float] = (0.5, 1.5)
    example_mv: float = 30.0
    sbp_current_a: float = 0.25e-6
    sbp_bits: int = 8
    decoding: u4.U4Config = u4.U4Config(decoders=("wiener", "gru"))  # noqa: RUF009
    power: PowerModel = PowerModel()  # noqa: RUF009


@dataclass(frozen=True)
class Chain:
    """One sensing chain: a front end and the converter behind it."""

    name: str
    frontend: FrontEnd
    adc: Adc

    @property
    def gain(self) -> float:
        """Gain that refers the amplifier output to the electrode."""
        return reference_gain(self.frontend)

    @property
    def limit_uv(self) -> float:
        """Input-referred level at which the amplifier output reaches a rail."""
        return self.frontend.lna.output_swing_v / self.gain * 1.0e6


def chains(config: U5Config) -> dict[str, Chain]:
    """Return U3's winning SBP chain and the conventional crossings chain."""
    ref = u3.REFERENCE
    return {
        "sbp": Chain(
            "sbp",
            u3.narrowband_frontend(config.sbp_current_a, config.power),
            Adc(fs_hz=SBP_FS_HZ, bits=config.sbp_bits),
        ),
        "crossings": Chain(
            "crossings",
            u3.wideband_frontend(ref.i_total_a, config.power),
            Adc(fs_hz=ref.fs_hz, bits=ref.bits),
        ),
    }


def scenario_name(distance_mm: float) -> str:
    """Return the label of a stimulation geometry."""
    return "same array" if distance_mm == 0 else f"{distance_mm:g} mm away"


# --------------------------------------------------------------------------- propagation


def scenario_amplitudes_uv(distance_mm: float, config: U5Config) -> tuple[FloatArray, int | None]:
    """Return the artifact amplitude on each of 96 channels, and the stimulating channel.

    The recording array is a 10 by 10 Utah array without its corners;
    stimulation is monopolar and the recording reference is distant. At 0 mm
    the stimulating electrode is the array site nearest the centre, and its
    index is returned; otherwise it sits ``distance_mm`` from the centre, in
    the plane of the tips.
    """
    sites = utah_positions_um()
    stim_channel: int | None = None
    if distance_mm == 0:
        stim_channel = int(np.argmin(np.linalg.norm(sites, axis=1)))
        stim = sites[stim_channel]
    else:
        stim = np.array([distance_mm * 1000.0, 0.0, 0.0])
    amplitude = artifact_amplitude_uv(
        sites, stim, config.pulse.amplitude_ua, sigma_s_per_m=config.sigma_s_per_m
    )
    return np.abs(amplitude), stim_channel


def propagation(config: U5Config) -> dict[str, Any]:
    """Summarise artifact amplitude against distance and for each geometry."""
    current, sigma = config.pulse.amplitude_ua, config.sigma_s_per_m
    distance_um = np.geomspace(200.0, 30_000.0, 60)
    points = np.column_stack([distance_um, np.zeros_like(distance_um), np.zeros_like(distance_um)])
    pair = np.array([[0.0, 0.0, 0.0], [-UTAH_PITCH_UM, 0.0, 0.0]])
    bipolar = potential_uv(points, pair, np.array([current, -current]), sigma)
    all_chains = chains(config)
    scenarios = []
    for distance_mm in config.scenarios_mm:
        amplitude, stim_channel = scenario_amplitudes_uv(distance_mm, config)
        others = np.delete(amplitude, stim_channel) if stim_channel is not None else amplitude
        scenarios.append(
            {
                "name": scenario_name(distance_mm),
                "distance_mm": distance_mm,
                "stim_channel": stim_channel,
                "median_mv": float(np.median(others)) / 1000.0,
                "min_mv": float(others.min()) / 1000.0,
                "max_mv": float(others.max()) / 1000.0,
                "share_above_adc": float(np.mean(others > all_chains["sbp"].adc.input_range_uv)),
                "share_above_amplifier": float(
                    np.mean(
                        others
                        > SaturatingAmplifier.from_frontend(
                            all_chains["sbp"].frontend
                        ).linear_input_v
                        * 1e6
                    )
                ),
                "amplitudes_mv": np.round(amplitude / 1000.0, 4).tolist(),
            }
        )
    return {
        "current_ua": current,
        "sigma_s_per_m": sigma,
        "charge_per_phase_nc": config.pulse.charge_per_phase_nc,
        "pair_separation_um": UTAH_PITCH_UM,
        "distance_mm": np.round(distance_um / 1000.0, 4).tolist(),
        "monopolar_mv": np.round(monopole_uv(current, distance_um, sigma) / 1000.0, 5).tolist(),
        "bipolar_mv": np.round(np.abs(bipolar) / 1000.0, 5).tolist(),
        "adc_range_mv": all_chains["sbp"].adc.input_range_uv / 1000.0,
        "amplifier_range_mv": SaturatingAmplifier.from_frontend(
            all_chains["sbp"].frontend
        ).linear_input_v
        * 1000.0,
        "scenarios": scenarios,
    }


# ------------------------------------------------------------------------------ recovery

_RECOVERY_GRID_S = np.unique(
    np.concatenate([np.arange(0.0, 0.03, 2.0e-6), np.arange(0.03, 1.2, 1.0e-4)])
)
_SPICE_STOP_S = 0.03
_SPICE_STEP_S = 5.0e-6


def _spice_response(
    frontend: FrontEnd, config: U5Config, amplitude_uv: float, gain: float
) -> tuple[FloatArray, FloatArray]:
    t_bp, wave = config.pulse.breakpoints()
    t, v_out, _ = spice.simulate_transient(
        frontend,
        t_bp,
        wave * amplitude_uv * 1.0e-6,
        t_stop_s=_SPICE_STOP_S,
        max_step_s=_SPICE_STEP_S,
    )
    return t, np.asarray(-v_out / gain * 1.0e6, dtype=np.float64)


def recovery(config: U5Config, *, with_spice: bool = True) -> dict[str, Any]:
    """Measure each chain's recovery after one pulse, in the model and in ngspice.

    Recovery time runs from the end of the second phase to the last moment
    the output, referred to the input, is outside the threshold. ``in_range``
    is the same for the ADC's input range: until then the samples are clipped.

    Args:
        config: Analysis parameters.
        with_spice: Run ngspice too. Without it the SPICE columns are ``None``.
    """
    pulse = config.pulse
    t_bp, wave = pulse.breakpoints()
    threshold = config.recovery_threshold_uv
    example_t = np.arange(0.0, 8.0e-3, 2.0e-5)
    out: dict[str, Any] = {
        "threshold_uv": threshold,
        "pulse_ms": pulse.duration_s * 1e3,
        "amplitudes_mv": list(config.amplitudes_mv),
        "spice": with_spice,
        "chains": {},
    }
    for name, chain in chains(config).items():
        amplifier = SaturatingAmplifier.from_frontend(chain.frontend)
        ideal = chain.frontend.without_electrode()
        rows: dict[str, list[Any]] = {
            "model_ms": [],
            "in_range_ms": [],
            "peak_mv": [],
            "spice_ms": [],
            "spice_electrode_ms": [],
            "waveform_difference_uv": [],
        }
        for amplitude_mv in config.amplitudes_mv:
            amplitude_uv = amplitude_mv * 1000.0
            y = pulse_response_uv(
                amplifier, t_bp, wave, amplitude_uv, _RECOVERY_GRID_S, gain=chain.gain
            )
            rows["model_ms"].append(
                recovery_time_s(_RECOVERY_GRID_S, y, pulse.duration_s, threshold) * 1e3
            )
            rows["in_range_ms"].append(
                recovery_time_s(_RECOVERY_GRID_S, y, pulse.duration_s, chain.adc.input_range_uv)
                * 1e3
            )
            rows["peak_mv"].append(float(np.abs(y).max()) / 1000.0)
            if not with_spice:
                for key in ("spice_ms", "spice_electrode_ms", "waveform_difference_uv"):
                    rows[key].append(None)
                continue
            t, y_spice = _spice_response(ideal, config, amplitude_uv, chain.gain)
            rows["spice_ms"].append(recovery_time_s(t, y_spice, pulse.duration_s, threshold) * 1e3)
            # Compare the waveforms after the pulse, where edges no longer dominate.
            after = np.arange(pulse.duration_s + 1.0e-4, _SPICE_STOP_S, 2.0e-5)
            difference = np.interp(after, t, y_spice) - np.interp(after, _RECOVERY_GRID_S, y)
            rows["waveform_difference_uv"].append(float(np.abs(difference).max()))
            t, y_electrode = _spice_response(chain.frontend, config, amplitude_uv, chain.gain)
            rows["spice_electrode_ms"].append(
                recovery_time_s(t, y_electrode, pulse.duration_s, threshold) * 1e3
            )
        example_uv = config.example_mv * 1000.0
        example: dict[str, Any] = {
            "amplitude_mv": config.example_mv,
            "time_ms": np.round(example_t * 1e3, 4).tolist(),
            "model_uv": np.round(
                pulse_response_uv(amplifier, t_bp, wave, example_uv, example_t, gain=chain.gain),
                3,
            ).tolist(),
            "spice_uv": None,
        }
        if with_spice:
            t, y_spice = _spice_response(ideal, config, example_uv, chain.gain)
            example["spice_uv"] = np.round(np.interp(example_t, t, y_spice), 3).tolist()
        if with_spice:
            model = np.asarray(rows["model_ms"])
            reference = np.asarray(rows["spice_ms"], dtype=np.float64)
            agreement: float | None = float(np.max(np.abs(model - reference)) * 1e3)
        else:
            agreement = None
        poles = amplifier.poles()
        out["chains"][name] = {
            "f_low_hz": chain.frontend.lna.f_low_hz,
            "tau_ms": 1e3 / (2.0 * np.pi * chain.frontend.lna.f_low_hz),
            "envelope_tau_ms": float(1e3 / np.min(np.abs(poles.real))),
            "ringing_hz": float(np.max(np.abs(poles.imag)) / (2.0 * np.pi)),
            "limit_mv": chain.limit_uv / 1000.0,
            "adc_range_mv": chain.adc.input_range_uv / 1000.0,
            **rows,
            "largest_model_spice_gap_us": agreement,
            "example": example,
        }
    return out


def reference_netlist(config: U5Config | None = None) -> str:
    """Return the netlist kept as ``spice/u5_saturation.cir``.

    The SBP chain's front end with its electrode, driven by the example pulse.
    """
    config = config or U5Config()
    t_bp, wave = config.pulse.breakpoints()
    return spice.transient_netlist(
        chains(config)["sbp"].frontend,
        t_bp,
        wave * config.example_mv * 1.0e-3,
        t_stop_s=_SPICE_STOP_S,
        max_step_s=_SPICE_STEP_S,
    )


# ---------------------------------------------------------------------- artifact kernels


@dataclass(frozen=True)
class Artifacts:
    """Everything needed to add the artifact to a chain and to remove it.

    Attributes:
        kernels: For each chain, one ``(n_phases, n_kernel)`` pulse response
            per amplitude, at the chain's ADC rate.
        blank_samples: For each chain and amplitude, the blanking window in ADC
            samples: just long enough for that amplitude's recovery. A window
            sized per channel is the most favourable case for blanking.
        n_taps: For each chain, the length of the template.
    """

    kernels: dict[str, list[FloatArray]]
    blank_samples: dict[str, list[int]]
    n_taps: dict[str, int]


def build_artifacts(config: U5Config, period_s: float) -> Artifacts:
    """Compute each chain's pulse responses and size its blanking window."""
    t_bp, wave = config.pulse.breakpoints()
    kernels: dict[str, list[FloatArray]] = {}
    blank_samples: dict[str, list[int]] = {}
    n_taps: dict[str, int] = {}
    for name, chain in chains(config).items():
        fs = chain.adc.fs_hz
        amplifier = SaturatingAmplifier.from_frontend(chain.frontend)
        n_phases = 1 if config.jitter_s == 0 else round(config.fine_hz / fs)
        kernels[name] = [
            pulse_kernel(amplifier, t_bp, wave, mv * 1000.0, fs, gain=chain.gain, n_phases=n_phases)
            for mv in config.amplitudes_mv
        ]
        period = round(period_s * fs)
        # With jitter the trigger is the first sample after the onset, one sample late.
        late = 0 if config.jitter_s == 0 else 1
        blank_samples[name] = []
        for kernel in kernels[name]:
            above = np.flatnonzero(np.abs(kernel).max(axis=0) > config.recovery_threshold_uv)
            last = int(above[-1]) if above.size else 0
            blank_samples[name].append(min(last + 1 + late, period))
        # The 1 Hz front end leaves an offset that outlasts the pulse period once it
        # has saturated, so on that chain the template has to cover the whole period.
        n_taps[name] = min(config.n_taps, period) if name == "sbp" else period
    return Artifacts(kernels, blank_samples, n_taps)


def _schedule(duration_s: float, config: U5Config, start_s: float, stream: int) -> Schedule:
    return random_schedule(
        duration_s,
        rate_hz=config.rate_hz,
        seed=[config.seed, stream],
        start_s=start_s,
        train_s=config.train_s,
        pause_s=config.pause_s,
        grid_s=BIN_S,
        clock_hz=SBP_FS_HZ,
        jitter_s=config.jitter_s,
    )


# -------------------------------------------------------------------------- signal level


def feature_highpass(codes: FloatArray, fs_hz: float) -> FloatArray:
    """Apply the SBP path's digital high-pass (the quantised filter of U2), in floats."""
    b, a = design_fixed_highpass(fs_hz, frac_bits=8, guard_bits=2).float_coefficients()
    return np.asarray(signal.lfilter(b, a, codes, axis=-1), dtype=np.float64)


def treat(
    codes: IntArray,
    onsets: IntArray,
    treatment: str,
    chain: Chain,
    config: U5Config,
    n_taps: int,
    blank_samples: list[int],
) -> tuple[FloatArray, BoolArray]:
    """Apply one artifact treatment to ADC codes.

    Args:
        codes: ``(n_rows, n_samples)`` codes, one row per amplitude.
        onsets: Sample index of each pulse onset.
        treatment: ``"none"``, ``"blank"`` or ``"template"``.
        chain: The chain the codes come from. The SBP chain's signal is
            band-limited and blanked samples become zero; the crossings chain's
            still carries its offset and they hold the last good value.
        config: Analysis parameters.
        n_taps: Length of the template.
        blank_samples: Blanking window of each row.

    Returns:
        ``(values, valid)``: the treated signal in LSB (the template canceller
        keeps a quarter of an LSB) and a mask that is false on blanked samples.
    """
    if treatment == "none":
        return codes.astype(np.float64), np.ones(codes.shape, dtype=bool)
    hold = chain.name != "sbp"
    if treatment == "blank":
        rows = [
            cancel.blank(row, onsets, n, fill="hold" if hold else "zero")
            for row, n in zip(codes, blank_samples, strict=True)
        ]
        return (
            np.stack([r[0] for r in rows]).astype(np.float64),
            np.stack([r[1] for r in rows]),
        )
    if treatment == "template":
        out_frac = min(2, config.template_frac_bits)
        scaled, valid = cancel.cancel_template(
            codes,
            onsets,
            n_taps=n_taps,
            shift=config.adapt_shift,
            code_min=chain.adc.code_min,
            code_max=chain.adc.code_max,
            frac_bits=config.template_frac_bits,
            out_frac_bits=out_frac,
        )
        values = scaled / float(1 << out_frac)
        return (cancel.hold_last_valid(values, valid) if hold else values), valid
    raise ValueError(f"unknown treatment {treatment!r}")


def sbp_under_stimulation(
    analog_uv: FloatArray,
    fs_hz: float,
    chain: Chain,
    schedule: Schedule,
    artifacts: Artifacts,
    config: U5Config,
) -> dict[str, Any]:
    """Run the SBP chain on one channel with the artifact added, under every treatment.

    Args:
        analog_uv: The amplifier's output for the neural signal alone, referred
            to the input. The artifact's response is added to it, which is
            exact wherever the sum is inside the ADC's range, because the
            amplifier is still linear there.
        fs_hz: Sample rate of ``analog_uv``.
        chain: The SBP chain.
        schedule: When the pulses happen.
        artifacts: Pulse responses and window lengths.
        config: Analysis parameters.

    Returns:
        ``clean`` SBP per bin, ``sbp[treatment]`` as ``(n_amplitudes, n_bins)``,
        ``valid[treatment]`` (share of samples kept in stimulated time),
        ``residual_uv[treatment]`` (RMS of what the treatment left behind on
        the kept samples), and the integer ``codes`` behind them.
    """
    adc = chain.adc
    base = sample(analog_uv, fs_hz, adc.fs_hz)
    artifact = np.stack(
        [pulse_train(k, schedule.onsets_s, adc.fs_hz, base.size) for k in artifacts.kernels["sbp"]]
    )
    clean_codes = quantise(base, adc)[0]
    codes = quantise(np.clip(base + artifact, -chain.limit_uv, chain.limit_uv), adc)[0]
    onsets = schedule.onset_samples(adc.fs_hz)
    per_bin = round(BIN_S * adc.fs_hz)
    on = schedule.samples_on(base.size, adc.fs_hz)
    clean, _ = cancel.masked_mean_abs(
        feature_highpass(clean_codes.astype(np.float64), adc.fs_hz),
        np.ones(clean_codes.shape, dtype=bool),
        per_bin,
    )
    out: dict[str, Any] = {
        "clean": clean,
        "clean_codes": clean_codes,
        "codes": codes,
        "sbp": {},
        "valid": {},
        "residual_uv": {},
        "treated": {},
        "mask": {},
    }
    for treatment in TREATMENTS:
        treated, valid = treat(
            codes,
            onsets,
            treatment,
            chain,
            config,
            artifacts.n_taps["sbp"],
            artifacts.blank_samples["sbp"],
        )
        out["sbp"][treatment], _ = cancel.masked_mean_abs(
            feature_highpass(treated, adc.fs_hz), valid, per_bin
        )
        keep = valid & on
        error = (treated - clean_codes) * adc.lsb_uv
        out["valid"][treatment] = keep.sum(axis=-1) / max(int(on.sum()), 1)
        out["residual_uv"][treatment] = np.sqrt(
            (error**2 * keep).sum(axis=-1) / np.maximum(keep.sum(axis=-1), 1)
        )
        out["treated"][treatment] = treated
        out["mask"][treatment] = valid
    return out


def _feature_change(
    result: dict[str, Any], bins: BoolArray, n_effective: float
) -> dict[str, dict[str, FloatArray]]:
    """Express each treatment's SBP in stimulated bins relative to the clean SBP.

    Returns ``bias`` and ``sd`` per amplitude: mean and standard deviation over
    stimulated bins of ``(sbp - clean) / floor``.
    """
    clean = result["clean"]
    floor = float(sbp_floor(clean[:, np.newaxis], n_effective)[0])
    out: dict[str, dict[str, FloatArray]] = {}
    for treatment, sbp in result["sbp"].items():
        error = (sbp[:, bins] - clean[bins]) / floor
        out[treatment] = {"bias": error.mean(axis=1), "sd": error.std(axis=1)}
    return out


SignalJob = tuple[u2.ChannelJob, U5Config, Schedule, Artifacts]


def _signal_job(args: SignalJob) -> dict[str, Any]:
    """Measure both chains on one MEArec channel at every amplitude and treatment."""
    job, config, schedule, artifacts = args
    cfg = job.config
    fs = job.fs_hz
    both = chains(config)
    x = job.clean_uv + config.noise_uv * job.background_uv
    seed = cfg.seed * 100_000 + job.channel * 100
    n_bins = int(x.size / fs / BIN_S)
    bins = schedule.bins_on(n_bins, BIN_S)
    n_effective = 2.0 * (NARROWBAND_HZ[1] - NARROWBAND_HZ[0]) * BIN_S

    narrow = emulate(x, fs, both["sbp"].frontend, seed=seed + 50, alias_zones=cfg.alias_zones)
    result = sbp_under_stimulation(
        narrow.input_referred_uv, fs, both["sbp"], schedule, artifacts, config
    )
    out: dict[str, Any] = {"sbp": _feature_change(result, bins, n_effective)}
    for treatment in TREATMENTS:
        out["sbp"][treatment]["valid"] = result["valid"][treatment]
        out["sbp"][treatment]["residual_uv"] = result["residual_uv"][treatment]

    chain = both["crossings"]
    adc = chain.adc
    wide = emulate(x, fs, chain.frontend, seed=seed + 77, alias_zones=cfg.alias_zones)
    base = sample(wide.input_referred_uv, fs, adc.fs_hz)
    onsets = schedule.onset_samples(adc.fs_hz)
    on = schedule.samples_on(base.size, adc.fs_hz)
    calibration = round(cfg.calibration_s * adc.fs_hz)
    tolerance = round(cfg.tolerance_s * adc.fs_hz)
    targets = job.targets[on[job.targets]]
    ignore = job.ignore[on[job.ignore]]

    def counts(codes: FloatArray, valid: BoolArray) -> tuple[int, int]:
        events, _ = det.detect(
            codes,
            adc.fs_hz,
            "threshold",
            factor=cfg.threshold_factor,
            calibration_samples=calibration,
        )
        events = events[on[events] & valid[events]]
        score = score_detections(events, targets, tolerance=tolerance, ignore=ignore)
        return score.true_positives, score.false_positives

    clean_codes = quantise(base, adc)[0]
    hits, false = counts(clean_codes.astype(np.float64), np.ones(base.shape, dtype=bool))
    out["crossings"] = {"clean": (hits, false), "on_s": float(on.sum() / adc.fs_hz)}
    for treatment in TREATMENTS:
        out["crossings"][treatment] = []
    for index, kernel in enumerate(artifacts.kernels["crossings"]):
        artifact = pulse_train(kernel, schedule.onsets_s, adc.fs_hz, base.size)
        codes = quantise(np.clip(base + artifact, -chain.limit_uv, chain.limit_uv), adc)[0]
        for treatment in TREATMENTS:
            treated, valid = treat(
                codes[np.newaxis, :],
                onsets,
                treatment,
                chain,
                config,
                artifacts.n_taps["crossings"],
                [artifacts.blank_samples["crossings"][index]],
            )
            out["crossings"][treatment].append(
                (*counts(treated[0], valid[0]), float(valid[0, on].mean()))
            )
    return out


def signal_level(
    recording: MearecRecording,
    config: U5Config,
    detection: u2.U2Config | None = None,
) -> dict[str, Any]:
    """Measure what stimulation does to both chains' features on MEArec.

    Stimulation starts after the detector's calibration segment, so thresholds
    are set on unstimulated data, as they would be in use.

    Returns:
        Per chain and treatment, lists over ``config.amplitudes_mv``. For SBP:
        ``bias`` and ``sd`` (mean and spread of the feature's change in
        stimulated bins, in noise floors, averaged over channels),
        ``valid`` (share of samples kept) and ``residual_uv``. For crossings:
        ``p_keep`` (share of the unstimulated chain's true detections that
        survive) and ``extra_rate_hz`` (false positives added per second).
    """
    detection = detection or u2.U2Config()
    schedule = _schedule(recording.duration_s, config, detection.calibration_s, 1)
    artifacts = build_artifacts(config, schedule.period_s)
    jobs: list[SignalJob] = [
        (job, config, schedule, artifacts) for job in u2.channel_jobs(recording, detection)
    ]
    results: list[dict[str, Any]] = u2._map(_signal_job, jobs, detection)
    out: dict[str, Any] = {
        "amplitudes_mv": list(config.amplitudes_mv),
        "n_channels": len(results),
        "n_pulses": int(schedule.onsets_s.size),
        "stimulated_s": float(np.diff(schedule.trains_s, axis=1).sum()),
        "realised_rate_hz": 1.0 / schedule.period_s,
        "blank_samples": artifacts.blank_samples,
        "blank_ms": {
            name: [1e3 * n / chains(config)[name].adc.fs_hz for n in windows]
            for name, windows in artifacts.blank_samples.items()
        },
        "n_taps": artifacts.n_taps,
        "sbp": {},
        "crossings": {},
    }
    for treatment in TREATMENTS:
        entry: dict[str, Any] = {}
        for key in ("bias", "sd", "valid", "residual_uv"):
            stacked = np.stack([r["sbp"][treatment][key] for r in results])
            entry[key] = stacked.mean(axis=0).tolist()
            if key == "bias":
                entry["bias_sd_over_channels"] = stacked.std(axis=0).tolist()
        out["sbp"][treatment] = entry
    clean_hits = sum(r["crossings"]["clean"][0] for r in results)
    clean_false = sum(r["crossings"]["clean"][1] for r in results)
    on_s = sum(r["crossings"]["on_s"] for r in results)
    out["crossings"]["clean"] = {"true_positives": clean_hits, "false_positives": clean_false}
    for treatment in TREATMENTS:
        table = np.array([r["crossings"][treatment] for r in results], dtype=np.float64)
        hits, false = table[:, :, 0].sum(axis=0), table[:, :, 1].sum(axis=0)
        out["crossings"][treatment] = {
            "p_keep": np.minimum(hits / max(clean_hits, 1), 1.0).tolist(),
            "extra_rate_hz": (np.maximum(false - clean_false, 0.0) / on_s).tolist(),
            "valid": table[:, :, 2].mean(axis=0).tolist(),
        }
    return out


def fixed_point_check(
    recording: MearecRecording, config: U5Config, channel: int = 0, seconds: float = 20.0
) -> dict[str, Any]:
    """Compare the integer canceller and filter with their floating-point twins.

    One channel, the example amplitude. The integer path is what U7 turns into
    RTL: :func:`neurochain.stim.cancel.cancel_template`, then
    :func:`neurochain.fixedpoint.fixed_highpass`, then a sum of magnitudes.
    """
    chain = chains(config)["sbp"]
    adc = chain.adc
    fs = recording.fs_hz
    n = min(round(seconds * fs), recording.clean_uv.shape[1])
    x = recording.signal_uv(channel, config.noise_uv)[:n]
    schedule = _schedule(n / fs, config, 1.0, 2)
    artifacts = build_artifacts(config, schedule.period_s)
    index = int(np.argmin(np.abs(np.asarray(config.amplitudes_mv) - config.example_mv)))
    analog = emulate(x, fs, chain.frontend, seed=config.seed + 9).input_referred_uv
    base = sample(analog, fs, adc.fs_hz)
    artifact = pulse_train(artifacts.kernels["sbp"][index], schedule.onsets_s, adc.fs_hz, base.size)
    codes = quantise(np.clip(base + artifact, -chain.limit_uv, chain.limit_uv), adc)[0]
    onsets = schedule.onset_samples(adc.fs_hz)
    taps = artifacts.n_taps["sbp"]
    out_frac = min(2, config.template_frac_bits)
    scale = float(1 << out_frac)
    fixed, valid = cancel.cancel_template(
        codes,
        onsets,
        n_taps=taps,
        shift=config.adapt_shift,
        code_min=adc.code_min,
        code_max=adc.code_max,
        frac_bits=config.template_frac_bits,
        out_frac_bits=out_frac,
    )
    floating, valid_float = cancel.cancel_template_float(
        codes.astype(np.float64),
        onsets,
        n_taps=taps,
        rate=2.0**-config.adapt_shift,
        low=adc.code_min,
        high=adc.code_max,
    )
    per_bin = round(BIN_S * adc.fs_hz)
    # The canceller's output already carries the filter's guard bits.
    filt = design_fixed_highpass(adc.fs_hz, frac_bits=8, guard_bits=0)
    integer_filtered, peak = fixed_highpass(fixed, filt)
    integer_sbp, _ = cancel.masked_mean_abs(integer_filtered / scale, valid, per_bin)
    float_sbp, _ = cancel.masked_mean_abs(
        feature_highpass(floating, adc.fs_hz), valid_float, per_bin
    )
    return {
        "channel": channel,
        "seconds": n / fs,
        "amplitude_mv": config.amplitudes_mv[index],
        "canceller_max_difference_lsb": float(np.abs(fixed / scale - floating).max()),
        "sbp_correlation": float(np.corrcoef(integer_sbp, float_sbp)[0, 1]),
        "sbp_rms_difference_lsb": float(np.sqrt(np.mean((integer_sbp - float_sbp) ** 2))),
        "sbp_mean_lsb": float(float_sbp.mean()),
        "masks_identical": bool(np.array_equal(valid, valid_float)),
        "filter_accumulator_bits": int(peak).bit_length() + 1,
        "template_bits": adc.bits + config.template_frac_bits,
    }


# ------------------------------------------------------------------------ real recording


def in_vivo(broadband_uv: FloatArray, fs_hz: float, config: U5Config) -> dict[str, Any]:
    """Inject the example artifact into a real broadband recording and treat it.

    Returns the feature change under each treatment, as in
    :func:`signal_level`, and a short stretch of the ADC's output for the
    figure, taken from the last train so the template has settled.
    """
    chain = chains(config)["sbp"]
    adc = chain.adc
    duration = broadband_uv.size / fs_hz
    schedule = _schedule(duration, config, 1.0, 3)
    artifacts = build_artifacts(config, schedule.period_s)
    index = int(np.argmin(np.abs(np.asarray(config.amplitudes_mv) - config.example_mv)))
    one = Artifacts(
        {"sbp": [artifacts.kernels["sbp"][index]]},
        {"sbp": [artifacts.blank_samples["sbp"][index]]},
        artifacts.n_taps,
    )
    analog = emulate(broadband_uv, fs_hz, chain.frontend, seed=config.seed + 11)
    result = sbp_under_stimulation(analog.input_referred_uv, fs_hz, chain, schedule, one, config)
    n_bins = result["clean"].size
    bins = schedule.bins_on(n_bins, BIN_S)
    n_effective = 2.0 * (NARROWBAND_HZ[1] - NARROWBAND_HZ[0]) * BIN_S
    change = _feature_change(result, bins, n_effective)
    period = round(schedule.period_s * adc.fs_hz)
    start = int(schedule.onset_samples(adc.fs_hz)[-6]) - 4
    window = slice(start, start + 3 * period + 4)
    lsb = adc.lsb_uv
    out: dict[str, Any] = {
        "amplitude_mv": config.amplitudes_mv[index],
        "seconds": duration,
        "n_pulses": int(schedule.onsets_s.size),
        "stimulated_bins": int(bins.sum()),
        "blank_ms": 1e3 * one.blank_samples["sbp"][0] / adc.fs_hz,
        "treatments": {
            treatment: {
                "bias": float(change[treatment]["bias"][0]),
                "sd": float(change[treatment]["sd"][0]),
                "valid": float(result["valid"][treatment][0]),
                "residual_uv": float(result["residual_uv"][treatment][0]),
            }
            for treatment in TREATMENTS
        },
        "trace": {
            "time_ms": (np.arange(window.stop - window.start) / adc.fs_hz * 1e3).tolist(),
            "onsets_ms": [(4 + k * period) / adc.fs_hz * 1e3 for k in range(3)],
            "clean_uv": (result["clean_codes"][window] * lsb).tolist(),
            "raw_uv": (result["codes"][0, window] * lsb).tolist(),
            **{
                f"{treatment}_uv": (result["treated"][treatment][0, window] * lsb).tolist()
                for treatment in ("blank", "template")
            },
            **{
                f"{treatment}_valid": result["mask"][treatment][0, window].tolist()
                for treatment in ("blank", "template")
            },
        },
    }
    return out


# ------------------------------------------------------------------------------ decoding


@dataclass
class _Fold:
    test: BoolArray
    decoder: Decoder
    mean: FloatArray
    scale: FloatArray


def fit_folds(
    name: str, data: u3.DecodingSet, features: FloatArray, config: U5Config
) -> list[_Fold]:
    """Fit one decoder per fold on the other folds, and keep it.

    Folds are contiguous blocks of trials and features are scaled by the
    training folds' statistics, exactly as in U4, so a decoder fitted once can
    be asked to predict the same held-out bins under several conditions.
    """
    make = u4.decoder_factory(name, data, config.decoding)
    folds = []
    for block in np.array_split(np.unique(data.groups), config.decoding.n_folds):
        test = np.isin(data.groups, block)
        train = features[~test]
        mean, scale = train.mean(axis=0), train.std(axis=0) + 1e-9
        decoder = make()
        decoder.fit((train - mean) / scale, data.targets[~test], data.groups[~test])
        folds.append(_Fold(test, decoder, mean, scale))
    return folds


def predict_folds(folds: list[_Fold], data: u3.DecodingSet, features: FloatArray) -> FloatArray:
    """Predict every bin with the decoder that did not see it."""
    out = np.empty_like(data.targets, dtype=np.float64)
    for fold in folds:
        scaled = (features[fold.test] - fold.mean) / fold.scale
        out[fold.test] = fold.decoder.predict(scaled, data.groups[fold.test])
    return out


def _at(table: dict[str, Any], key: str, amplitudes_mv: list[float], at_uv: FloatArray) -> Any:
    """Read a measured quantity at each channel's amplitude (linear in log amplitude)."""
    return np.interp(np.log10(at_uv / 1000.0), np.log10(amplitudes_mv), table[key])


def stimulated_features(
    kind: str,
    features: FloatArray,
    on: BoolArray,
    amplitude_uv: FloatArray,
    stim_channel: int | None,
    treatment: str,
    measured: dict[str, Any],
    rng: np.random.Generator,
) -> FloatArray:
    """Return recorded features as the chain would deliver them under stimulation.

    In stimulated bins each channel's feature is changed by what
    :func:`signal_level` measured at that channel's artifact amplitude. The
    stimulating electrode's own channel is disconnected from its amplifier
    while it stimulates, so it reads a constant (its median).

    Args:
        kind: ``"sbp"`` or ``"crossings"``.
        features: ``(n_bins, n_channels)`` unstimulated features.
        on: Which bins are stimulated.
        amplitude_uv: Artifact amplitude on each channel.
        stim_channel: Index of the stimulating electrode's channel, if it is on
            this array.
        treatment: Artifact treatment.
        measured: The output of :func:`signal_level`.
        rng: Random generator.
    """
    table = measured[kind][treatment]
    grid = measured["amplitudes_mv"]
    out = np.array(features, dtype=np.float64)
    if kind == "sbp":
        n_effective = 2.0 * (NARROWBAND_HZ[1] - NARROWBAND_HZ[0]) * BIN_S
        floor = sbp_floor(features, n_effective)
        bias = _at(table, "bias", grid, amplitude_uv)
        sd = _at(table, "sd", grid, amplitude_uv)
        noise = rng.standard_normal((int(on.sum()), features.shape[1]))
        out[on] = features[on] + floor * (bias + sd * noise)
    else:
        p_keep = np.clip(_at(table, "p_keep", grid, amplitude_uv), 0.0, 1.0)
        extra = _at(table, "extra_rate_hz", grid, amplitude_uv)
        counts = np.rint(features[on]).astype(np.int64)
        out[on] = rng.binomial(counts, p_keep) + rng.poisson(extra * BIN_S, size=counts.shape)
    if stim_channel is not None and stim_channel < features.shape[1]:
        out[on, stim_channel] = np.median(features[:, stim_channel])
    return out


def _chain_features(data: u3.DecodingSet, u3_summary: dict[str, Any], role: str) -> FloatArray:
    """Rebuild the features U3 gave its winner or its baseline on this dataset."""
    labels = [chain.label for chain in u3.grid(u3.U3Config())]
    row = u3_summary["configurations"][u3_summary["datasets"][data.name][role]]
    return u4.chain_features(data, row, labels.index(row["label"]), u3_summary["config"]["seed"])


def decoding(
    data: u3.DecodingSet,
    u3_summary: dict[str, Any],
    measured: dict[str, Any],
    config: U5Config,
) -> dict[str, Any]:
    """Score the decoders on stimulated bins, for every geometry and treatment.

    The main protocol fits each decoder on unstimulated data and tests it with
    stimulation on, the case of a decoder calibrated before feedback is
    switched on. A second pass, on the first geometry, refits each decoder on
    stimulated data.

    Scores are R² over the stimulated bins of the held-out folds. ``loss`` is
    the paired difference from the same decoder's score on the same bins
    without stimulation.
    """
    cfg = config.decoding
    n_bins = data.targets.shape[0]
    on = _schedule(n_bins * data.bin_s, config, 0.0, 4).bins_on(n_bins, data.bin_s)
    truth, groups = data.targets[on], data.groups[on]
    roles = {"sbp": "winner", "crossings": "baseline"}
    rows: list[dict[str, Any]] = []
    reference: dict[str, dict[str, float]] = {}

    def score(predictions: FloatArray, baseline: FloatArray) -> dict[str, float]:
        r2, low, high = bootstrap_r2(
            truth, predictions[on], groups, n_boot=cfg.n_boot, seed=cfg.seed
        )
        loss, loss_low, loss_high = bootstrap_r2_difference(
            truth, predictions[on], baseline[on], groups, n_boot=cfg.n_boot, seed=cfg.seed
        )
        return {
            "r2": r2,
            "r2_low": low,
            "r2_high": high,
            "loss": loss,
            "loss_low": loss_low,
            "loss_high": loss_high,
        }

    for c, kind in enumerate(CHAINS):
        features = _chain_features(data, u3_summary, roles[kind])
        folds = {name: fit_folds(name, data, features, config) for name in cfg.decoders}
        clean = {name: predict_folds(folds[name], data, features) for name in cfg.decoders}
        reference[kind] = {name: r2_score(truth, clean[name][on]) for name in cfg.decoders} | {
            f"{name}_all_bins": r2_score(data.targets, clean[name]) for name in cfg.decoders
        }
        for s, distance_mm in enumerate(config.scenarios_mm):
            amplitude, stim_channel = scenario_amplitudes_uv(distance_mm, config)
            amplitude = amplitude[: features.shape[1]]
            for t, treatment in enumerate(TREATMENTS):
                changed = stimulated_features(
                    kind,
                    features,
                    on,
                    amplitude,
                    stim_channel,
                    treatment,
                    measured,
                    np.random.default_rng([config.seed, c, s, t]),
                )
                for name in cfg.decoders:
                    rows.append(
                        {
                            "chain": kind,
                            "scenario": scenario_name(distance_mm),
                            "treatment": treatment,
                            "decoder": name,
                            "fitted": "without stimulation",
                            **score(predict_folds(folds[name], data, changed), clean[name]),
                        }
                    )
                for name in cfg.decoders if s == 0 else ():
                    refit = fit_folds(name, data, changed, config)
                    rows.append(
                        {
                            "chain": kind,
                            "scenario": scenario_name(distance_mm),
                            "treatment": treatment,
                            "decoder": name,
                            "fitted": "with stimulation",
                            **score(predict_folds(refit, data, changed), clean[name]),
                        }
                    )
    return {
        "dataset": data.name,
        "n_bins": n_bins,
        "stimulated_bins": int(on.sum()),
        "no_stimulation_r2": reference,
        "rows": rows,
    }


# ---------------------------------------------------------------------------------- cost


def canceller_cost(config: U5Config, n_taps: dict[str, int]) -> dict[str, Any]:
    """Return the canceller's cost per channel in the terms of U3's power model.

    The canceller has no multiplier: per sample it does one subtraction (the
    output) and one shifted addition (the update). Each is charged here as a
    full multiply-accumulate, so the figure is an upper bound until U7
    synthesises the logic. It scales with the sample rate, so the same
    canceller costs fifteen times more on the 30 kS/s chain.
    """
    adds = 2
    energy = config.power.energy_per_multiply_j
    ref = u3.REFERENCE
    configs = {
        "sbp": ChainConfig("sbp", config.sbp_current_a, config.sbp_bits, SBP_FS_HZ),
        "crossings": ref,
    }
    out: dict[str, Any] = {"adds_per_sample": adds, "multiplies_per_sample": 0, "chains": {}}
    for name, chain in chains(config).items():
        canceller_uw = adds * energy * chain.adc.fs_hz * 1e6
        chain_uw = chain_power(configs[name], config.power).total_w * 1e6
        out["chains"][name] = {
            "canceller_uw": canceller_uw,
            "chain_uw": chain_uw,
            "share_of_chain": canceller_uw / chain_uw,
            "n_taps": n_taps[name],
            "template_bits_per_channel": n_taps[name]
            * (chain.adc.bits + config.template_frac_bits),
        }
    return out


# ------------------------------------------------------------------------------- compute


def _key(summary: dict[str, Any], config: U5Config) -> dict[str, Any]:
    """Collect the unit's headline numbers from the full summary."""
    recover = summary["recovery"]["chains"]
    index = int(np.argmin(np.abs(np.asarray(config.amplitudes_mv) - config.example_mv)))
    key: dict[str, Any] = {
        "example_mv": config.amplitudes_mv[index],
        "recovery_ms": {name: recover[name]["model_ms"][index] for name in recover},
        "spice_recovery_ms": {name: recover[name]["spice_ms"][index] for name in recover},
        "valid": {
            treatment: summary["signal"]["sbp"][treatment]["valid"][index]
            for treatment in TREATMENTS
        },
        "bias": {
            treatment: summary["signal"]["sbp"][treatment]["bias"][index]
            for treatment in TREATMENTS
        },
    }
    decode = summary.get("decoding")
    if decode:
        scenario = scenario_name(config.scenarios_mm[0])
        decoder = config.decoding.decoders[-1]
        key["scenario"] = scenario
        key["decoder"] = decoder
        key["no_stimulation_r2"] = decode["no_stimulation_r2"]["sbp"][decoder]
        for row in decode["rows"]:
            if (
                row["scenario"] == scenario
                and row["decoder"] == decoder
                and row["fitted"] == "without stimulation"
            ):
                key[f"{row['chain']}_{row['treatment']}"] = {
                    k: row[k] for k in ("r2", "r2_low", "r2_high", "loss", "loss_low", "loss_high")
                }
    return key


def compute_u5(
    recording: MearecRecording,
    *,
    config: U5Config | None = None,
    detection: u2.U2Config | None = None,
    u3_summary: dict[str, Any] | None = None,
    data: u3.DecodingSet | None = None,
    broadband: tuple[FloatArray, float] | None = None,
    with_spice: bool = True,
) -> dict[str, Any]:
    """Run the U5 analysis.

    Args:
        recording: The MEArec recording, for the signal-level measurements.
        config: Analysis parameters.
        detection: Detector settings and worker count, shared with U2.
        u3_summary: The contents of ``results/u3_metrics.json``; with ``data``
            it enables the decoding step.
        data: The decoding dataset.
        broadband: ``(samples_uv, fs_hz)`` of a real recording for the
            injection example; skipped when ``None``.
        with_spice: Run ngspice for the recovery comparison.

    Returns:
        The JSON-serialisable results.
    """
    config = config or U5Config()
    measured = signal_level(recording, config, detection)
    as_dict = asdict(config)
    summary: dict[str, Any] = {
        "config": as_dict,
        "propagation": propagation(config),
        "recovery": recovery(config, with_spice=with_spice),
        "signal": measured,
        "fixed_point": fixed_point_check(recording, config),
        "in_vivo": None if broadband is None else in_vivo(*broadband, config),
        "decoding": None
        if data is None or u3_summary is None
        else decoding(data, u3_summary, measured, config),
        "cost": canceller_cost(config, measured["n_taps"]),
    }
    summary["key"] = _key(summary, config)
    return summary
