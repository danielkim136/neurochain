"""U3 analysis: accuracy against microwatts across the whole sensing chain.

For every configuration in a grid (feature, amplifier current, ADC bits and
sample rate) this computes two things and puts them on one plot:

* **Power per channel**, from :mod:`neurochain.power`.
* **Decoding accuracy on held-out data**, by degrading recorded features the
  way that configuration would (see :mod:`neurochain.bridge`) and decoding them
  with the same linear readout.

The degradation of threshold crossings is measured, not assumed: each
configuration is run on the MEArec recording at in-vivo noise, and the fraction
of real spikes it keeps and the false positives it adds are read off against
ground truth. The degradation of spiking-band power is the analytic ratio of
noise powers, checked against MEArec in ``tests/test_u3.py``.

Held-out data means contiguous blocks of trials within a session. Decoding
across the two NoMAD sessions, 274 days apart, fails for a fixed decoder; that
number is reported here and is the subject of U6.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from neurochain.adc import Adc, convert
from neurochain.analysis import u2
from neurochain.bridge import degrade_crossings, degrade_sbp, sbp_floor, sbp_noise_power_ratio
from neurochain.data import nomad
from neurochain.data.mearec import MearecRecording
from neurochain.decode.ridge import blocked_cross_validation, fit_ridge, lagged, r_squared
from neurochain.dsp.spectra import welch_psd
from neurochain.features import detect as det
from neurochain.features.score import DetectionScore
from neurochain.frontend.electrode import UTAH_PT_IN_VITRO
from neurochain.frontend.emulator import emulate
from neurochain.frontend.model import BandwidthError, FrontEnd, design_lna
from neurochain.power import (
    NARROWBAND_HZ,
    SBP_FS_HZ,
    ChainConfig,
    PowerModel,
    chain_power,
)
from neurochain.recording import FloatArray, IntArray

REFERENCE = ChainConfig("crossings", 10.0e-6, 10, 30_000.0)
"""The chain the recordings are assumed to have been made with."""

BIN_S = 0.05
"""Bin width the bridge is measured at."""


@dataclass(frozen=True)
class U3Config:
    """Parameters of the U3 analysis.

    Attributes:
        seed: Seed for every stochastic step.
        currents_a: Amplifier supply currents in the grid.
        crossing_bits: ADC resolutions tried for the wideband chain.
        crossing_fs_hz: ADC sample rates tried for the wideband chain.
        sbp_bits: ADC resolutions tried for the narrowband chain.
        reference_noise_uv: Neural background level the bridge is measured at.
        n_repeats: Random realisations of the crossing degradation per
            configuration.
        tolerance: Share of the conventional chain's accuracy a configuration may
            lose and still count as matching it; the cheapest such configuration
            is the winner.
        nomad_rebin: 20 ms NoMAD bins summed into one decoding bin.
        power: Technology constants.
    """

    seed: int = 0
    currents_a: tuple[float, ...] = (
        0.025e-6, 0.05e-6, 0.1e-6, 0.25e-6, 0.5e-6, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6,
    )  # fmt: skip
    crossing_bits: tuple[int, ...] = (6, 8, 10)
    crossing_fs_hz: tuple[float, ...] = (10_000.0, 15_000.0, 30_000.0)
    sbp_bits: tuple[int, ...] = (6, 8, 10)
    reference_noise_uv: float = 13.0
    n_repeats: int = 3
    tolerance: float = 0.05
    nomad_rebin: int = 3
    power: PowerModel = PowerModel()  # noqa: RUF009 - frozen dataclass, safe as a default


def wideband_frontend(i_total_a: float, model: PowerModel) -> FrontEnd:
    """Return the 1 Hz to 7.5 kHz front end at a given supply current."""
    return FrontEnd(design_lna(i_total_a=i_total_a, nef=model.nef), UTAH_PT_IN_VITRO)


def narrowband_frontend(i_total_a: float, model: PowerModel) -> FrontEnd:
    """Return the 300 Hz to 1 kHz front end at a given supply current."""
    low, high = NARROWBAND_HZ
    lna = design_lna(f_high_hz=high, f_low_hz=low, i_total_a=i_total_a, nef=model.nef)
    return FrontEnd(lna, UTAH_PT_IN_VITRO)


QualityJob = tuple[u2.ChannelJob, float, tuple[Adc, ...]]


def _binned_events(events: IntArray, fs_hz: float, job: u2.ChannelJob, bin_s: float) -> FloatArray:
    """Count events in bins of ``bin_s``, after the calibration segment."""
    per_bin = bin_s * fs_hz
    n_bins = int(job.clean_uv.size / job.fs_hz / bin_s)
    counts = np.bincount((events / per_bin).astype(np.int64), minlength=n_bins)[:n_bins]
    return np.asarray(counts[round(job.config.calibration_s / bin_s) :], dtype=np.float64)


def _events(job: u2.ChannelJob, analog_uv: FloatArray, adc: Adc) -> IntArray:
    """Run the threshold detector on one channel through one ADC."""
    codes = convert(analog_uv, job.fs_hz, adc).codes.astype(np.float64)
    events, _ = det.detect(
        codes,
        adc.fs_hz,
        "threshold",
        factor=job.config.threshold_factor,
        calibration_samples=round(job.config.calibration_s * adc.fs_hz),
    )
    return events


def _quality_job(args: QualityJob) -> list[DetectionScore]:
    """Score the threshold detector on one channel for several ADCs."""
    job, noise, adcs = args
    cfg = job.config
    x = job.clean_uv + noise * job.background_uv
    seed = cfg.seed * 100_000 + job.channel * 100 + 77
    analog = emulate(x, job.fs_hz, job.wideband, seed=seed, alias_zones=cfg.alias_zones)
    return [u2._score(job, _events(job, analog.input_referred_uv, adc), adc.fs_hz) for adc in adcs]


def measure_crossings(
    recording: MearecRecording, config: U3Config, detection: u2.U2Config | None = None
) -> dict[tuple[float, int, float], dict[str, float]]:
    """Measure what each wideband configuration does to threshold crossings.

    Returns:
        For each ``(current, bits, sample rate)``: the rate of detections that
        land on a real spike, the rate of those that land on nothing (both per
        channel per second), and the pooled F1.
    """
    detection = detection or u2.U2Config()
    adcs = tuple(
        Adc(fs_hz=fs, bits=bits) for bits in config.crossing_bits for fs in config.crossing_fs_hz
    )
    jobs = u2.channel_jobs(recording, detection)
    seconds = (recording.duration_s - detection.calibration_s) * recording.n_channels
    out: dict[tuple[float, int, float], dict[str, float]] = {}
    for current in config.currents_a:
        try:
            frontend = wideband_frontend(current, config.power)
        except BandwidthError:
            continue  # this current cannot give a 7.5 kHz amplifier at all
        work: list[QualityJob] = [
            (replace(job, wideband=frontend), config.reference_noise_uv, adcs) for job in jobs
        ]
        per_channel: list[list[DetectionScore]] = u2._map(_quality_job, work, detection)
        for index, adc in enumerate(adcs):
            pooled = DetectionScore(0, 0, 0)
            for scores in per_channel:
                pooled = pooled + scores[index]
            out[(current, adc.bits, adc.fs_hz)] = {
                "real_rate_hz": (pooled.true_positives + pooled.ignored) / seconds,
                "false_rate_hz": pooled.false_positives / seconds,
                "f1": pooled.f1,
            }
    return out


def background_psd(
    recording: MearecRecording, reference_noise_uv: float
) -> tuple[FloatArray, FloatArray]:
    """Return the neural background's PSD at the reference level, in µV²/Hz.

    Averaged over channels. The loader scales the background to 1 µV RMS over
    300 Hz to 7.5 kHz, so the reference level multiplies its power.
    """
    freqs, psd = welch_psd(recording.background_uv, recording.fs_hz, resolution_hz=10.0)
    return freqs, np.asarray(psd.mean(axis=0) * reference_noise_uv**2, dtype=np.float64)


def grid(config: U3Config) -> list[ChainConfig]:
    """Return every configuration in the design space."""
    configs = [replace(REFERENCE, feature="raw")]
    for current in config.currents_a:
        for bits in config.crossing_bits:
            for fs in config.crossing_fs_hz:
                configs.append(ChainConfig("crossings", current, bits, fs))
        for bits in config.sbp_bits:
            configs.append(ChainConfig("sbp", current, bits, SBP_FS_HZ))
        for bits in config.crossing_bits:
            configs.append(ChainConfig("both", current, bits, 30_000.0))
    return configs


@dataclass(frozen=True)
class Degradation:
    """What one configuration does to each feature, relative to the reference chain.

    Attributes:
        p_keep: Share of the reference chain's real detections this chain also
            makes; recorded crossings are thinned by it.
        extra_rate_hz: False detections this chain makes beyond those that
            survive the thinning, per channel per second; added as noise.
        sbp_noise_ratio: Noise power in the SBP signal relative to the
            reference chain's.
    """

    p_keep: float = 1.0
    extra_rate_hz: float = 0.0
    sbp_noise_ratio: float = 1.0


def crossing_degradation(
    chain: ChainConfig, crossings: dict[tuple[float, int, float], dict[str, float]]
) -> tuple[float, float]:
    """Return ``(p_keep, extra_rate_hz)`` for a wideband configuration.

    A recording cannot be improved, so a current above the reference chain's
    is treated as the reference current: only the ADC then differs.
    """
    current = min(chain.i_total_a, REFERENCE.i_total_a)
    here = crossings[(current, chain.bits, chain.fs_hz)]
    ref = crossings[(REFERENCE.i_total_a, REFERENCE.bits, REFERENCE.fs_hz)]
    p_keep = min(1.0, here["real_rate_hz"] / ref["real_rate_hz"])
    extra = max(0.0, here["false_rate_hz"] - p_keep * ref["false_rate_hz"])
    return p_keep, extra


def sbp_ratio(
    chain: ChainConfig, background: tuple[FloatArray, FloatArray], model: PowerModel
) -> float:
    """Return a chain's SBP noise power relative to the reference current's."""

    def absolute(i_total_a: float, bits: int) -> float:
        adc = Adc(fs_hz=chain.fs_hz, bits=bits)
        if chain.feature == "sbp":
            return sbp_noise_power_ratio(narrowband_frontend(i_total_a, model), adc, *background)
        return sbp_noise_power_ratio(
            wideband_frontend(i_total_a, model), adc, *background, digital_band_hz=NARROWBAND_HZ
        )

    here = absolute(chain.i_total_a, chain.bits)
    return max(1.0, here / absolute(REFERENCE.i_total_a, REFERENCE.bits))


def degradation_of(
    chain: ChainConfig,
    crossings: dict[tuple[float, int, float], dict[str, float]],
    background: tuple[FloatArray, FloatArray],
    model: PowerModel,
) -> Degradation:
    """Work out how a configuration changes recorded features.

    Raises:
        BandwidthError: If the amplifier cannot be built at this current.
    """
    if chain.feature == "raw":
        return Degradation()
    p_keep, extra = 1.0, 0.0
    if chain.feature in ("crossings", "both"):
        if (min(chain.i_total_a, REFERENCE.i_total_a), chain.bits, chain.fs_hz) not in crossings:
            wideband_frontend(chain.i_total_a, model)  # raises with the reason
        p_keep, extra = crossing_degradation(chain, crossings)
    ratio = sbp_ratio(chain, background, model) if chain.feature in ("sbp", "both") else 1.0
    return Degradation(p_keep, extra, ratio)


@dataclass(frozen=True)
class DecodingSet:
    """Recorded features and behaviour, ready for the readout.

    Attributes:
        name: Dataset label.
        crossings: ``(n_bins, n_channels)`` threshold-crossing counts.
        sbp: ``(n_bins, n_channels)`` spiking-band power, or ``None``.
        targets: ``(n_bins, n_targets)`` behaviour to decode.
        groups: ``(n_bins,)`` trial or segment label, for blocked folds.
        bin_s: Bin width in seconds.
        n_lags: Bins of history given to the readout.
    """

    name: str
    crossings: FloatArray
    sbp: FloatArray | None
    targets: FloatArray
    groups: IntArray
    bin_s: float
    n_lags: int


def nason_set() -> DecodingSet:
    """Return the Nason online data: 96 channels, finger velocity, 50 ms bins."""
    data = u2.online_features()
    return DecodingSet(
        name="nason",
        crossings=data["crossings"],
        sbp=data["sbp"],
        targets=data["velocity"][:, np.newaxis],
        groups=data["groups"].astype(np.int64),
        bin_s=0.05,
        n_lags=5,
    )


def nomad_set(session: nomad.NomadSession, rebin: int) -> DecodingSet:
    """Return one NoMAD session: 192 channels, cursor velocity, rebinned.

    Bins are summed in groups of ``rebin`` inside each contiguous segment, and
    segments shorter than one group are dropped.
    """
    starts = [*session.segment_starts.tolist(), session.n_bins]
    counts, velocity, groups = [], [], []
    for index, (a, b) in enumerate(itertools.pairwise(starts)):
        n = (b - a) // rebin
        if n < 2:
            continue
        use = slice(a, a + n * rebin)
        counts.append(session.binned_spikes[use].reshape(n, rebin, -1).sum(axis=1))
        velocity.append(session.cursor_vel[use].reshape(n, rebin, 2).mean(axis=1))
        groups.append(np.full(n, index, dtype=np.int64))
    return DecodingSet(
        name=f"nomad_{session.name}",
        crossings=np.concatenate(counts).astype(np.float64),
        sbp=None,
        targets=np.concatenate(velocity),
        groups=np.concatenate(groups),
        bin_s=session.bin_s * rebin,
        n_lags=4,
    )


def features_for(
    data: DecodingSet,
    chain: ChainConfig,
    degradation: Degradation,
    rng: np.random.Generator,
) -> FloatArray | None:
    """Return the features a configuration would deliver, or ``None`` if it cannot."""
    parts = []
    if chain.feature in ("crossings", "both", "raw"):
        parts.append(
            degrade_crossings(
                data.crossings, degradation.p_keep, degradation.extra_rate_hz, data.bin_s, rng
            )
        )
    if chain.feature in ("sbp", "both", "raw"):
        if data.sbp is None:
            # Without recorded SBP only crossings can be assessed: a raw stream
            # reduces to them, and the SBP chains cannot be scored at all.
            return parts[0] if chain.feature == "raw" else None
        n_effective = 2.0 * (NARROWBAND_HZ[1] - NARROWBAND_HZ[0]) * data.bin_s
        parts.append(
            degrade_sbp(
                data.sbp,
                degradation.sbp_noise_ratio,
                sbp_floor(data.sbp, n_effective),
                n_effective=n_effective,
                rng=rng,
            )
        )
    return np.concatenate(parts, axis=1)


def decode(data: DecodingSet, features: FloatArray) -> float:
    """Return held-out R², averaged over the behaviour's dimensions."""
    return blocked_cross_validation(features, data.targets, data.groups, n_lags=data.n_lags).r2


def cross_session_r2(train: DecodingSet, test: DecodingSet) -> float:
    """Fit the readout on one session and score it on another, with no recalibration."""
    x_train = lagged(train.crossings, train.groups, train.n_lags)
    x_test = lagged(test.crossings, test.groups, test.n_lags)
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0) + 1e-9
    scores = []
    for d in range(train.targets.shape[1]):
        weights, intercept = fit_ridge((x_train - mean) / scale, train.targets[:, d], 0.01)
        prediction = (x_test - mean) / scale @ weights + intercept
        scores.append(r_squared(test.targets[:, d], prediction))
    return float(np.mean(scores))


BridgeJob = tuple[u2.ChannelJob, FrontEnd, FrontEnd, FrontEnd, FrontEnd, float, float, float, float]


def _bridge_job(args: BridgeJob) -> dict[str, float]:
    """Compare bridge-predicted features with directly simulated ones on one channel."""
    job, wide_ref, wide_low, narrow_ref, narrow_low, noise, p_keep, extra, ratio = args
    cfg = job.config
    fs = job.fs_hz
    rng = np.random.default_rng(cfg.seed * 7919 + job.channel)
    x = job.clean_uv + noise * job.background_uv
    seed = cfg.seed * 100_000 + job.channel * 100 + 77
    adc = Adc(fs_hz=REFERENCE.fs_hz, bits=REFERENCE.bits)
    out: dict[str, float] = {}

    def record(name: str, actual: FloatArray, predicted: FloatArray, reference: FloatArray) -> None:
        for label, series in (("actual", actual), ("predicted", predicted)):
            out[f"{name}_mean_{label}"] = float(series.mean())
            out[f"{name}_sd_{label}"] = float(series.std())
            constant = series.std() == 0.0 or reference.std() == 0.0
            out[f"{name}_r_{label}"] = (
                0.0 if constant else float(np.corrcoef(series, reference)[0, 1])
            )

    def crossing_counts(frontend: FrontEnd) -> FloatArray:
        analog = emulate(x, fs, frontend, seed=seed, alias_zones=cfg.alias_zones)
        return _binned_events(_events(job, analog.input_referred_uv, adc), adc.fs_hz, job, BIN_S)

    # Crossings are judged against the truth: the real spikes on this channel.
    real = np.sort(np.concatenate((job.targets, job.ignore)))
    truth = _binned_events(real, fs, job, BIN_S)
    reference_counts = crossing_counts(wide_ref)
    predicted = degrade_crossings(reference_counts[:, np.newaxis], p_keep, extra, BIN_S, rng)[:, 0]
    record("crossings", crossing_counts(wide_low), predicted, truth)

    sbp_adc = Adc(fs_hz=SBP_FS_HZ)
    per_bin = round(BIN_S * SBP_FS_HZ)
    start = round(cfg.calibration_s / BIN_S)

    def sbp(frontend: FrontEnd) -> FloatArray:
        analog = emulate(x, fs, frontend, seed=seed + 50, alias_zones=cfg.alias_zones)
        sampled = convert(analog.input_referred_uv, fs, sbp_adc).microvolts
        n = sampled.size // per_bin
        return np.asarray(
            np.abs(sampled[: n * per_bin]).reshape(n, per_bin).mean(axis=1)[start:],
            dtype=np.float64,
        )

    reference_sbp = sbp(narrow_ref)
    n_effective = 2.0 * (NARROWBAND_HZ[1] - NARROWBAND_HZ[0]) * BIN_S
    predicted_sbp = degrade_sbp(
        reference_sbp[:, np.newaxis],
        ratio,
        sbp_floor(reference_sbp[:, np.newaxis], n_effective),
        n_effective=n_effective,
        rng=rng,
    )[:, 0]
    record("sbp", sbp(narrow_low), predicted_sbp, reference_sbp)
    return out


def bridge_check(
    recording: MearecRecording,
    config: U3Config,
    detection: u2.U2Config,
    crossings: dict[tuple[float, int, float], dict[str, float]],
    *,
    low_current_a: float = 0.5e-6,
) -> dict[str, float]:
    """Test the bridge where the truth is known.

    On MEArec the worse chain can be run directly. So: compute a feature with
    the reference chain, degrade it with the bridge, and compare the result
    with the same feature computed through the worse chain itself. If the
    bridge is sound the two agree in level, in spread, and in how well each
    still follows the reference feature.

    Returns:
        For crossings and for SBP: the feature's mean, its standard deviation
        across bins, and a correlation, each as measured through the
        low-current chain (``actual``) and as the bridge predicts
        (``predicted``), averaged over channels. The correlation is with the
        true spike counts for crossings, and with the reference chain's SBP for
        SBP.
    """
    model = config.power
    background = background_psd(recording, config.reference_noise_uv)
    p_keep, extra = crossing_degradation(ChainConfig("crossings", low_current_a), crossings)
    ratio = sbp_ratio(ChainConfig("sbp", low_current_a), background, model)
    fronts = (
        wideband_frontend(REFERENCE.i_total_a, model),
        wideband_frontend(low_current_a, model),
        narrowband_frontend(REFERENCE.i_total_a, model),
        narrowband_frontend(low_current_a, model),
    )
    work: list[BridgeJob] = [
        (job, *fronts, config.reference_noise_uv, p_keep, extra, ratio)
        for job in u2.channel_jobs(recording, detection)
    ]
    per_channel: list[dict[str, float]] = u2._map(_bridge_job, work, detection)
    out = {key: float(np.mean([r[key] for r in per_channel])) for key in per_channel[0]}
    out.update(
        low_current_ua=low_current_a * 1e6,
        p_keep=p_keep,
        extra_rate_hz=extra,
        sbp_noise_ratio=ratio,
    )
    return out


def pareto_front(power_uw: FloatArray, accuracy: FloatArray) -> IntArray:
    """Return the indices of points no other point beats on both power and accuracy.

    Sorted by increasing power. A point is kept if it is more accurate than
    every cheaper point.
    """
    order = np.lexsort((-accuracy, power_uw))
    front, best = [], -np.inf
    for index in order:
        if accuracy[index] > best:
            front.append(int(index))
            best = float(accuracy[index])
    return np.asarray(front, dtype=np.int64)


def winner(power_uw: FloatArray, accuracy: FloatArray, target: float) -> int:
    """Return the cheapest point whose accuracy is at least ``target``."""
    good = np.flatnonzero(accuracy >= target)
    return int(good[np.argmin(power_uw[good])])


def rescore(
    summary: dict[str, Any], model: PowerModel, tolerance: float
) -> dict[str, dict[str, Any]]:
    """Recompute power and the cheapest match under different technology constants.

    Accuracy does not depend on the power constants, so the sweep need not be
    rerun. One subtlety: a different NEF changes the current needed for the
    same noise. Each stored configuration is kept at its noise level, and its
    amplifier current is scaled by ``(NEF / NEF_swept)**2``.

    Returns:
        Per dataset: the conventional chain's power and R², and the label,
        power and R² of the cheapest configuration within ``tolerance`` of it.
    """
    swept_nef = summary["config"]["power"]["nef"]
    rows = summary["configurations"]
    power = []
    for row in rows:
        chain = ChainConfig(row["feature"], row["i_total_ua"] * 1e-6, row["bits"], row["fs_hz"])
        breakdown = chain_power(chain, model)
        lna_w = breakdown.lna_w * (model.nef / swept_nef) ** 2
        power.append((breakdown.total_w - breakdown.lna_w + lna_w) * 1e6)
    power_uw = np.asarray(power)
    out: dict[str, dict[str, Any]] = {}
    for name, info in summary["datasets"].items():
        indices = np.flatnonzero([name in row["r2"] and row["feature"] != "raw" for row in rows])
        accuracy = np.array([rows[i]["r2"][name]["mean"] for i in indices])
        target = (1.0 - tolerance) * info["baseline_r2"]
        best = int(indices[winner(power_uw[indices], accuracy, target)])
        out[name] = {
            "baseline_uw": float(power_uw[info["baseline"]]),
            "baseline_r2": info["baseline_r2"],
            "winner": rows[best]["label"],
            "winner_uw": float(power_uw[best]),
            "winner_r2": rows[best]["r2"][name]["mean"],
        }
    raw = next((i for i, row in enumerate(rows) if row["feature"] == "raw"), None)
    if raw is not None:
        out["raw_stream"] = {"total_uw": float(power_uw[raw])}
    return out


def input_range_check(
    recording_uv: FloatArray, fs_hz: float, model: PowerModel
) -> dict[str, float]:
    """Measure how much of a real broadband channel fits a ±1 mV converter.

    The wideband chain passes the local field potential, which on a real
    channel is far larger than the spikes. The narrowband chain's amplifier
    removes it before the converter.

    Args:
        recording_uv: A broadband recording in microvolts (one channel).
        fs_hz: Its sample rate.
        model: Technology constants.

    Returns:
        The recording's RMS, and the fraction of samples outside the ADC range
        for the wideband chain and for the narrowband chain.
    """
    limit = Adc().input_range_uv
    wide = emulate(
        recording_uv, fs_hz, wideband_frontend(REFERENCE.i_total_a, model), seed=0, add_noise=False
    ).input_referred_uv
    narrow = emulate(
        recording_uv,
        fs_hz,
        narrowband_frontend(REFERENCE.i_total_a, model),
        seed=0,
        add_noise=False,
    ).input_referred_uv
    return {
        "broadband_rms_uv": float(np.std(recording_uv)),
        "wideband_clipped": float(np.mean(np.abs(wide) > limit)),
        "narrowband_clipped": float(np.mean(np.abs(narrow) > limit)),
    }


def compute_u3(
    recording: MearecRecording,
    *,
    config: U3Config | None = None,
    detection: u2.U2Config | None = None,
    sets: list[DecodingSet] | None = None,
    broadband: tuple[FloatArray, float] | None = None,
) -> dict[str, Any]:
    """Run the co-design sweep and return a JSON-serialisable summary.

    Args:
        recording: The MEArec recording the bridge is measured on.
        config: Sweep parameters.
        detection: Detection and scoring parameters (U2's).
        sets: Decoding datasets; defaults to the Nason online data and both
            NoMAD sessions.
        broadband: A real broadband channel in microvolts and its sample rate,
            for the input-range check.
    """
    config = config or U3Config()
    detection = detection or u2.U2Config()
    cross_session: dict[str, float] | None = None
    if sets is None:
        sessions = nomad.load_sessions()
        nomad_sets = [nomad_set(s, config.nomad_rebin) for s in sessions]
        sets = [nason_set(), *nomad_sets]
        by_time = sorted(zip(sessions, nomad_sets, strict=True), key=lambda p: p[0].start_time_ns)
        (early_session, early), (late_session, late) = by_time
        cross_session = {
            "days_apart": abs(nomad.days_between(early_session, late_session)),
            "train_early_test_late": cross_session_r2(early, late),
            "train_late_test_early": cross_session_r2(late, early),
        }

    crossings = measure_crossings(recording, config, detection)
    background = background_psd(recording, config.reference_noise_uv)
    rows: list[dict[str, Any]] = []
    infeasible: list[dict[str, str]] = []
    for position, chain in enumerate(grid(config)):
        # Each configuration draws from its own stream, so a single row can be
        # recomputed without rerunning the rest.
        rng = np.random.default_rng([config.seed, position])
        try:
            degradation = degradation_of(chain, crossings, background, config.power)
        except BandwidthError as reason:
            infeasible.append({"label": chain.label, "reason": str(reason)})
            continue
        row: dict[str, Any] = {
            "label": chain.label,
            "feature": chain.feature,
            "i_total_ua": chain.i_total_a * 1e6,
            "bits": chain.bits,
            "fs_hz": chain.fs_hz,
            "bit_rate_bps": chain.bit_rate_bps,
            **chain_power(chain, config.power).as_microwatts(),
            **asdict(degradation),
            "r2": {},
        }
        stochastic = (
            degradation.p_keep < 1.0
            or degradation.extra_rate_hz > 0.0
            or degradation.sbp_noise_ratio > 1.0
        )
        for data in sets:
            scores = []
            for _ in range(config.n_repeats if stochastic else 1):
                features = features_for(data, chain, degradation, rng)
                if features is None:
                    break
                scores.append(decode(data, features))
            if scores:
                row["r2"][data.name] = {
                    "mean": float(np.mean(scores)),
                    "sd": float(np.std(scores)),
                    "n": len(scores),
                }
        rows.append(row)

    summary: dict[str, Any] = {
        "config": asdict(config),
        "reference": REFERENCE.label,
        "crossing_quality": [
            {
                "i_total_ua": k[0] * 1e6,
                "bits": k[1],
                "fs_hz": k[2],
                **v,
                "p_keep": crossing_degradation(ChainConfig("crossings", *k), crossings)[0],
            }
            for k, v in crossings.items()
        ],
        "configurations": rows,
        "infeasible": infeasible,
        "bridge_check": bridge_check(recording, config, detection, crossings),
        "datasets": {},
        "cross_session": cross_session,
        "input_range": (None if broadband is None else input_range_check(*broadband, config.power)),
    }
    power = np.array([row["total_uw"] for row in rows])
    for data in sets:
        onboard = np.array([data.name in row["r2"] and row["feature"] != "raw" for row in rows])
        indices = np.flatnonzero(onboard)
        accuracy = np.array([rows[i]["r2"][data.name]["mean"] for i in indices])
        front = indices[pareto_front(power[indices], accuracy)]
        baseline = next(i for i, row in enumerate(rows) if row["label"] == REFERENCE.label)
        baseline_r2 = rows[baseline]["r2"][data.name]["mean"]
        best = indices[winner(power[indices], accuracy, (1.0 - config.tolerance) * baseline_r2)]
        summary["datasets"][data.name] = {
            "n_bins": int(data.groups.size),
            "n_channels": int(data.crossings.shape[1]),
            "bin_s": data.bin_s,
            "baseline": int(baseline),
            "baseline_r2": float(baseline_r2),
            "most_accurate": int(indices[int(np.argmax(accuracy))]),
            "pareto": [int(i) for i in front],
            "winner": int(best),
        }
    return summary
