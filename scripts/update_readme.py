"""Regenerate the results tables in README.md from ``results/*.json``.

The README never contains a hand-typed result. Each table sits between a pair
of markers and is rebuilt from the metrics file that the unit's own script
wrote. ``--check`` exits non-zero if the README is out of date, which is what
pre-commit and CI run.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from neurochain.paths import repo_root, results_dir


def u0_table(m: dict[str, Any]) -> str:
    """Build the U0 results table from ``results/u0_metrics.json``."""
    ds, noise, spec, spikes, cfg = m["dataset"], m["noise"], m["spectrum"], m["spikes"], m["config"]
    low, high = cfg["spike_band_hz"]
    band = f"{low:g} Hz to {high / 1000:g} kHz"

    def iqr(q: dict[str, float], fmt: str = "{:.2f}") -> str:
        return f"{fmt.format(q['median'])} (IQR {fmt.format(q['p25'])} to {fmt.format(q['p75'])})"

    rows = [
        (
            "Clips analysed",
            f"{ds['n_clean_clips']} of {ds['n_clips']}",
            f"{ds['n_saturated_clips_excluded']} excluded: more than "
            f"{cfg['max_saturation_fraction']:.1%} of samples on an ADC rail",
        ),
        ("Recording time analysed", f"{ds['clean_duration_s']:,.0f} s", f"{ds['fs_hz']:,.0f} Hz"),
        ("Spike-band noise σ", f"{iqr(noise['sigma'])} LSB", f"MAD estimate per clip, {band}"),
        (
            "Noise σ ÷ ideal quantisation noise",
            f"{noise['sigma_over_quantisation_rms']:.1f}×",
            "quantisation noise of a 1 LSB step is 1/√12 LSB rms",
        ),
        (
            f"Share of power below {cfg['lfp_cutoff_hz']:g} Hz",
            iqr(spec["lfp_power_fraction"], "{:.1%}"),
            f"LFP band power ÷ (LFP + spike band power), from {cfg['lfp_floor_hz']:g} Hz",
        ),
        (
            "PSD slope",
            f"f^{spec['psd_slope_lfp_band']:.2f} and f^{spec['psd_slope_spike_band']:.2f}",
            f"10 to {cfg['lfp_cutoff_hz']:g} Hz, and {low:g} Hz to 5 kHz; median spectrum",
        ),
        (
            f"Spikes detected at −{cfg['threshold_sigma']:g}σ",
            f"{spikes['n_spikes']:,}",
            f"{spikes['rate_hz_overall']:.2f} /s pooled; median clip "
            f"{spikes['rate_hz_per_clip']['median']:.2f} /s; "
            f"{spikes['clips_without_spikes']} clips have none",
        ),
        (
            "Detections expected from Gaussian noise alone",
            f"{spikes['gaussian_noise_crossing_rate_hz']:.3f} /s",
            "false-alarm floor of the detector at this threshold",
        ),
        (
            "Median spike trough depth",
            f"{iqr(spikes['amplitude'], '{:.1f}')} LSB",
            f"= {spikes['snr']['median']:.2f} σ; measured on the filtered spike band",
        ),
        (
            "Events whose larger phase is negative",
            f"{spikes['detected_polarity_fraction']:.1%}",
            "two opposite-sign crossings within 1 ms count as one event",
        ),
    ]
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in rows]
    return "\n".join(lines)


def u1_table(m: dict[str, Any]) -> str:
    """Build the U1 results table from ``results/u1_metrics.json``."""
    amp, cpl, emu = m["amplifier"], m["coupling"], m["emulator"]
    base = m["electrodes"]["in_saline"]
    noise, ideal = m["noise"]["in_saline"], m["noise"]["ideal_capacitor"]
    low, high = m["noise_band_hz"]
    band = f"{low:g} Hz to {high / 1000:g} kHz"
    fast, slow = m["settling"]["runs"]

    def total(method: str) -> str:
        budget = noise[method]
        return "not run" if budget is None else f"{budget['total_uv']:.3f}"

    spice_note = (
        f"ngspice impedance within {base['spice_max_relative_error']:.2%} of the analytic model"
        if base["spice_max_relative_error"] is not None
        else "ngspice not run"
    )
    rows = [
        (
            "Electrode impedance at 1 kHz",
            f"{base['z_1khz_ohm'] / 1e3:.0f} kΩ at −{abs(base['phase_1khz_deg']):.0f}°",
            f"platinum Utah electrode in saline (Straka et al., 2018); {spice_note}",
        ),
        (
            "Front-end noise: hand / analytic / ngspice",
            f"{total('hand')} / {total('analytic')} / {total('spice')} µV rms",
            f"referred to the tissue, {band}, at {amp['temperature_k'] - 273.15:.0f} °C",
        ),
        (
            "Largest disagreement between methods",
            f"{max(e['max_disagreement'] for e in m['noise'].values()):.2%}",
            "across three electrode models; the done-criterion is 10 %",
        ),
        (
            "Electrode / amplifier share",
            f"{noise['analytic']['electrode_uv']:.2f} / {noise['analytic']['lna_uv']:.2f} µV rms",
            "analytic model; the two add in quadrature",
        ),
        (
            "Same electrode with an ideal-capacitor double layer",
            f"{ideal['analytic']['electrode_uv']:.2f} µV electrode, "
            f"{ideal['analytic']['total_uv']:.2f} µV total",
            "the textbook Randles circuit, matched to the same impedance magnitude at 1 kHz",
        ),
        (
            "Amplifier",
            f"gain {amp['gain_1khz']:.1f}, {amp['f_low_hz']:.2f} Hz to "
            f"{amp['f_high_hz'] / 1e3:.2f} kHz",
            f"{amp['i_total_ua']:g} µA at {amp['v_supply_v']:g} V = {amp['power_uw']:g} µW; "
            f"NEF {amp['nef_measured']:.2f}",
        ),
        (
            "Electrode offset tolerated",
            f"AC: blocked; DC at the same gain: {cpl['dc_offset_tolerance_same_gain_mv']:.1f} mV",
            f"surviving {cpl['worst_offset_mv']:g} mV DC-coupled needs gain "
            f"{cpl['dc_gain_for_worst_offset']:.2g} and {cpl['adc_bits_dc']:.1f} ADC bits, "
            f"against {cpl['adc_bits_ac']:.1f}",
        ),
        (
            f"Time on the rail after a {m['settling']['step_mv']:g} mV offset step",
            f"{fast['saturation_time_simulated_s']:.2f} s at {fast['f_low_hz']:g} Hz; "
            f"{slow['saturation_time_simulated_s']:.2f} s at {slow['f_low_hz']:g} Hz",
            f"emulated; the formula gives {fast['saturation_time_analytic_s']:.3f} s and "
            f"{slow['saturation_time_analytic_s']:.3f} s",
        ),
        (
            "Emulator noise against the model",
            f"{emu['band_output_rms_measured_uv']:.1f} against "
            f"{emu['band_output_rms_predicted_uv']:.1f} µV rms",
            f"at the amplifier output, {band}, {emu['seconds']:g} s of emulated silence",
        ),
    ]
    nason = m["nason"]
    if nason is None:
        rows.append(("Nason in-vivo channel", "not run", "needs the Nason data (`make data`)"))
    else:
        rows += [
            (
                "Noise σ of a real in-vivo channel",
                f"{nason['sigma_recorded_uv']:.2f} µV",
                f"{nason['recorded_over_model']:.1f} times the model's "
                f"{nason['sigma_model_floor_uv']:.2f} µV through the same filter "
                "(Nason et al., one channel, 10 s)",
            ),
            (
                "That channel after the emulated front end",
                f"{nason['sigma_emulated_uv']:.2f} µV "
                f"(predicted {nason['sigma_predicted_uv']:.2f})",
                f"up from {nason['sigma_band_limited_uv']:.2f} µV with the emulator's noise off",
            ),
            (
                "The same at a quarter of the bias current",
                f"{nason['sigma_quarter_bias_uv']:.2f} µV",
                f"amplifier power {nason['quarter_bias_power_uw']:g} µW instead of "
                f"{amp['power_uw']:g} µW",
            ),
        ]
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in rows]
    return "\n".join(lines)


def u2_table(m: dict[str, Any]) -> str:
    """Build the U2 results table from ``results/u2_metrics.json``."""
    cfg = m["config"]
    ref = f"{cfg['reference_noise_uv']:g}"
    rows: list[tuple[str, str, str]] = []
    mearec = m["mearec"]
    if mearec is None:
        rows.append(("MEArec sweeps", "not run", "needs the recording (`make mearec`)"))
    else:
        rec, det, fixed = mearec["recording"], mearec["detection"], mearec["fixed_point"]
        track = mearec["tracking"]

        def f1(sweep: str, method: str, value: str) -> str:
            return f"{det[sweep][method][value]['f1']:.2f}"

        def ci(method: str) -> str:
            row = det["noise"][method][ref]
            return f"{row['f1']:.2f} ({row['f1_low']:.2f} to {row['f1_high']:.2f})"

        rows += [
            (
                "Ground-truth recording",
                f"{rec['n_channels']} channels, {rec['n_units']} units, {rec['duration_s']:g} s",
                f"MEArec simulation; thresholds set on the first {cfg['calibration_s']:g} s, "
                f"{rec['target_spikes']:,} target spikes scored on the rest",
            ),
            (
                f"Detection F1 at in-vivo noise ({ref} µV)",
                f"threshold {ci('threshold')}; NEO {ci('neo')}",
                "causal, 30 kS/s, 10 bits; pooled over channels with a 95 % bootstrap interval",
            ),
            (
                "The same with a zero-phase filter",
                ci("offline"),
                "what an offline analysis reports and an implant cannot run",
            ),
            (
                "Threshold F1 against background noise",
                " / ".join(f1("noise", "threshold", f"{n:g}") for n in cfg["noise_levels_uv"]),
                "at " + ", ".join(f"{n:g}" for n in cfg["noise_levels_uv"]) + " µV rms",
            ),
            (
                "Threshold F1 against ADC bits",
                " / ".join(f1("bits", "threshold", str(b)) for b in (12, 10, 8, 6, 5)),
                "at 12, 10, 8, 6, 5 bits over ±1 mV (steps of 0.5, 2, 8, 31 and 62 µV)",
            ),
            (
                "Threshold F1 against sample rate",
                " / ".join(f1("fs", "threshold", str(f)) for f in (30000, 15000, 10000, 5000)),
                "at 30, 15, 10, 5 kS/s with no anti-alias filter beyond the amplifier",
            ),
            (
                "Tracking the largest unit on a channel",
                f"crossings {track['crossings'][ref]['mean']:.2f}; SBP "
                f"{track['sbp_implant'][ref]['mean']:.2f}",
                f"correlation with its firing at {ref} µV; crossings at 30 kS/s, SBP from a "
                "300 Hz to 1 kHz amplifier sampled at 2 kS/s",
            ),
            (
                "Fixed point against floating point",
                f"threshold F1 {fixed['fixed_threshold_f1']:.3f} against "
                f"{det['noise']['threshold'][ref]['f1']:.3f}; SBP r = "
                f"{fixed['sbp_fixed_vs_float_r']:.3f}",
                f"integer-only paths; accumulators of {fixed['acc_bits_wide']:.0f} bits at "
                f"30 kS/s and {fixed['acc_bits_narrow']:.0f} bits at 2 kS/s",
            ),
            (
                "Offset from truncating inside the 30 kS/s filter",
                f"{abs(fixed['offset_wide_trunc_guard0']):.0f} codes",
                f"against {abs(fixed['offset_wide_round_guard6']):.3f} with rounding and six "
                "guard bits",
            ),
        ]
    nason = m["nason"]
    if nason is None:
        rows.append(("Nason reproduction", "not run", "needs the Nason data (`make data`)"))
    else:
        sim = nason["simulation"]["arrangements"][0]
        vivo, online = nason["in_vivo"], nason["online"]
        rows += [
            (
                "SBP recomputed from the paper's broadband",
                f"largest error {sim['sbp_reproduction_max_error']:.0e}",
                "against the SBP stored with the Nason simulation; identical to rounding",
            ),
            (
                "Correlation with the largest unit, in vivo",
                f"SBP {vivo['sbp_r'][0]:.2f}; crossings {vivo['crossings_r'][0]:.2f}",
                f"one channel, three sorted units; the largest is "
                f"{vivo['unit_amplitude_uv'][0]:.0f} µV peak to peak",
            ),
            (
                "Finger velocity decoded from 96 channels",
                f"SBP R² {online['velocity_sbp']['r2']:.2f} ± "
                f"{online['velocity_sbp']['r2_sem']:.2f}; "
                f"crossings {online['velocity_crossings']['r2']:.2f} ± "
                f"{online['velocity_crossings']['r2_sem']:.2f}; both "
                f"{online['velocity_both']['r2']:.2f}",
                f"same ridge readout, {online['n_folds']}-fold cross-validation over blocks of "
                "trials, ± fold SEM",
            ),
        ]
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in rows]
    return "\n".join(lines)


def u3_table(m: dict[str, Any]) -> str:
    """Build the U3 results table from ``results/u3_metrics.json``."""
    rows, cfg, check = m["configurations"], m["config"], m["bridge_check"]
    power = cfg["power"]
    names = {
        "nason": "Nason, finger velocity",
        "nomad_session1": "NoMAD session 1, cursor velocity",
        "nomad_session2": "NoMAD session 2, cursor velocity",
    }

    def describe(row: dict[str, Any]) -> str:
        feature = {"crossings": "crossings", "sbp": "SBP", "both": "both", "raw": "raw"}
        return (
            f"{feature[row['feature']]}, {row['i_total_ua']:g} µA, {row['bits']} b, "
            f"{row['fs_hz'] / 1e3:g} kS/s"
        )

    out: list[tuple[str, str, str]] = [
        (
            "Design space",
            f"{len(rows)} configurations, {len(m['infeasible'])} more infeasible",
            "feature x amplifier current x ADC bits x sample rate; infeasible means the "
            "wideband amplifier cannot reach 7.5 kHz at that current",
        ),
        (
            "Power model constants",
            f"{power['v_supply_v']:g} V, NEF {power['nef']:g}, {power['adc_fom_j'] * 1e15:g} "
            f"fJ/step, {power['energy_per_multiply_j'] * 1e12:g} pJ/multiply, "
            f"{power['energy_per_bit_j'] * 1e12:g} pJ/bit",
            "assumptions, with sources in docs/U3_design.md",
        ),
    ]
    for name, info in m["datasets"].items():
        base, best = rows[info["baseline"]], rows[info["winner"]]
        top = rows[info["most_accurate"]]
        title = names.get(name, name)
        out += [
            (
                f"{title}: conventional chain",
                f"R² {info['baseline_r2']:.3f} at {base['total_uw']:.2f} µW",
                describe(base),
            ),
            (
                f"{title}: cheapest within {cfg['tolerance']:.0%}",
                f"R² {best['r2'][name]['mean']:.3f} at {best['total_uw']:.2f} µW "
                f"({base['total_uw'] / best['total_uw']:.1f} times less)",
                describe(best),
            ),
        ]
        if top["label"] != base["label"]:
            out.append(
                (
                    f"{title}: most accurate",
                    f"R² {top['r2'][name]['mean']:.3f} at {top['total_uw']:.2f} µW",
                    describe(top),
                )
            )
    raw = next((row for row in rows if row["feature"] == "raw"), None)
    first = m["datasets"][next(iter(m["datasets"]))]
    feature_bps = rows[first["winner"]]["bit_rate_bps"]
    if raw is not None:
        out.append(
            (
                "Streaming every sample instead",
                f"{raw['total_uw']:.2f} µW, of which {raw['link_uw']:.2f} µW is radio",
                f"{raw['bit_rate_bps'] / 1e3:g} kbit/s per channel, against {feature_bps:g} bit/s "
                "for the cheapest match",
            )
        )
    cross = m["cross_session"]
    if cross is not None:
        out.append(
            (
                "Fixed decoder across the two NoMAD sessions",
                f"R² {cross['train_early_test_late']:.2f} and {cross['train_late_test_early']:.2f}",
                f"trained on one, tested on the other, {cross['days_apart']:.0f} days apart, "
                "no recalibration",
            )
        )
    clip = m.get("input_range")
    if clip is not None:
        out.append(
            (
                "Samples of a real channel outside the ±1 mV ADC range",
                f"{clip['wideband_clipped']:.1%} wideband; {clip['narrowband_clipped']:.2%} "
                "through the 300 Hz to 1 kHz amplifier",
                f"Nason in-vivo broadband, RMS {clip['broadband_rms_uv']:.0f} µV; the "
                "wideband chain would need more range, hence more bits",
            )
        )
    out.append(
        (
            f"Bridge check at {check['low_current_ua']:g} µA",
            f"crossings r = {check['crossings_r_actual']:.2f} measured, "
            f"{check['crossings_r_predicted']:.2f} predicted; SBP "
            f"{check['sbp_r_actual']:.3f} and {check['sbp_r_predicted']:.3f}",
            "on MEArec, where the cheaper chain can be simulated directly; crossings against "
            "true spike counts, SBP against the reference chain's",
        )
    )
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in out]
    return "\n".join(lines)


def u4_table(m: dict[str, Any]) -> str:
    """Build the U4 results table from ``results/u4_metrics.json``."""
    names = {
        "nason": "Nason",
        "nomad_session1": "NoMAD session 1",
        "nomad_session2": "NoMAD session 2",
    }
    decoders = {
        "ridge": "U3's ridge readout",
        "wiener": "Wiener filter",
        "kalman": "Kalman filter",
        "gru": "GRU",
    }
    rows = {(r["dataset"], r["label"], r["decoder"]): r for r in m["results"]}

    def score(row: dict[str, Any]) -> str:
        return f"{row['r2']:.3f} [{row['r2_low']:.3f}, {row['r2_high']:.3f}]"

    out: list[tuple[str, str, str]] = []
    for name, head in m["headline"].items():
        base_uw = rows[name, head["conventional"], "ridge"]["total_uw"]
        best_uw = rows[name, head["winner"], "ridge"]["total_uw"]
        for decoder, entry in head["decoders"].items():
            diff = entry["winner_minus_conventional"]
            out.append(
                (
                    f"{names.get(name, name)}, {decoders[decoder]}",
                    f"R² {score(rows[name, head['conventional'], decoder])} at {base_uw:.2f} µW; "
                    f"{score(rows[name, head['winner'], decoder])} at {best_uw:.2f} µW",
                    f"conventional chain, then U3's choice ({head['winner']}); paired difference "
                    f"{diff['difference']:+.3f} [{diff['low']:+.3f}, {diff['high']:+.3f}]",
                )
            )
    first = next(iter(m["headline"]))
    cheapest = min((r for r in m["results"] if r["dataset"] == first), key=lambda r: r["total_uw"])
    for decoder in head["decoders"]:
        row = rows[first, cheapest["label"], decoder]
        out.append(
            (
                f"{names.get(first, first)}, {decoders[decoder]}, cheapest chain decoded",
                f"R² {score(row)} at {row['total_uw']:.2f} µW",
                cheapest["label"],
            )
        )
    cross = m["cross_session"]
    if cross:
        for r in cross["rows"]:
            out.append(
                (
                    f"Across sessions, {decoders[r['decoder']]}",
                    f"R² {score(r)}",
                    f"fitted on {names.get(r['train'], r['train'])}, tested on "
                    f"{names.get(r['test'], r['test'])}, {cross['days_apart']:.0f} days apart, "
                    "no recalibration",
                )
            )
    for decoder, cost in m["decoder_cost"][first].items():
        out.append(
            (
                f"Compute, {decoders[decoder]}",
                f"{cost['multiplies_per_bin']} multiplies per bin, "
                f"{cost['uw_per_channel']:.4f} µW per channel",
                f"{names.get(first, first)}, 96 channels, 50 ms bins, 1 pJ per multiply, if run on "
                "the implant",
            )
        )
    cfg = m["config"]
    out.append(
        (
            "Evaluation",
            f"{cfg['n_folds']} folds, {cfg['n_boot']} bootstrap resamples, seed {cfg['seed']}",
            "folds are contiguous blocks of trials; intervals are 95 %, from resampling trials "
            "(NoMAD: recording segments)",
        )
    )
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in out]
    return "\n".join(lines)


def u5_table(m: dict[str, Any]) -> str:
    """Build the U5 results table from ``results/u5_metrics.json``."""
    key, sig, rec, prop = m["key"], m["signal"], m["recovery"]["chains"], m["propagation"]
    index = sig["amplitudes_mv"].index(key["example_mv"])
    at = f"{key['example_mv']:g} mV artifact"
    chains = {"sbp": "0.48 µW chain", "crossings": "conventional chain"}
    treatments = {"none": "no treatment", "blank": "blanking", "template": "template canceller"}
    out: list[tuple[str, str, str]] = []
    for s in prop["scenarios"]:
        out.append(
            (
                f"Artifact, stimulating electrode {s['name']}"
                if s["distance_mm"]
                else "Artifact on the stimulating array",
                f"{s['median_mv']:.1f} mV median ({s['min_mv']:.1f} to {s['max_mv']:.1f})",
                f"{prop['current_ua']:.0f} µA monopolar pulse, {prop['charge_per_phase_nc']:.0f} "
                f"nC per phase, tissue {prop['sigma_s_per_m']:g} S/m (assumed); "
                f"{s['share_above_adc']:.0%} of channels past the ADC's "
                f"±{prop['adc_range_mv']:.0f} mV, {s['share_above_amplifier']:.0%} past the "
                f"amplifier's ±{prop['amplifier_range_mv']:.0f} mV",
            )
        )
    for name, chain in rec.items():
        spice = chain["spice_ms"][index]
        electrode = chain["spice_electrode_ms"][index]
        checked = (
            "ngspice not run"
            if spice is None
            else f"ngspice {spice:.2f} ms, {electrode:.2f} ms with the electrode in series"
        )
        out.append(
            (
                f"Recovery time, {chains[name]}",
                f"{chain['model_ms'][index]:.2f} ms ({checked})",
                f"{at}; from the end of the pulse until the output stays within "
                f"±{m['recovery']['threshold_uv']:.0f} µV at the input; high-pass corner "
                f"{chain['f_low_hz']:.0f} Hz",
            )
        )
    gaps = [c["largest_model_spice_gap_us"] for c in rec.values()]
    if None not in gaps:
        out.append(
            (
                "Python model against ngspice",
                f"recovery times within {max(gaps):.0f} µs",
                f"both chains, {len(sig['amplitudes_mv'])} amplitudes from "
                f"{sig['amplitudes_mv'][0]:g} to {sig['amplitudes_mv'][-1]:g} mV, ideal source",
            )
        )
    for treatment, label in treatments.items():
        sbp = sig["sbp"][treatment]
        out.append(
            (
                f"SBP while stimulating, {label}",
                f"shift {sbp['bias'][index]:+.2f} noise floors, {sbp['valid'][index]:.0%} of "
                "samples kept",
                f"{at}, {sig['realised_rate_hz']:.0f} pulses per second, MEArec, "
                f"{sig['n_channels']} channels; shift is the mean change of a stimulated 50 ms "
                "bin",
            )
        )
    real = m["in_vivo"]
    if real:
        parts = ", ".join(
            f"{treatments[t]} {v['bias']:+.2f}" for t, v in real["treatments"].items()
        )
        out.append(
            (
                "Same on a real recording",
                parts,
                f"SBP shift in noise floors, Nason in-vivo channel, {at}, "
                f"{real['stimulated_bins']} stimulated bins",
            )
        )
    decode = m["decoding"]
    if decode:
        scenario, decoder = key["scenario"], key["decoder"]
        names = {"gru": "GRU", "wiener": "Wiener filter"}
        rows = {
            (r["chain"], r["scenario"], r["treatment"], r["decoder"], r["fitted"]): r
            for r in decode["rows"]
        }
        others = [d for d in decode["no_stimulation_r2"]["sbp"] if d != decoder]
        others = [d for d in others if not d.endswith("_all_bins")]
        elsewhere = list(dict.fromkeys(r["scenario"] for r in decode["rows"]))[1:]

        def change(row: dict[str, Any]) -> str:
            return f"{row['loss']:+.3f} [{row['loss_low']:+.3f}, {row['loss_high']:+.3f}]"

        for name, label in chains.items():
            reference = decode["no_stimulation_r2"][name]
            out.append(
                (
                    f"Decoding, {label}, no stimulation",
                    ", ".join(
                        f"R² {reference[d]:.3f} ({names.get(d, d)})" for d in (decoder, *others)
                    ),
                    "Nason, finger velocity; the stimulated bins of the held-out folds, before "
                    "any artifact is added",
                )
            )
            for treatment, treatment_label in treatments.items():
                plain = rows[name, scenario, treatment, decoder, "without stimulation"]
                refit = rows[name, scenario, treatment, decoder, "with stimulation"]
                out.append(
                    (
                        f"Decoding, {label}, {treatment_label}",
                        f"R² {plain['r2']:.3f}, change {change(plain)}; refitted R² "
                        f"{refit['r2']:.3f}, change {change(refit)}",
                        f"{names.get(decoder, decoder)}, stimulating electrode on the {scenario}; "
                        "decoder fitted without stimulation, then refitted with it on; change "
                        "is paired, against no stimulation",
                    )
                )
            for other in others:
                parts = []
                for treatment in ("blank", "template"):
                    plain = rows[name, scenario, treatment, other, "without stimulation"]
                    refit = rows[name, scenario, treatment, other, "with stimulation"]
                    parts.append(
                        f"{treatments[treatment]} {plain['loss']:+.3f}, refitted "
                        f"{refit['loss']:+.3f}"
                    )
                out.append(
                    (
                        f"Decoding, {label}, {names.get(other, other)}",
                        "; ".join(parts),
                        f"change in R², {scenario}",
                    )
                )
            parts = []
            for place in elsewhere:
                blank = rows[name, place, "blank", decoder, "without stimulation"]
                template = rows[name, place, "template", decoder, "without stimulation"]
                parts.append(
                    f"{place}: blanking {blank['loss']:+.3f}, template canceller "
                    f"{template['loss']:+.3f}"
                )
            if parts:
                out.append(
                    (
                        f"Decoding, {label}, other geometries",
                        "; ".join(parts),
                        f"change in R², {names.get(decoder, decoder)} fitted without stimulation",
                    )
                )
    cost = m["cost"]["chains"]
    for name, label in chains.items():
        c = cost[name]
        out.append(
            (
                f"Canceller cost, {label}",
                f"{c['canceller_uw']:.3f} µW per channel ({c['share_of_chain']:.1%} of the "
                f"chain), {c['template_bits_per_channel']} bits of memory",
                f"{m['cost']['adds_per_sample']} additions per sample and no multiplier, each "
                "charged as a 1 pJ multiply; an upper bound until U7",
            )
        )
    fixed = m["fixed_point"]
    out.append(
        (
            "Integer canceller against floating point",
            f"largest difference {fixed['canceller_max_difference_lsb']:.2f} LSB; SBP "
            f"correlation {fixed['sbp_correlation']:.4f}",
            f"one MEArec channel, {fixed['seconds']:.0f} s, {fixed['template_bits']}-bit "
            "template; the integer version is the reference for U7",
        )
    )
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in out]
    return "\n".join(lines)


def u6_table(m: dict[str, Any]) -> str:
    """Build the U6 results table from ``results/u6_metrics.json``."""
    key, model, decode = m["key"], m["model"], m["decoding"]
    week, last_week = key["week"], key["last_week"]
    first, last = model["states"][0], model["states"][-1]
    fine = model["fine_weeks"]
    at = min(range(len(fine)), key=lambda i: abs(fine[i] - week))
    amplitude = model["amplitude"]
    chains = {"sbp": "0.48 µW chain", "crossings": "conventional chain"}
    decoders = {"ridge": "ridge", "wiener": "Wiener", "kalman": "Kalman", "gru": "GRU"}
    datasets = {"nason": "Nason", "nomad_session2": "NoMAD session 2"}

    def pm(row: dict[str, Any], name: str) -> str:
        return f"{row[name]:+.3f} [{row[name + '_low']:+.3f}, {row[name + '_high']:+.3f}]"

    out: list[tuple[str, str, str]] = [
        (
            "Electrode after encapsulation",
            f"{first['z_1khz_kohm']:.0f} to {last['z_1khz_kohm']:.0f} kΩ at 1 kHz; front-end "
            f"noise {first['frontend_noise_uv']:.1f} to {last['frontend_noise_uv']:.1f} µV",
            "Randles electrode plus a series sheath solved for a "
            f"{m['config']['aging']['impedance_factor_final']:g}-fold rise (Barrese 2013; "
            "assumption); noise over 300 Hz to 7.5 kHz referred to the tissue signal, "
            "conventional chain",
        ),
        (
            f"Spike amplitude at week {week:g}",
            f"median {amplitude['p50'][at]:.2f} of week 0 ({amplitude['p10'][at]:.2f} to "
            f"{amplitude['p90'][at]:.2f})",
            f"{m['config']['aging']['amplitude_decline_per_month']:.1%} lost per month on "
            "average (Chestek 2011; assumption), each channel at its own rate; 10th to 90th "
            "percentile of channels",
        ),
    ]
    settled = m["response"]["states"][m["response"]["settled"]]
    grid = m["response"]["amplitude_grid"]
    out.append(
        (
            "Crossings kept by the settled electrode",
            ", ".join(
                f"{settled['p_keep'][grid.index(a)]:.0%} at amplitude {a:g}"
                for a in m["config"]["check_amplitudes"]
            ),
            f"MEArec, {m['response']['n_channels']} channels, 13 µV background, thresholds "
            "recalibrated; share of the week-0 chain's detections of real spikes",
        )
    )
    check = m["bridge_check"]["amplitudes"]
    half = check[min(check, key=float)]
    out.append(
        (
            f"Bridge check at amplitude {min(check, key=float)}",
            f"SBP {half['sbp_actual']['r']:.3f} simulated, {half['sbp_predicted']['r']:.3f} "
            f"bridged; crossings {half['crossings_actual']['r']:.3f} simulated, "
            f"{half['crossings_predicted']['r']:.3f} bridged",
            "correlation of the 50 ms feature with the true spike count, mean over MEArec "
            "channels: the aged feature simulated directly against the bridge's prediction "
            "from the week-0 feature",
        )
    )
    for condition, info in decode["conditions"].items():
        for name in ("wiener", "gru"):
            if name not in key["curves"][condition]:
                continue
            curve = key["curves"][condition][name]
            fixed, zscore = curve["key_fixed"], curve["key_zscore"]
            value = (
                f"week 0: {curve['week0']:.3f}; week {week:g}: {fixed['r2']:.3f} left alone, "
                f"{zscore['r2']:.3f} with running z-score ({pm(zscore, 'gain')})"
            )
            if "key_refit" in curve:
                value += f", {curve['key_refit']['r2']:.3f} refitted"
            value += f"; week {last_week:g}: {curve['last_fixed']['r2']:.3f} left alone, "
            value += f"{curve['last_zscore']['r2']:.3f} with running z-score"
            out.append(
                (
                    f"R², {chains[info['chain']]}, {datasets.get(info['dataset'], info['dataset'])}"
                    f", {decoders[name]}",
                    value,
                    f"{info['chain_uw']:.2f} µW per channel; decoder fitted at week 0, tested on "
                    "held-out blocks aged by the model; paired 95 % interval against left alone",
                )
            )
    cross = m.get("cross_session")
    if cross:
        rows = {(r["decoder"], r["mode"]): r for r in cross["rows"]}
        for name in decoders:
            if (name, "fixed") not in rows:
                continue
            aligned = rows[name, "aligned"]
            out.append(
                (
                    f"Real session pair, {decoders[name]}",
                    f"{rows[name, 'fixed']['r2']:.3f} left alone, "
                    f"{rows[name, 'zscore']['r2']:.3f} with running z-score, "
                    f"{aligned['r2']:.3f} with {cross['n_latents']} aligned latents "
                    f"({pm(aligned, 'gain')}), {rows[name, 'within']['r2']:.3f} refitted",
                    f"NoMAD, fitted on {cross['train'].replace('nomad_', '')}, tested on "
                    f"{cross['test'].replace('nomad_', '')} {cross['days_apart']:.1f} days later; "
                    f"alignment from the first {cross['calibration_s']:.0f} s without labels; "
                    "refitted: held-out blocks within the later session",
                )
            )
    stim = m.get("stimulation")
    if stim:
        rows = {(r["decoder"], r["mode"]): r for r in stim["rows"]}
        for name in decoders:
            if (name, "fixed") not in rows:
                continue
            out.append(
                (
                    f"Stimulation shift, {decoders[name]}",
                    f"loss {rows[name, 'fixed']['loss']:+.3f} left alone, "
                    f"{rows[name, 'zscore']['loss']:+.3f} with running z-score, "
                    f"{pm(rows[name, 'gated zscore'], 'loss')} gated by the stimulator, "
                    f"{rows[name, 'refit']['loss']:+.3f} refitted",
                    f"U5's test: 0.48 µW chain, stimulating electrode on the {stim['scenario']}, "
                    f"{stim['treatment']} canceller, decoder fitted before stimulation; R² on "
                    "stimulated bins against the same decoder with no stimulation",
                )
            )
    cheap = next((c for c, i in decode["conditions"].items() if i["chain"] == "sbp"), None)
    rival = None
    if cheap is not None:
        rival = next(
            (
                c
                for c, i in decode["conditions"].items()
                if i["chain"] == "crossings"
                and i["dataset"] == decode["conditions"][cheap]["dataset"]
            ),
            None,
        )
    if m["variations"] and cheap is not None and rival is not None:
        best = max(key["curves"][cheap], key=lambda d: key["curves"][cheap][d]["week0"])
        table = {
            (r["variation"], r["condition"], r["mode"]): r["r2"]
            for r in decode["rows"]
            if r["decoder"] == best and r["week"] == week
        }
        names = ["baseline", *m["variations"]]
        gains = {
            (v, c): table[v, c, "zscore"] - table[v, c, "fixed"]
            for v in names
            for c in (cheap, rival)
        }
        ahead = sum(table[v, cheap, "zscore"] >= table[v, rival, "zscore"] for v in names)
        worst = min(names, key=lambda v: table[v, cheap, "zscore"])
        out.append(
            (
                "Changed assumptions",
                f"stabiliser gain {min(gains.values()):+.3f} to {max(gains.values()):+.3f}; "
                f"0.48 µW chain at least as good as the conventional chain in {ahead} of "
                f"{len(names)}; worst case R² {table[worst, cheap, 'zscore']:.3f} "
                f"({worst})",
                f"{len(names) - 1} assumptions changed one at a time; {decoders[best]} on "
                f"{datasets.get(decode['conditions'][cheap]['dataset'], 'the dataset')} at week "
                f"{week:g}; both chains compared with the running z-score",
            )
        )
    cost = m["cost"]
    zscore_cost = cost["modes"]["zscore"]
    out.append(
        (
            "Stabiliser cost",
            f"{zscore_cost['uw_per_channel']:.5f} µW per channel "
            f"({cost['chains']['sbp']['zscore_share_of_chain']:.2%} of the 0.48 µW chain), "
            f"{zscore_cost['register_bits_per_channel']} register bits; latent projection "
            f"{cost['modes']['aligned']['uw_per_channel']:.5f} µW",
            f"{zscore_cost['operations_per_bin']} add or shift steps per 50 ms bin and no "
            "multiplier, each charged as a 1 pJ multiply (an upper bound); the alignment's "
            "axes are found outside the implant",
        )
    )
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in out]
    return "\n".join(lines)


def u7_table(m: dict[str, Any]) -> str:
    """Build the U7 results table from ``results/u7_metrics.json``."""
    stim, real, worst, gate = m["stimulus"], m["bit_exact"], m["worst_case"], m["gate_level"]
    synth, power, key, tools = m["synthesis"], m["power"], m["key"], m["tools"]
    widths = {w["node"]: w for w in m["widths"]}
    rows = {r["n_channels"]: r for r in synth["sweep"] if "error" not in r}
    own = rows[key["n_channels"]]
    if stim.get("source") == "neuralink":
        data = (
            f"{real['n_channels']} channels x {real['seconds']:.0f} s of the Neuralink clips "
            f"({stim['n_clips']} clips; band-passed {stim['band_hz'][0]:g} Hz to "
            f"{stim['band_hz'][1] / 1000:g} kHz, sampled at 2 kS/s, 10-bit codes divided by 4 to "
            f"8 bits; no voltage scale), U5's stimulation artifacts added at "
            f"{stim['pulse_rate_hz']:.0f} pulses per second (seed {stim['seed']})"
        )
    else:
        data = (
            f"{real['n_channels']} channels x {real['seconds']:.0f} s of {stim.get('source')} codes"
        )
    order = ("canceller output", "filter accumulator", "filter output", "bin sum")
    out: list[tuple[str, str, str]] = [
        (
            "RTL against Python",
            f"{real['total_mismatches']} mismatches in {real['values_compared']:,} values",
            f"{data}; canceller output, blanking flag and filter output per sample, sum and "
            f"count per 50 ms bin; Verilator {tools.get('verilator', '?')} against "
            "`neurochain.rtl.spec.ChainSpec.reference`",
        ),
        (
            "Worst-case stimulus",
            f"{worst['total_mismatches']} mismatches in {worst['values_compared']:,} values; "
            f"filter output reaches {widths['filter output']['peak_worst_case']} of a bound of "
            f"{widths['filter output']['bound']}",
            "codes built to drive every register to the largest value it can hold",
        ),
        (
            "Gate-level netlist against RTL",
            f"{gate['mismatches_against_rtl']} mismatches in {gate['values_compared']:,} values",
            f"Yosys NAND/NOR netlist ({gate['gates']['gates']} gates, "
            f"{gate['gates']['flip_flops']} flip-flops) simulated on the first "
            f"{gate['seconds']:.0f} s of the same data",
        ),
        (
            "Word widths",
            ", ".join(f"{name} {widths[name]['bits']}" for name in order)
            + f" bits; {m['state_bits']['total']} state bits per channel",
            "worst-case bounds derived from the parameters; this data used "
            + ", ".join(str(widths[name]["bits_used_real"]) for name in order),
        ),
        (
            f"ECP5 resources, {key['n_channels']} channels",
            f"{own['lut4']} LUT4, {own['flip_flops']} flip-flops, {own['block_rams']} of "
            f"{own['block_rams_available']} block RAMs, {own['multipliers']} hard multipliers",
            f"Yosys {tools.get('yosys', '?')} (`synth_ecp5 -nodsp`) and nextpnr-ecp5 for a "
            f"Lattice {synth['device']}, {synth['package']}, speed grade {synth['speed_grade']}; "
            f"{own['logic_cells']} of {own['logic_cells_available']:,} logic cells; no hardware "
            "in the loop",
        ),
        (
            "Fastest clock (fmax)",
            f"{own['fmax_median_mhz']:.1f} MHz ({min(own['fmax_mhz']):.1f} to "
            f"{max(own['fmax_mhz']):.1f})",
            f"median and range of {len(own['seeds'])} placement seeds; {key['n_channels']} "
            f"channels need {own['clock_needed_hz'] / 1000:.0f} kHz; the slowest path is "
            f"{own['critical_path'].get('delay_ns', float('nan')):.1f} ns, of which the block "
            f"RAM's output delay is {own['critical_path'].get('clk_to_q_ns', float('nan')):.1f}",
        ),
    ]
    others = [n for n in rows if n != key["n_channels"]]
    if others:
        out.append(
            (
                "Other channel counts",
                "; ".join(
                    f"{n}: {rows[n]['lut4']} LUT4, {rows[n]['block_rams']} block RAMs"
                    for n in others
                ),
                "the same datapath shared by more or fewer channels; one channel alone has "
                "its state in flip-flops and LUT RAM",
            )
        )
    star = synth["variants"]
    if "error" not in star["star_lut"] and "error" not in star["star_dsp"]:
        out.append(
            (
                "Filter products written as `*`",
                f"{star['star_lut']['lut4']} LUT4, or {star['star_dsp']['lut4']} plus "
                f"{star['star_dsp']['multipliers']} hard multipliers",
                f"against {own['lut4']} with shifts and adds (canonical signed digits)",
            )
        )
    energy, measured, assumed = power["energy"], power["measured"], power["assumptions"]
    budget = power["budget"]
    out += [
        (
            "Digital energy per sample",
            f"{energy['dynamic_pj']:.2f} pJ: logic {energy['logic_pj']:.2f}, clock "
            f"{energy['clock_pj']:.2f}, state memory {energy['memory_pj']:.2f}",
            f"{measured['toggles_per_sample']:.0f} node toggles (gate-level simulation of the "
            f"real data) and {measured['memory_bits_per_sample']:.0f} memory bits per sample, "
            f"times assumed {assumed['toggle_fj']:.2f} fJ per toggle "
            f"({assumed['c_node_f'] * 1e15:g} fF at {assumed['v_supply_v']:g} V) and "
            f"{assumed['memory_j_per_bit'] * 1e15:g} fJ per bit; an ASIC estimate, not the "
            "FPGA's power",
        ),
        (
            "Digital power",
            f"{energy['total_uw']:.4f} µW per channel ({key['digital_low_uw']:.4f} to "
            f"{key['digital_high_uw']:.3f})",
            f"dynamic at 2 kS/s plus {energy['leakage_uw']:.4f} µW of leakage (assumed "
            f"{assumed['gate_leakage_w'] * 1e12:g} pW per gate and per bit); range: every "
            f"assumed constant a tenth to ten times; the placeholder of U3 and U5 was "
            f"{power['placeholder']['total_uw']:.3f}",
        ),
        (
            "Chain budget",
            f"{budget['before_uw']:.3f} to {budget['after_uw']:.3f} µW per channel "
            f"({budget['after_low_uw']:.3f} to {budget['after_high_uw']:.3f})",
            f"amplifier {budget['lna_uw']:.3f} and ADC {budget['adc_uw']:.3f} from U3's model; "
            f"digital from this estimate; radio word {power['radio']['bits_per_bin_rtl']} bits "
            f"per bin as built instead of {power['radio']['bits_per_bin_model']:.0f}; the "
            f"amplifier is {budget['lna_share']:.0%}",
        ),
    ]
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in out]
    return "\n".join(lines)


TABLES = {
    "U0_RESULTS": ("u0_metrics.json", u0_table),
    "U1_RESULTS": ("u1_metrics.json", u1_table),
    "U2_RESULTS": ("u2_metrics.json", u2_table),
    "U3_RESULTS": ("u3_metrics.json", u3_table),
    "U4_RESULTS": ("u4_metrics.json", u4_table),
    "U5_RESULTS": ("u5_metrics.json", u5_table),
    "U6_RESULTS": ("u6_metrics.json", u6_table),
    "U7_RESULTS": ("u7_metrics.json", u7_table),
}


def render(readme: str) -> str:
    """Return the README text with every marked table rebuilt."""
    for marker, (filename, build) in TABLES.items():
        begin, end = f"<!-- BEGIN {marker} -->", f"<!-- END {marker} -->"
        if begin not in readme or end not in readme:
            raise SystemExit(f"README.md is missing the {marker} markers")
        metrics = json.loads((results_dir() / filename).read_text())
        head, rest = readme.split(begin, 1)
        _, tail = rest.split(end, 1)
        readme = f"{head}{begin}\n{build(metrics)}\n{end}{tail}"
    return readme


def main() -> int:
    """Rewrite README.md, or with ``--check`` report whether it is current."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if README.md would change")
    args = parser.parse_args()
    path = repo_root() / "README.md"
    current = path.read_text()
    updated = render(current)
    if args.check:
        if updated != current:
            print("README.md is out of date; run `make readme`", file=sys.stderr)
            return 1
        return 0
    path.write_text(updated)
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
