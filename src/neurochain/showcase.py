"""Build the results site: one self-contained HTML page drawn from ``results/*.json``.

The page has three sources and nothing else:

* ``site/template.html``: the prose, with ``{{name}}`` placeholders wherever a
  result is quoted;
* ``site/style.css`` and ``site/app.js``: the look and the interactive charts;
* ``results/u*_metrics.json``: every number.

:func:`build_facts` formats the quoted numbers, :func:`build_payload` selects
the arrays the charts draw, :func:`build_tables` renders the static tables, and
:func:`build_site` joins them into ``build/site/index.html``. Nothing is
computed here that changes a result: values are read, rounded for display and
laid out. A placeholder with no fact fails the build, so the prose cannot quote
a number the results do not hold.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from neurochain.paths import repo_root, results_dir

REPO_URL = "https://github.com/danielkim136/neurochain"
"""Where the links to code, design notes and result files point."""

UNITS = range(8)
DECODERS = ("ridge", "wiener", "kalman", "gru")
DECODER_NAMES = {
    "ridge": "Ridge readout",
    "wiener": "Wiener filter",
    "kalman": "Kalman filter",
    "gru": "GRU",
}
DATASET_NAMES = {
    "nason": "Nason, finger velocity",
    "nomad_session1": "NoMAD session 1, cursor velocity",
    "nomad_session2": "NoMAD session 2, cursor velocity",
}
MINUS = "−"

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z0-9_]+)\s*\}\}")


def load_results(directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """Read ``u0_metrics.json`` to ``u7_metrics.json`` into a dict keyed ``"u0"`` to ``"u7"``."""
    directory = directory or results_dir()
    out: dict[str, dict[str, Any]] = {}
    for unit in UNITS:
        with (directory / f"u{unit}_metrics.json").open(encoding="utf-8") as handle:
            out[f"u{unit}"] = json.load(handle)
    return out


def _fixed(value: float, places: int = 2) -> str:
    """Format with a fixed number of decimals and a typographic minus sign."""
    return f"{value:.{places}f}".replace("-", MINUS)


def _signed(value: float, places: int = 3) -> str:
    """Format with an explicit sign, for differences."""
    return f"{value:+.{places}f}".replace("-", MINUS)


def _sig(value: float) -> str:
    """Format to three significant figures without an exponent."""
    text = f"{value:.3g}"
    if "e" in text:
        text = f"{value:.6f}".rstrip("0")
    return text.replace("-", MINUS)


def _count(value: float) -> str:
    """Format an integer count with thin-space thousands separators."""
    return f"{round(value):,}".replace(",", " ")


def _percent(share: float, places: int = 0) -> str:
    """Format a share between 0 and 1 as a percentage."""
    return f"{100 * share:.{places}f}%"


def _round(value: Any, digits: int = 4) -> Any:
    """Round every float in a nested structure, to keep the embedded data small."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def _one(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any]:
    """Return the single row matching every ``key=value`` given; fail if it is not unique."""
    found = [row for row in rows if all(row[key] == value for key, value in where.items())]
    if len(found) != 1:
        raise KeyError(f"expected one row for {where}, found {len(found)}")
    return found[0]


def _chains(u3: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pick the three chains the page keeps returning to, from the U3 sweep on Nason."""
    configs = u3["configurations"]
    nason = u3["datasets"]["nason"]
    return {
        "conv": configs[nason["baseline"]],
        "low": configs[nason["winner"]],
        "raw": next(row for row in configs if row["feature"] == "raw"),
    }


def build_facts(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Format every number the page's prose quotes, keyed by placeholder name."""
    u0, u1, u2, u3 = results["u0"], results["u1"], results["u2"], results["u3"]
    u4, u5, u6, u7 = results["u4"], results["u5"], results["u6"], results["u7"]
    facts: dict[str, str] = {"repo_url": REPO_URL}

    # Groundwork (U0 to U2).
    facts["u0_clips"] = _count(u0["dataset"]["n_clean_clips"])
    facts["u0_sigma_lsb"] = _fixed(u0["noise"]["sigma"]["median"])
    facts["u0_lfp_share"] = _percent(u0["spectrum"]["lfp_power_fraction"]["median"], 1)
    noise = u1["noise"]
    facts["u1_noise_hand"] = _fixed(noise["in_saline"]["hand"]["total_uv"], 3)
    facts["u1_noise_analytic"] = _fixed(noise["in_saline"]["analytic"]["total_uv"], 3)
    facts["u1_noise_spice"] = _fixed(noise["in_saline"]["spice"]["total_uv"], 3)
    facts["u1_disagreement"] = _percent(
        max(entry["max_disagreement"] for entry in noise.values()), 2
    )
    facts["u1_nef"] = _fixed(u1["amplifier"]["nef_measured"])
    facts["u1_z_kohm"] = _fixed(u1["electrodes"]["in_saline"]["z_1khz_ohm"] / 1e3, 0)
    by_bits = u2["mearec"]["detection"]["bits"]["threshold"]
    facts["u2_f1_10b"] = _fixed(by_bits["10"]["f1"])
    facts["u2_f1_6b"] = _fixed(by_bits["6"]["f1"])
    facts["u2_f1_5b"] = _fixed(by_bits["5"]["f1"])
    facts["u2_noise_uv"] = _fixed(u3["config"]["reference_noise_uv"], 0)

    # Power and co-design (U3).
    chains = _chains(u3)
    top = chains["raw"]["total_uw"]
    for name, row in chains.items():
        facts[f"{name}_label"] = row["label"]
        facts[f"{name}_uw"] = _fixed(row["total_uw"])
        facts[f"{name}_pct"] = f"{100 * row['total_uw'] / top:.2f}"
        for stage, key in (("amp", "lna_uw"), ("adc", "adc_uw"), ("dsp", "dsp_uw")):
            facts[f"{name}_{stage}"] = _sig(row[key])
        facts[f"{name}_link"] = _sig(row["link_uw"])
        facts[f"{name}_bps"] = _count(row["bit_rate_bps"])
    facts["power_ratio"] = _fixed(chains["conv"]["total_uw"] / chains["low"]["total_uw"], 1)
    facts["amp_share"] = _percent(chains["conv"]["lna_uw"] / chains["conv"]["total_uw"])
    facts["n_configs"] = str(len(u3["configurations"]))
    facts["n_infeasible"] = str(len(u3["infeasible"]))
    facts["tolerance"] = _percent(u3["config"]["tolerance"])
    power = u3["config"]["power"]
    facts["v_supply"] = _sig(power["v_supply_v"])
    facts["nef"] = _sig(power["nef"])
    facts["adc_fom_fj"] = _sig(power["adc_fom_j"] * 1e15)
    facts["multiply_pj"] = _sig(power["energy_per_multiply_j"] * 1e12)
    facts["bit_pj"] = _sig(power["energy_per_bit_j"] * 1e12)
    facts["conv_r2_ridge"] = _fixed(u3["datasets"]["nason"]["baseline_r2"], 3)
    facts["low_r2_ridge"] = _fixed(chains["low"]["r2"]["nason"]["mean"], 3)
    for short, key in (("s1", "nomad_session1"), ("s2", "nomad_session2")):
        info = u3["datasets"][key]
        winner = u3["configurations"][info["winner"]]
        baseline = u3["configurations"][info["baseline"]]
        facts[f"{short}_low_uw"] = _fixed(winner["total_uw"])
        facts[f"{short}_ratio"] = _fixed(baseline["total_uw"] / winner["total_uw"], 1)
    facts["fixed_cross_a"] = _fixed(u3["cross_session"]["train_early_test_late"])
    facts["fixed_cross_b"] = _fixed(u3["cross_session"]["train_late_test_early"])
    bridge = u3["bridge_check"]
    facts["bridge_ua"] = _sig(bridge["low_current_ua"])
    facts["bridge_cross_actual"] = _fixed(bridge["crossings_r_actual"])
    facts["bridge_cross_predicted"] = _fixed(bridge["crossings_r_predicted"])
    facts["bridge_sbp_actual"] = _fixed(bridge["sbp_r_actual"], 3)
    facts["bridge_sbp_predicted"] = _fixed(bridge["sbp_r_predicted"], 3)

    # Decoders (U4).
    for name in ("wiener", "gru"):
        row = u4["headline"]["nason"]["decoders"][name]
        diff = row["winner_minus_conventional"]
        facts[f"{name}_conv_r2"] = _fixed(row["conventional_r2"], 3)
        facts[f"{name}_low_r2"] = _fixed(row["winner_r2"], 3)
        facts[f"{name}_diff"] = _signed(diff["difference"])
        facts[f"{name}_diff_low"] = _signed(diff["low"])
        facts[f"{name}_diff_high"] = _signed(diff["high"])
    facts["n_folds"] = str(u4["config"]["n_folds"])
    facts["n_boot"] = _count(u4["config"]["n_boot"])
    facts["gru_hidden"] = str(u4["config"]["gru_hidden"])
    facts["gru_uw"] = _sig(u4["decoder_cost"]["nason"]["gru"]["uw_per_channel"])
    facts["days_apart"] = _fixed(u4["cross_session"]["days_apart"], 1)

    # Stimulation (U5).
    pulse = u5["config"]["pulse"]
    key5 = u5["key"]
    facts["pulse_ua"] = _sig(pulse["amplitude_ua"])
    facts["pulse_us"] = _sig(pulse["phase_s"] * 1e6)
    facts["pulse_rate"] = _sig(u5["config"]["rate_hz"])
    facts["example_mv"] = _sig(key5["example_mv"])
    same = _one(u5["propagation"]["scenarios"], name="same array")
    facts["artifact_mv"] = _fixed(same["median_mv"], 1)
    facts["artifact_min_mv"] = _fixed(same["min_mv"], 1)
    facts["artifact_max_mv"] = _fixed(same["max_mv"], 1)
    facts["adc_range_mv"] = _sig(u5["propagation"]["adc_range_mv"])
    facts["recovery_low_ms"] = _fixed(key5["recovery_ms"]["sbp"])
    facts["recovery_low_spice_ms"] = _fixed(key5["spice_recovery_ms"]["sbp"])
    facts["recovery_conv_ms"] = _fixed(key5["recovery_ms"]["crossings"])
    facts["recovery_threshold_uv"] = _sig(u5["recovery"]["threshold_uv"])
    facts["bias_none"] = _signed(key5["bias"]["none"], 2)
    facts["bias_blank"] = _signed(key5["bias"]["blank"], 2)
    facts["bias_template"] = _signed(key5["bias"]["template"], 2)
    facts["valid_blank"] = _percent(key5["valid"]["blank"])
    facts["valid_template"] = _percent(key5["valid"]["template"])
    facts["stim_clean_r2"] = _fixed(key5["no_stimulation_r2"], 3)
    rows5 = u5["decoding"]["rows"]
    for treatment in ("none", "blank", "template"):
        for fitted, tag in (("without stimulation", "before"), ("with stimulation", "refit")):
            row = _one(
                rows5,
                chain="sbp",
                scenario="same array",
                treatment=treatment,
                decoder="gru",
                fitted=fitted,
            )
            facts[f"stim_{treatment}_{tag}_r2"] = _fixed(row["r2"], 3)
            facts[f"stim_{treatment}_{tag}_loss"] = _signed(row["loss"])
            facts[f"stim_{treatment}_{tag}_loss_low"] = _signed(row["loss_low"])
            facts[f"stim_{treatment}_{tag}_loss_high"] = _signed(row["loss_high"])
    facts["canceller_uw"] = _sig(u5["cost"]["chains"]["sbp"]["canceller_uw"])
    facts["canceller_bits"] = str(u5["cost"]["chains"]["sbp"]["template_bits_per_channel"])
    facts["in_vivo_bias_none"] = _signed(u5["in_vivo"]["treatments"]["none"]["bias"], 2)
    facts["in_vivo_bias_template"] = _signed(u5["in_vivo"]["treatments"]["template"]["bias"], 2)

    # Aging (U6).
    key6 = u6["key"]
    states = u6["model"]["states"]
    facts["key_week"] = _sig(key6["week"])
    facts["last_week"] = _sig(key6["last_week"])
    facts["z_fresh_kohm"] = _fixed(states[0]["z_1khz_kohm"], 0)
    facts["z_aged_kohm"] = _fixed(states[-1]["z_1khz_kohm"], 0)
    facts["noise_fresh_uv"] = _fixed(states[0]["frontend_noise_uv"], 1)
    facts["noise_aged_uv"] = _fixed(states[-1]["frontend_noise_uv"], 1)
    facts["impedance_rise"] = _sig(u6["config"]["aging"]["impedance_factor_final"])
    facts["decline_per_month"] = _percent(u6["config"]["aging"]["amplitude_decline_per_month"], 1)
    for tag, condition in (("low", "nason_sbp"), ("conv", "nason_crossings")):
        curve = key6["curves"][condition]["gru"]
        facts[f"age_{tag}_week0"] = _fixed(curve["week0"], 3)
        facts[f"age_{tag}_fixed"] = _fixed(curve["key_fixed"]["r2"], 3)
        facts[f"age_{tag}_zscore"] = _fixed(curve["key_zscore"]["r2"], 3)
        facts[f"age_{tag}_refit"] = _fixed(curve["key_refit"]["r2"], 3)
        facts[f"age_{tag}_gain"] = _signed(curve["key_zscore"]["gain"])
        facts[f"age_{tag}_gain_low"] = _signed(curve["key_zscore"]["gain_low"])
        facts[f"age_{tag}_gain_high"] = _signed(curve["key_zscore"]["gain_high"])
    facts["stabiliser_uw"] = _sig(key6["stabiliser_uw"])
    facts["stabiliser_bits"] = str(u6["cost"]["modes"]["zscore"]["register_bits_per_channel"])
    facts["n_latents"] = str(u6["cross_session"]["n_latents"])
    facts["calibration_s"] = _fixed(u6["cross_session"]["calibration_s"], 0)
    facts["n_variations"] = str(len(u6["variations"]))

    # RTL (U7).
    key7 = u7["key"]
    row96 = _one(u7["synthesis"]["sweep"], n_channels=key7["n_channels"])
    budget = u7["power"]["budget"]
    facts["rtl_values"] = _count(key7["values_compared"])
    facts["rtl_mismatches"] = _count(key7["mismatches"])
    facts["rtl_channels"] = str(key7["n_channels"])
    facts["rtl_seconds"] = _fixed(key7["seconds"], 0)
    facts["rtl_gate_values"] = _count(u7["gate_level"]["values_compared"])
    facts["rtl_gate_mismatches"] = _count(u7["gate_level"]["mismatches_against_rtl"])
    facts["rtl_gates"] = _count(u7["gate_level"]["gates"]["gates"])
    facts["rtl_lut4"] = _count(key7["lut4"])
    facts["rtl_ff"] = _count(key7["flip_flops"])
    facts["rtl_bram"] = str(key7["block_rams"])
    facts["rtl_bram_available"] = str(row96["block_rams_available"])
    facts["rtl_cells"] = _count(key7["logic_cells"])
    facts["rtl_cells_available"] = _count(row96["logic_cells_available"])
    facts["rtl_fmax"] = _fixed(key7["fmax_mhz"], 1)
    facts["rtl_fmax_low"] = _fixed(min(row96["fmax_mhz"]), 1)
    facts["rtl_fmax_high"] = _fixed(max(row96["fmax_mhz"]), 1)
    facts["rtl_clock_khz"] = _fixed(key7["clock_needed_hz"] / 1e3, 0)
    facts["rtl_state_bits"] = str(u7["state_bits"]["total"])
    facts["rtl_device"] = u7["synthesis"]["device"]
    facts["digital_uw"] = _fixed(key7["digital_uw"], 4)
    facts["digital_low_uw"] = _fixed(key7["digital_low_uw"], 4)
    facts["digital_high_uw"] = _fixed(key7["digital_high_uw"], 3)
    facts["budget_uw"] = _fixed(budget["after_uw"], 3)
    facts["budget_low_uw"] = _fixed(budget["after_low_uw"], 3)
    facts["budget_high_uw"] = _fixed(budget["after_high_uw"], 3)
    facts["budget_amp_share"] = _percent(budget["lna_share"])
    facts["verilator"] = u7["tools"]["verilator"]
    facts["yosys"] = u7["tools"]["yosys"]
    return facts


def build_payload(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Select and round the arrays the interactive charts draw."""
    u3, u4, u5, u6, u7 = (results[key] for key in ("u3", "u4", "u5", "u6", "u7"))

    pareto = {
        "tolerance": u3["config"]["tolerance"],
        "configs": [
            {
                "label": row["label"],
                "feature": row["feature"],
                "uw": row["total_uw"],
                "parts": [row["lna_uw"], row["adc_uw"], row["dsp_uw"], row["link_uw"]],
                "r2": {key: value["mean"] for key, value in row["r2"].items()},
            }
            for row in u3["configurations"]
        ],
        "datasets": {
            key: {
                "name": DATASET_NAMES[key],
                "channels": info["n_channels"],
                "baseline": info["baseline"],
                "winner": info["winner"],
                "front": info["pareto"],
            }
            for key, info in u3["datasets"].items()
        },
    }

    decoders: dict[str, Any] = {"names": DECODER_NAMES, "datasets": {}}
    for key, head in u4["headline"].items():
        entry: dict[str, Any] = {"name": DATASET_NAMES[key], "rows": {}}
        for role in ("conventional", "winner"):
            label = head[role]
            entry[role] = {"label": label}
            for name in DECODERS:
                row = _one(u4["results"], dataset=key, label=label, decoder=name)
                entry[role]["uw"] = row["total_uw"]
                entry["rows"].setdefault(name, {})[role] = [
                    row["r2"],
                    row["r2_low"],
                    row["r2_high"],
                ]
        for name in DECODERS:
            diff = head["decoders"][name]["winner_minus_conventional"]
            entry["rows"][name]["diff"] = [diff["difference"], diff["low"], diff["high"]]
        decoders["datasets"][key] = entry
    example = u4["example"]
    decoders["example"] = {
        "label": example["label"],
        "bin_s": example["bin_s"],
        "truth": example["truth"],
        **{name: example[name] for name in DECODERS},
    }

    stim_rows = [
        {
            "chain": row["chain"],
            "treatment": row["treatment"],
            "fitted": row["fitted"],
            "r2": [row["r2"], row["r2_low"], row["r2_high"]],
            "loss": [row["loss"], row["loss_low"], row["loss_high"]],
        }
        for row in u5["decoding"]["rows"]
        if row["scenario"] == "same array" and row["decoder"] == "gru"
    ]
    stim = {
        "trace": u5["in_vivo"]["trace"],
        "amplitude_mv": u5["in_vivo"]["amplitude_mv"],
        "rows": stim_rows,
        "clean": {
            chain: u5["decoding"]["no_stimulation_r2"][chain]["gru"]
            for chain in ("sbp", "crossings")
        },
    }

    aging = {
        "key_week": u6["key"]["week"],
        "conditions": {
            key: {
                "name": DATASET_NAMES[info["dataset"]],
                "chain": info["chain"],
                "uw": info["chain_uw"],
            }
            for key, info in u6["decoding"]["conditions"].items()
        },
        "rows": [
            [
                row["condition"],
                row["decoder"],
                row["week"],
                row["mode"],
                row["r2"],
                row["r2_low"],
                row["r2_high"],
            ]
            for row in u6["decoding"]["rows"]
            if row["variation"] == "baseline" and row["mode"] in ("fixed", "zscore", "refit")
        ],
    }

    example7 = u7["example"]
    rtl = {
        "fs_hz": example7["fs_hz"],
        "scale": example7["scale"],
        "codes": example7["codes"],
        "cancelled": example7["rtl"]["cancelled"],
        "valid": example7["rtl"]["valid"],
        "identical": all(
            example7["rtl"][key] == example7["python"][key] for key in example7["rtl"]
        ),
        "channel": example7["channel"],
    }

    payload = {"pareto": pareto, "decoders": decoders, "stim": stim, "aging": aging, "rtl": rtl}
    rounded: dict[str, Any] = _round(payload)
    return rounded


def _table(head: list[str], rows: list[list[str]]) -> str:
    """Render an HTML table; the first column is a row header, the rest are numbers."""
    out = ["<table><thead><tr>"]
    out += [f"<th scope='col'>{html.escape(cell)}</th>" for cell in head]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append(f"<tr><th scope='row'>{html.escape(row[0])}</th>")
        out += [f"<td>{html.escape(cell)}</td>" for cell in row[1:]]
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def build_tables(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Render the static tables: the real session pair, changed assumptions, channel counts."""
    u6, u7 = results["u6"], results["u7"]

    pair = u6["key"]["cross_session"]
    session = _table(
        ["Decoder", "Left alone", "Running z-score", "Aligned latents", "Refitted"],
        [
            [
                DECODER_NAMES[name],
                *(_fixed(pair[name][mode], 3) for mode in ("fixed", "zscore", "aligned", "within")),
            ]
            for name in DECODERS
        ],
    )

    rows6 = u6["decoding"]["rows"]
    week = u6["key"]["week"]
    changed: list[list[str]] = []
    for variation in ["baseline", *u6["variations"]]:

        def at(condition: str, mode: str, variation: str = variation) -> dict[str, Any]:
            return _one(
                rows6, variation=variation, condition=condition, decoder="gru", week=week, mode=mode
            )

        changed.append(
            [
                "As modelled" if variation == "baseline" else variation.capitalize(),
                _fixed(at("nason_sbp", "fixed")["r2"], 3),
                _fixed(at("nason_sbp", "zscore")["r2"], 3),
                _fixed(at("nason_crossings", "fixed")["r2"], 3),
                _fixed(at("nason_crossings", "zscore")["r2"], 3),
            ]
        )
    variations = _table(
        [
            "Assumption changed",
            "Low-power, left alone",
            "Low-power, z-score",
            "Conventional, left alone",
            "Conventional, z-score",
        ],
        changed,
    )

    channels = _table(
        ["Channels", "LUT4", "Flip-flops", "Block RAMs", "fmax, MHz", "Clock needed, kHz"],
        [
            [
                _count(row["n_channels"]),
                _count(row["lut4"]),
                _count(row["flip_flops"]),
                str(row["block_rams"]),
                _fixed(row["fmax_median_mhz"], 1),
                _sig(row["clock_needed_hz"] / 1e3),
            ]
            for row in u7["synthesis"]["sweep"]
            if "error" not in row
        ],
    )
    return {
        "table_session_pair": session,
        "table_variations": variations,
        "table_channels": channels,
    }


def fill(template: str, values: dict[str, str]) -> str:
    """Replace every ``{{name}}`` in ``template``; an unknown name raises ``KeyError``."""

    def swap(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise KeyError(f"the template quotes {{{{{name}}}}}, which no result provides")
        return values[name]

    return _PLACEHOLDER.sub(swap, template)


def render(results: dict[str, dict[str, Any]], source_dir: Path | None = None) -> tuple[str, str]:
    """Return the page as ``(head, body)`` HTML fragments, with style, script and data inline."""
    source_dir = source_dir or repo_root() / "site"
    values = {**build_facts(results), **build_tables(results)}
    template = (source_dir / "template.html").read_text(encoding="utf-8")
    head_part, _, body_part = template.partition("<!-- BODY -->")
    style = (source_dir / "style.css").read_text(encoding="utf-8")
    script = (source_dir / "app.js").read_text(encoding="utf-8")
    data = json.dumps(build_payload(results), ensure_ascii=False, separators=(",", ":"))
    head = fill(head_part, values) + f"<style>\n{style}</style>\n"
    body = (
        fill(body_part, values)
        + '<script type="application/json" id="nc-data">'
        + data.replace("</", "<\\/")
        + "</script>\n"
        + f"<script>\n{script}</script>\n"
    )
    return head, body


def build_site(
    out_dir: Path | None = None,
    results_directory: Path | None = None,
    source_dir: Path | None = None,
) -> Path:
    """Write the site and return the path of its ``index.html``.

    ``index.html`` is a complete document that opens from disk with no server.
    ``fragment.html`` is the same page without the document skeleton, for hosts
    that supply their own.
    """
    out_dir = out_dir or repo_root() / "build" / "site"
    out_dir.mkdir(parents=True, exist_ok=True)
    head, body = render(load_results(results_directory), source_dir)
    document = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head}</head>\n<body>\n{body}</body>\n</html>\n"
    )
    index = out_dir / "index.html"
    index.write_text(document, encoding="utf-8")
    (out_dir / "fragment.html").write_text(head + body, encoding="utf-8")
    return index
