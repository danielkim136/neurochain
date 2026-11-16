"""U7 end to end.

The full run needs Verilator, Yosys and nextpnr-ecp5 and skips without them.
The budget arithmetic, the figures, the README table and the key-number
printout are also exercised on the committed metrics, which needs no tool.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from neurochain.analysis import u7
from neurochain.analysis.u7_plots import plot_bit_exact, plot_power, plot_synthesis
from neurochain.paths import repo_root, results_dir
from neurochain.rtl import tools
from neurochain.rtl.spec import ChainSpec
from neurochain.rtl.stimulus import synthetic_stimulus

needs_tools = pytest.mark.skipif(
    not tools.tools_available("verilator", "yosys", "nextpnr-ecp5"),
    reason="verilator, yosys or nextpnr-ecp5 not installed",
)
ROWS = (
    "RTL against Python",
    "Worst-case stimulus",
    "Gate-level netlist against RTL",
    "Word widths",
    "ECP5 resources",
    "Fastest clock",
    "Digital energy per sample",
    "Digital power",
    "Chain budget",
)


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, repo_root() / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outputs(summary: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Draw the figures, build the table and print the key numbers of a summary."""
    for plot in (plot_bit_exact, plot_synthesis, plot_power):
        assert plot(summary, tmp_path / f"{plot.__name__}.png").stat().st_size > 20_000
    table = _script("update_readme").u7_table(summary)
    assert all(row in table for row in ROWS)
    _script("u7_key_number").show("title", summary)
    printed = capsys.readouterr().out
    assert "RTL against Python" in printed and "digital power" in printed
    return str(table)


def test_power_budget_from_counts() -> None:
    spec = ChainSpec(n_channels=4)
    config = u7.U7Config(spec=spec)
    stimulus = synthetic_stimulus(spec, 2.0, seed=0)
    gate = {
        "gates": {"gates": 2700, "flip_flops": 130, "memory_bits": 4 * 307},
        "activity": {"toggles_per_sample": 900.0, "cycles_per_sample": 1.0},
    }
    result = u7.power(config, gate, stimulus)
    assert result["placeholder"]["total_uw"] == pytest.approx(0.010)
    assert result["budget"]["before_uw"] == pytest.approx(0.488, abs=1e-6)
    assert result["energy"]["logic_pj"] == pytest.approx(900 * 1.44e-3)
    assert result["radio"]["bits_per_bin_rtl"] == 25 and result["radio"]["bits_per_bin_model"] == 12
    budget = result["budget"]
    assert budget["after_low_uw"] < budget["after_uw"] < budget["after_high_uw"]
    expected = budget["lna_uw"] + budget["adc_uw"] + result["energy"]["total_uw"]
    assert budget["after_uw"] == pytest.approx(expected + result["radio"]["link_rtl_uw"])
    spans = result["sensitivity"]
    assert spans["all three"]["tenfold_uw"] == pytest.approx(10 * result["energy"]["total_uw"])
    assert spans["leakage"]["tenth_uw"] < result["energy"]["total_uw"]
    assert result["cross_check"]["adders_per_sample"] == 18


def test_committed_metrics_meet_the_done_criterion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = results_dir() / "u7_metrics.json"
    if not path.exists():
        pytest.skip("results/u7_metrics.json not written yet")
    summary = json.loads(path.read_text())
    key = summary["key"]
    assert key["bit_exact"] and key["mismatches"] == 0
    assert key["seconds"] >= 600.0 and summary["stimulus"]["source"] == "neuralink"
    assert summary["worst_case"]["total_mismatches"] == 0
    assert summary["gate_level"]["mismatches_against_rtl"] == 0
    assert key["lut4"] > 0 and key["fmax_min_mhz"] * 1e6 > key["clock_needed_hz"]
    assert 0 < key["digital_low_uw"] < key["digital_uw"] < key["digital_high_uw"]
    assert all(w["peak_real"] <= w["bound"] and w["bits_used_real"] <= w["bits"]
               for w in summary["widths"])  # fmt: skip
    table = _outputs(summary, tmp_path, capsys)
    assert "0 mismatches" in table


@needs_tools
def test_u7_runs_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = ChainSpec(n_channels=4)
    config = u7.U7Config(
        spec=spec, activity_s=1.0, channel_counts=(1, 4), pnr_seeds=(1,), max_workers=1
    )
    stimulus = synthetic_stimulus(spec, 3.0, seed=11)
    summary = json.loads(json.dumps(u7.compute_u7(config, stimulus, directory=tmp_path)))
    key = summary["key"]
    assert key["bit_exact"] and key["values_compared"] == 3 * 4 * 6000 + 2 * 4 * 60
    assert summary["worst_case"]["total_mismatches"] == 0
    assert summary["gate_level"]["mismatches_against_rtl"] == 0
    assert [r["n_channels"] for r in summary["synthesis"]["sweep"]] == [1, 4]
    assert all("error" not in r for r in summary["synthesis"]["sweep"])
    variants = summary["synthesis"]["variants"]
    assert variants["star_dsp"]["multipliers"] == 3 and variants["star_lut"]["multipliers"] == 0
    assert variants["star_lut"]["lut4"] > key["lut4"]  # shifts and adds beat "*"
    assert summary["power"]["energy"]["total_uw"] > 0
    assert summary["example"]["rtl"] == summary["example"]["python"]
    assert not list(tmp_path.glob("*.bin"))  # simulation outputs are cleaned up
    _outputs(summary, tmp_path, capsys)
    reduced = u7.compute_u7(config, stimulus, sweep=False, directory=tmp_path)
    assert [r["n_channels"] for r in reduced["synthesis"]["sweep"]] == [4]
