"""The RTL's specification, stimulus and tools.

Tests that compile or synthesise need Verilator, Yosys or nextpnr-ecp5 and skip
without them (CI has none). The rest hold the Python side to its claims with no
tool and no downloaded data.
"""

from pathlib import Path

import numpy as np
import pytest

from neurochain.analysis import u7
from neurochain.data import neuralink
from neurochain.fixedpoint import fixed_highpass, fixed_masked_sbp
from neurochain.rtl import energy, tools
from neurochain.rtl.spec import ChainSpec, csd_digits
from neurochain.rtl.stimulus import (
    band_limit,
    neuralink_stimulus,
    synthetic_stimulus,
    worst_case_stimulus,
)
from neurochain.stim import cancel

needs_verilator = pytest.mark.skipif(
    not tools.tools_available("verilator"), reason="verilator not installed"
)
needs_yosys = pytest.mark.skipif(
    not tools.tools_available("verilator", "yosys"), reason="verilator or yosys not installed"
)
needs_nextpnr = pytest.mark.skipif(
    not tools.tools_available("yosys", "nextpnr-ecp5"), reason="yosys or nextpnr-ecp5 not installed"
)

SMALL = ChainSpec(n_channels=4)


# ---------------------------------------------------------------------- specification


def test_masked_sbp_is_the_integer_form_of_u5s_mean() -> None:
    rng = np.random.default_rng(0)
    filtered = rng.integers(-900, 900, size=(3, 1050))
    valid = rng.random((3, 1050)) > 0.2
    valid[1, 200:300] = False  # one bin with nothing left
    sums, counts = fixed_masked_sbp(filtered, valid, 100)
    assert sums.shape == counts.shape == (3, 10)
    assert counts[1, 2] == 0 and sums[1, 2] == 0
    mean, fraction = cancel.masked_mean_abs(filtered, valid, 100)
    kept = counts > 0
    np.testing.assert_allclose(sums[kept] / counts[kept], mean[kept])
    np.testing.assert_allclose(counts / 100, fraction)


def test_csd_recoding_is_exact_and_sparse() -> None:
    for n in range(1, 3000):
        plus, minus = csd_digits(n)
        assert plus - minus == n
        digits = plus | minus
        assert plus & minus == 0 and digits & (digits << 1) == 0
        assert csd_digits(-n) == (plus, minus)
    assert csd_digits(191) == (256, 65)  # 256 - 64 - 1: three terms, not seven


def test_default_widths_match_the_design_note() -> None:
    spec = ChainSpec()
    filt = spec.highpass
    assert (filt.b0, filt.a1, filt.a2, filt.guard_bits) == (129, -191, 70, 0)
    assert spec.cancel_bound == 1016
    assert spec.y_bound == 1536
    assert (spec.template_bits, spec.cancel_bits, spec.y_bits) == (14, 11, 12)
    assert (spec.accumulator_bits, spec.sum_bits, spec.count_bits) == (20, 18, 7)
    assert sum(spec.state_bits.values()) == 307
    assert spec.adders_per_sample == 18
    assert spec.verilog_parameters()["BIN_LEN"] == 100
    assert spec.memory_bits_per_sample(0.0) == 2 * (5 + 46 + 32)
    assert spec.memory_bits_per_sample(1.0) == 2 * (5 + 46 + 32 + 14)


def test_worst_case_stimulus_reaches_the_bounds_and_no_further() -> None:
    spec = ChainSpec(n_channels=6)
    stimulus = worst_case_stimulus(spec)
    ref = spec.reference(stimulus.codes, stimulus.onsets)
    assert np.abs(ref.cancelled).max() == spec.cancel_bound
    peak = int(np.abs(ref.filtered).max())
    assert 0.99 * spec.y_bound <= peak <= spec.y_bound
    assert ref.accumulator_peak < 1 << (spec.accumulator_bits - 1)
    assert ref.sums.max() <= spec.sum_bound
    # U2 measured 18 bits on noise; the worst case needs the full 20.
    assert ref.accumulator_peak.bit_length() + 1 == spec.accumulator_bits


def test_reference_is_the_chain_u5_checked() -> None:
    stimulus = synthetic_stimulus(SMALL, 2.0, seed=3)
    ref = SMALL.reference(stimulus.codes, stimulus.onsets)
    row = stimulus.codes[2].astype(np.int64)
    cancelled, valid = cancel.cancel_template(
        row, stimulus.onsets, n_taps=16, shift=4, code_min=-128, code_max=127
    )
    filtered, _ = fixed_highpass(cancelled, SMALL.highpass)
    np.testing.assert_array_equal(ref.cancelled[2], cancelled)
    np.testing.assert_array_equal(ref.valid[2], valid)
    np.testing.assert_array_equal(ref.filtered[2], filtered)
    assert ref.sums.shape == (4, stimulus.n_samples // 100)
    assert not ref.valid.all() and ref.counts.min() < 100
    one = SMALL.reference(stimulus.codes[2], stimulus.onsets)
    np.testing.assert_array_equal(one.sums[0], ref.sums[2])


def test_specification_rejects_what_the_rtl_cannot_do() -> None:
    with pytest.raises(ValueError, match="two template taps"):
        ChainSpec(n_taps=1)
    with pytest.raises(ValueError, match="out_frac_bits"):
        ChainSpec(out_frac_bits=7)
    with pytest.raises(ValueError, match="adapt_shift"):
        ChainSpec(adapt_shift=-1)
    with pytest.raises(ValueError, match="at least 16 samples apart"):
        ChainSpec().reference(np.zeros(100, dtype=np.int64), np.array([10, 20]))


# --------------------------------------------------------------------------- stimulus


def test_synthetic_stimulus_is_seeded_and_rails() -> None:
    a = synthetic_stimulus(SMALL, 3.0, seed=1)
    b = synthetic_stimulus(SMALL, 3.0, seed=1)
    c = synthetic_stimulus(SMALL, 3.0, seed=2)
    np.testing.assert_array_equal(a.codes, b.codes)
    assert not np.array_equal(a.codes, c.codes)
    assert a.codes.dtype == np.int8 and a.codes.shape == (4, 6000)
    assert a.seconds == 3.0
    assert a.info["railed_fraction"] > 0 and 1.2 < a.info["quiet_rms_lsb"] < 1.7
    flags = a.trigger()
    assert flags.sum() == a.onsets.size == a.info["n_pulses"]
    assert a.window_fraction(16) == pytest.approx(16 * a.onsets.size / 6000, rel=0.01)
    head = a.head(1000)
    assert head.codes.shape == (4, 1000) and head.onsets.max() < 1000


def test_band_limit_keeps_the_band_and_drops_dc() -> None:
    fs = 19_531.0
    t = np.arange(round(2.0 * fs)) / fs
    inside = band_limit(300.0 + 100.0 * np.sin(2 * np.pi * 550.0 * t), fs, out_fs_hz=2000.0)
    assert inside.size == int(t.size * 2000.0 / fs)
    assert abs(inside[:50]).max() < 110.0  # the DC level causes no start-up transient
    assert np.sqrt(2.0) * inside[1000:].std() == pytest.approx(100.0, rel=0.05)
    assert abs(inside[1000:].mean()) < 0.5
    outside = band_limit(100.0 * np.sin(2 * np.pi * 30.0 * t), fs, out_fs_hz=2000.0)
    assert outside[1000:].std() < 2.0


@pytest.mark.data
def test_neuralink_stimulus_on_real_clips() -> None:
    try:
        neuralink.list_clips()
    except FileNotFoundError:
        pytest.skip("Neuralink clips not downloaded")
    stimulus = neuralink_stimulus(SMALL, 12.0, seed=0)
    assert stimulus.codes.shape == (4, 24_000) and stimulus.codes.dtype == np.int8
    info = stimulus.info
    assert info["n_clips"] == 743 and info["unique_seconds"] > 3600
    assert 0.5 < info["quiet_rms_lsb"] < 4.0
    assert info["n_pulses"] > 100 and info["pulse_period_samples"] == 20
    assert info["railed_fraction"] > 0  # the 10 mV and larger artifacts rail the converter
    again = neuralink_stimulus(SMALL, 12.0, seed=0)
    np.testing.assert_array_equal(stimulus.codes, again.codes)
    ref = SMALL.reference(stimulus.codes, stimulus.onsets)
    assert ref.accumulator_peak < 1 << (SMALL.accumulator_bits - 1)


# ------------------------------------------------------------------- tools and energy


def test_sources_are_where_the_tools_expect_them() -> None:
    assert all(path.exists() for path in tools.source_paths())
    assert (tools.rtl_dir() / tools.TESTBENCH).exists()
    assert tools.build_dir().parent.name == "build"  # ignored by git


def test_missing_tool_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROCHAIN_VERILATOR", "/nowhere/verilator")
    assert tools.find_tool("verilator") is None
    assert not tools.tools_available("verilator")
    with pytest.raises(tools.ToolError, match="not installed"):
        tools.build_model(SMALL, Path("unused"))
    with pytest.raises(tools.ToolError, match="verilator is not installed"):
        u7.compute_u7(u7.U7Config(spec=SMALL), synthetic_stimulus(SMALL, 1.0, seed=0))


def test_sign_extension() -> None:
    raw = np.array([0, 1, 1023, 1024, 2047], dtype=np.uint16)
    np.testing.assert_array_equal(tools.sign_extend(raw, 11), [0, 1, 1023, -1024, -1])


def test_saif_reader_counts_one_instance(tmp_path: Path) -> None:
    text = """(SAIFILE
 (INSTANCE TOP
  (NET
   (clk (T0 5) (T1 5) (TZ 0) (TX 0) (TB 0) (TC 20))
  )
  (INSTANCE nc_sbp_chain
   (NET
    (clk (T0 5) (T1 5) (TZ 0) (TX 0) (TB 0) (TC 20))
    (rst (T0 10) (T1 0) (TZ 0) (TX 0) (TB 0) (TC 0))
    (_0007_ (T0 4) (T1 6) (TZ 0) (TX 0) (TB 0) (TC 7))
    (s4_y\\[3\\] (T0 4) (T1 6) (TZ 0) (TX 0) (TB 0) (TC 3))
   )
  )
 )
)
"""
    path = tmp_path / "a.saif"
    path.write_text(text)
    toggles = tools.read_saif(path)
    assert toggles == {"clk": 20, "rst": 0, "_0007_": 7, "s4_y[3]": 3}
    activity = energy.summarise_activity(toggles, samples=10)
    assert activity["nets"] == 2 and activity["toggles_per_sample"] == 1.0
    assert activity["cycles_per_sample"] == 1.0 and activity["activity"] == 0.5


def test_energy_model_adds_up() -> None:
    model = energy.EnergyModel()
    assert model.toggle_j == pytest.approx(1.44e-15)
    result = energy.chain_energy(
        model,
        toggles_per_sample=1000.0,
        flip_flops=100,
        gates=2000,
        memory_bits_per_sample=200.0,
        memory_bits=30_000,
        n_channels=100,
        fs_hz=2000.0,
    )
    assert result["logic_pj"] == pytest.approx(1.44)
    assert result["clock_pj"] == pytest.approx(0.288)
    assert result["memory_pj"] == pytest.approx(4.0)
    assert result["dynamic_uw"] == pytest.approx((1.44 + 0.288 + 4.0) * 2000.0 * 1e-6)
    assert result["leakage_uw"] == pytest.approx((2600 + 30_000) * 5e-12 / 100 * 1e6)
    assert result["total_uw"] == pytest.approx(result["dynamic_uw"] + result["leakage_uw"])


# ------------------------------------------------------------------ RTL against Python


@needs_verilator
def test_rtl_matches_python_on_noise_and_artifacts(tmp_path: Path) -> None:
    stimulus = synthetic_stimulus(SMALL, 3.0, seed=5)
    summary, out = u7.bit_exact(stimulus, SMALL, tmp_path, name="run", max_workers=1)
    assert summary["total_mismatches"] == 0 and summary["tag_errors"] == 0
    assert summary["samples_compared"] == 4 * 6000 and summary["bins_compared"] == 4 * 60
    assert summary["blanked_fraction"] > 0.005  # the comparison saw blanked samples
    assert out.sums.shape == (60, 4)
    example = u7.example_traces(stimulus, out, SMALL, 0.2)
    assert example["rtl"] == example["python"] and len(example["codes"]) == 400


@needs_verilator
def test_rtl_matches_python_at_the_worst_case(tmp_path: Path) -> None:
    summary, _ = u7.bit_exact(
        worst_case_stimulus(SMALL), SMALL, tmp_path, name="worst", max_workers=1
    )
    assert summary["total_mismatches"] == 0
    assert summary["peak"]["cancelled"] == SMALL.cancel_bound
    rows = u7.width_table(SMALL, summary, summary)
    assert all(row["peak_real"] <= row["bound"] for row in rows)
    assert all(row["bits_used_real"] <= row["bits"] for row in rows)


@needs_verilator
def test_a_narrow_accumulator_is_caught(tmp_path: Path) -> None:
    # U2's 18 bits, measured on noise, overflow when a template is wrong by a full range.
    narrow = ChainSpec(n_channels=4, y_bits_override=10)
    assert narrow.accumulator_bits == 18
    summary, _ = u7.bit_exact(
        worst_case_stimulus(narrow), narrow, tmp_path, name="narrow", max_workers=1
    )
    assert summary["mismatches"]["filtered"] > 0 and summary["mismatches"]["cancelled"] == 0
    assert summary["first_mismatch"] is not None


@needs_verilator
@pytest.mark.parametrize(
    "spec",
    [
        ChainSpec(n_channels=1),
        ChainSpec(n_channels=3, n_taps=12, adapt_shift=2, bin_s=0.02),
        ChainSpec(n_channels=2, template_frac_bits=4, out_frac_bits=1, coef_frac_bits=10),
        ChainSpec(n_channels=5, code_bits=6, corner_hz=250.0, shift_add=False),
    ],
    ids=["one channel", "12 taps", "other fractions", "6 bits and *"],
)
def test_rtl_follows_its_parameters(tmp_path: Path, spec: ChainSpec) -> None:
    for stimulus in (synthetic_stimulus(spec, 2.0, seed=7), worst_case_stimulus(spec)):
        summary, _ = u7.bit_exact(stimulus, spec, tmp_path, name="run", max_workers=1)
        assert summary["total_mismatches"] == 0 and summary["tag_errors"] == 0


@needs_yosys
def test_gate_netlist_matches_rtl_and_toggles(tmp_path: Path) -> None:
    config = u7.U7Config(spec=SMALL, activity_s=1.0)
    stimulus = synthetic_stimulus(SMALL, 1.0, seed=9)
    gate = u7.gate_level(config, stimulus, tmp_path)
    assert gate["mismatches_against_rtl"] == 0 and gate["values_compared"] > 20_000
    assert gate["gates"]["memories"] == 4 and gate["gates"]["memory_bits"] == 4 * 307
    assert 1000 < gate["gates"]["gates"] < 6000
    assert gate["activity"]["cycles_per_sample"] == pytest.approx(1.0, abs=0.01)
    assert gate["activity"]["toggles_per_sample"] > 100
    silent = u7.gate_level(u7.U7Config(spec=SMALL, activity_s=1.0, silent=True), stimulus, tmp_path)
    assert silent["mismatches_against_rtl"] == 0
    assert silent["activity"]["toggles_per_sample"] < 0.2 * gate["activity"]["toggles_per_sample"]


@needs_nextpnr
def test_ecp5_synthesis_reports_resources_and_timing(tmp_path: Path) -> None:
    row = tools.synthesise_ecp5(SMALL, tmp_path, seeds=(1,))
    assert 200 < row["lut4"] < 2000 and row["multipliers"] == 0
    assert row["logic_cells"] < row["logic_cells_available"] == 24_288
    assert len(row["fmax_mhz"]) == 1 and row["fmax_mhz"][0] > 10.0
    assert set(tools.tool_versions()) >= {"yosys", "nextpnr-ecp5"}
