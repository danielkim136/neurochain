import pytest

from neurochain.power import ChainConfig, PowerModel, chain_power


def test_each_stage_follows_its_formula() -> None:
    model = PowerModel()
    power = chain_power(ChainConfig("crossings", 10e-6, 10, 30_000.0), model)
    assert power.lna_w == pytest.approx(1.8 * 10e-6)
    assert power.adc_w == pytest.approx(50e-15 * 1024 * 30_000)
    assert power.dsp_w == pytest.approx(1e-12 * 3 * 30_000)
    assert power.link_w == pytest.approx(10e-12 * 4 / 0.05)
    assert power.total_w == pytest.approx(power.lna_w + power.adc_w + power.dsp_w + power.link_w)
    assert power.as_microwatts()["total_uw"] == pytest.approx(19.627, abs=1e-3)


def test_a_bit_or_a_doubling_of_rate_doubles_adc_power() -> None:
    base = chain_power(ChainConfig("crossings", 1e-6, 8, 15_000.0)).adc_w
    assert chain_power(ChainConfig("crossings", 1e-6, 9, 15_000.0)).adc_w == pytest.approx(2 * base)
    assert chain_power(ChainConfig("crossings", 1e-6, 8, 30_000.0)).adc_w == pytest.approx(2 * base)


def test_sbp_chain_is_pinned_to_two_kilosamples() -> None:
    chain = ChainConfig("sbp", 0.25e-6, 8, 30_000.0)
    assert chain.fs_hz == 2000.0
    assert chain.band_hz == (300.0, 1000.0)
    assert ChainConfig("crossings", 1e-6).band_hz == (1.0, 7500.0)
    assert chain.label == "sbp, 0.25 µA, 8 b, 2 kS/s"


def test_bit_rates() -> None:
    assert ChainConfig("raw", 10e-6, 10, 30_000.0).bit_rate_bps == 300_000.0
    assert ChainConfig("crossings", 1e-6).bit_rate_bps == 80.0
    assert ChainConfig("sbp", 1e-6).bit_rate_bps == 240.0
    assert ChainConfig("both", 1e-6).bit_rate_bps == 320.0
    assert ChainConfig("crossings", 1e-6, bin_s=0.02).bit_rate_bps == 200.0


def test_raw_streaming_pays_in_the_radio_not_the_logic() -> None:
    raw = chain_power(ChainConfig("raw", 10e-6))
    feature = chain_power(ChainConfig("crossings", 10e-6))
    assert raw.dsp_w == 0.0
    assert raw.link_w == pytest.approx(3e-6)
    assert raw.link_w > 1000 * feature.link_w


def test_constants_can_be_changed() -> None:
    chain = ChainConfig("sbp", 1e-6, 8)
    worse = chain_power(chain, PowerModel(adc_fom_j=500e-15, v_supply_v=1.2))
    assert worse.adc_w == pytest.approx(10 * chain_power(chain).adc_w)
    assert worse.lna_w == pytest.approx(1.2e-6)
