"""U3 on the real inputs: the committed metrics against a recomputation."""

import json

import numpy as np
import pytest

from neurochain.analysis import u3
from neurochain.data import nason, nomad
from neurochain.paths import results_dir
from neurochain.power import ChainConfig, PowerModel

HAVE_NASON = (nason.nason_dir() / nason.ONLINE_FILE).is_file()
HAVE_NOMAD = all((nomad.nomad_dir() / name).is_file() for name in nomad.SESSION_FILES)
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not HAVE_NASON, reason="Nason files not downloaded"),
]


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads((results_dir() / "u3_metrics.json").read_text())


def test_committed_rows_are_reproduced_from_their_degradation(committed: dict) -> None:
    """A row's R² follows from its stored degradation and its own random stream."""
    data = u3.nason_set()
    rows = committed["configurations"]
    info = committed["datasets"]["nason"]
    grid = u3.grid(u3.U3Config())
    feasible = [c for c in grid if c.label not in {i["label"] for i in committed["infeasible"]}]
    for index in (info["baseline"], info["winner"]):
        row = rows[index]
        position = next(i for i, c in enumerate(grid) if c.label == row["label"])
        assert feasible[index].label == row["label"]
        degradation = u3.Degradation(row["p_keep"], row["extra_rate_hz"], row["sbp_noise_ratio"])
        scores = []
        rng = np.random.default_rng([committed["config"]["seed"], position])
        for _ in range(row["r2"]["nason"]["n"]):
            features = u3.features_for(data, feasible[index], degradation, rng)
            assert features is not None
            scores.append(u3.decode(data, features))
        assert float(np.mean(scores)) == pytest.approx(row["r2"]["nason"]["mean"], abs=1e-9)


def test_headline(committed: dict) -> None:
    rows = committed["configurations"]
    info = committed["datasets"]["nason"]
    best, base = rows[info["winner"]], rows[info["baseline"]]
    assert best["feature"] == "sbp"
    assert base["total_uw"] / best["total_uw"] > 20.0
    assert best["r2"]["nason"]["mean"] >= 0.95 * info["baseline_r2"]
    again = u3.rescore(
        committed, PowerModel(**committed["config"]["power"]), committed["config"]["tolerance"]
    )
    assert again["nason"]["winner"] == best["label"]
    assert again["nason"]["winner_uw"] == pytest.approx(best["total_uw"])


def test_winner_survives_tenfold_changes_in_the_constants(committed: dict) -> None:
    base = PowerModel(**committed["config"]["power"])
    for model in (
        PowerModel(adc_fom_j=base.adc_fom_j * 10),
        PowerModel(energy_per_bit_j=base.energy_per_bit_j * 100),
        PowerModel(energy_per_multiply_j=base.energy_per_multiply_j * 10),
        PowerModel(nef=2.0),
    ):
        result = u3.rescore(committed, model, committed["config"]["tolerance"])
        assert ChainConfig("sbp", 1e-6).feature in result["nason"]["winner"]


@pytest.mark.skipif(not HAVE_NOMAD, reason="NoMAD files not downloaded")
def test_a_fixed_decoder_does_not_transfer_across_nomad_sessions(committed: dict) -> None:
    sessions = nomad.load_sessions()
    first, second = (u3.nomad_set(s, 3) for s in sessions)
    cross = committed["cross_session"]
    scores = sorted([u3.cross_session_r2(first, second), u3.cross_session_r2(second, first)])
    stored = sorted([cross["train_early_test_late"], cross["train_late_test_early"]])
    assert scores == pytest.approx(stored, abs=1e-9)
    assert max(scores) < 0.1
    assert cross["days_apart"] == pytest.approx(273.9, abs=0.1)
    # Within a session the same readout works.
    assert committed["datasets"]["nomad_session2"]["baseline_r2"] > 0.5
