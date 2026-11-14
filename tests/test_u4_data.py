"""U4 on the real inputs: the committed metrics against a recomputation."""

import json

import pytest

from neurochain.analysis import u3, u4
from neurochain.data import nason
from neurochain.paths import results_dir

HAVE_NASON = (nason.nason_dir() / nason.ONLINE_FILE).is_file()
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not HAVE_NASON, reason="Nason files not downloaded"),
]


def test_committed_linear_scores_are_reproduced() -> None:
    """The ridge, Wiener and Kalman rows for the two headline chains recompute exactly."""
    u3_summary = json.loads((results_dir() / "u3_metrics.json").read_text())
    committed = json.loads((results_dir() / "u4_metrics.json").read_text())
    head = committed["headline"]["nason"]
    labels = (head["conventional"], head["winner"])
    result = u4.compute_u4(
        u3_summary,
        config=u4.U4Config(decoders=("ridge", "wiener", "kalman")),
        sets={"nason": (u3.nason_set(), labels)},
    )
    expected = {
        (r["label"], r["decoder"]): r for r in committed["results"] if r["dataset"] == "nason"
    }
    for row in result["results"]:
        reference = expected[row["label"], row["decoder"]]
        assert row["r2"] == pytest.approx(reference["r2"], abs=1e-9)
        assert row["r2_low"] == pytest.approx(reference["r2_low"], abs=1e-9)
        assert row.get("delay_bins") == reference.get("delay_bins")
        assert row.get("penalty") == reference.get("penalty")


def test_committed_headline_is_consistent_with_its_rows() -> None:
    committed = json.loads((results_dir() / "u4_metrics.json").read_text())
    rows = {(r["dataset"], r["label"], r["decoder"]): r["r2"] for r in committed["results"]}
    assert set(committed["headline"]) == {"nason", "nomad_session1", "nomad_session2"}
    for name, head in committed["headline"].items():
        for decoder, entry in head["decoders"].items():
            assert entry["winner_r2"] == rows[name, head["winner"], decoder]
            assert entry["retained"] == pytest.approx(
                rows[name, head["winner"], decoder] / rows[name, head["conventional"], decoder]
            )
    assert len(committed["cross_session"]["rows"]) == 8
