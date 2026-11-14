"""U5 on the real inputs: the committed metrics against a recomputation."""

import json

import numpy as np
import pytest

from neurochain.analysis import u5
from neurochain.data import mearec
from neurochain.paths import results_dir

HAVE_MEAREC = (mearec.mearec_dir() / mearec.DEFAULT_FILE).is_file()
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not HAVE_MEAREC, reason="MEArec recording not generated"),
]


def test_committed_signal_level_is_reproduced() -> None:
    committed = json.loads((results_dir() / "u5_metrics.json").read_text())
    result = u5.signal_level(mearec.load_mearec(), u5.U5Config())
    for treatment in u5.TREATMENTS:
        for key in ("bias", "sd", "valid", "residual_uv"):
            assert np.allclose(
                result["sbp"][treatment][key], committed["signal"]["sbp"][treatment][key]
            ), (treatment, key)
        assert np.allclose(
            result["crossings"][treatment]["p_keep"],
            committed["signal"]["crossings"][treatment]["p_keep"],
        )
    assert result["blank_samples"] == committed["signal"]["blank_samples"]


def test_committed_key_is_consistent_with_its_rows() -> None:
    committed = json.loads((results_dir() / "u5_metrics.json").read_text())
    key = committed["key"]
    rows = {
        (r["chain"], r["scenario"], r["treatment"], r["decoder"], r["fitted"]): r
        for r in committed["decoding"]["rows"]
    }
    for chain in u5.CHAINS:
        for treatment in u5.TREATMENTS:
            row = rows[chain, key["scenario"], treatment, key["decoder"], "without stimulation"]
            assert key[f"{chain}_{treatment}"]["r2"] == row["r2"]
    u4_metrics = json.loads((results_dir() / "u4_metrics.json").read_text())
    head = u4_metrics["headline"]["nason"]["decoders"]["gru"]
    # The unstimulated decoders are U4's, scored here on all bins as there.
    reference = committed["decoding"]["no_stimulation_r2"]
    assert reference["sbp"]["gru_all_bins"] == pytest.approx(head["winner_r2"], abs=1e-6)
    assert reference["crossings"]["gru_all_bins"] == pytest.approx(
        head["conventional_r2"], abs=1e-6
    )
