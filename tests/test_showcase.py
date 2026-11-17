import json
import re
from pathlib import Path

import pytest

from neurochain import showcase
from neurochain.paths import repo_root

RESULTS = repo_root() / "results"
HAVE_RESULTS = all((RESULTS / f"u{n}_metrics.json").exists() for n in showcase.UNITS)


def test_fill_replaces_known_names() -> None:
    assert showcase.fill("a {{x}} b {{ y }}", {"x": "1", "y": "2"}) == "a 1 b 2"


def test_fill_rejects_unknown_names() -> None:
    with pytest.raises(KeyError, match="nope"):
        showcase.fill("{{nope}}", {})


def test_formatters_use_typographic_minus() -> None:
    assert showcase._fixed(-0.5, 2) == "−0.50"
    assert showcase._signed(0.0123) == "+0.012"
    assert showcase._count(347904000) == "347 904 000"
    assert showcase._sig(0.00034) == "0.00034"


def test_round_is_recursive() -> None:
    assert showcase._round({"a": [1.123456, {"b": 2.000049}]}, 3) == {"a": [1.123, {"b": 2.0}]}


def test_one_requires_a_unique_match() -> None:
    rows = [{"k": 1}, {"k": 1}, {"k": 2}]
    assert showcase._one(rows, k=2) == {"k": 2}
    with pytest.raises(KeyError):
        showcase._one(rows, k=1)


@pytest.mark.skipif(not HAVE_RESULTS, reason="results/*.json not present")
def test_site_builds_and_quotes_only_real_numbers(tmp_path: Path) -> None:
    results = showcase.load_results()
    facts = showcase.build_facts(results)
    # Spot-check that quoted numbers are the stored ones, rounded.
    assert facts["rtl_mismatches"] == "0"
    assert float(facts["low_uw"]) == pytest.approx(
        results["u3"]["configurations"][results["u3"]["datasets"]["nason"]["winner"]]["total_uw"],
        abs=0.005,
    )
    index = showcase.build_site(tmp_path)
    page = index.read_text(encoding="utf-8")
    assert "{{" not in page
    assert page.startswith("<!doctype html>")
    payload = re.search(r'id="nc-data">(.*?)</script>', page, re.S)
    assert payload is not None
    data = json.loads(payload.group(1).replace("<\\/", "</"))
    assert set(data) == {"pareto", "decoders", "stim", "aging", "rtl"}
    assert data["rtl"]["identical"] is True
    fragment = (tmp_path / "fragment.html").read_text(encoding="utf-8")
    assert fragment.startswith("<title>")
