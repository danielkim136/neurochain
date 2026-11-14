"""Print the U4 key numbers, optionally with one decoder setting changed.

With no options this prints, from ``results/u4_metrics.json``, how U3's chosen
chain compares with the conventional chain under each decoder. With an option
it refits the affected decoder on those two chains and prints the new scores,
which is how the prediction checks in ``docs/U4_design.md`` are answered::

    uv run python scripts/u4_key_number.py --kalman-delays 0 --dataset nomad_session1
    uv run python scripts/u4_key_number.py --ridge-lags 20 --dataset nomad_session1
    uv run python scripts/u4_key_number.py --gru-hidden 8 --dataset nason
    uv run python scripts/u4_key_number.py --seed 1 --dataset nason
    uv run python scripts/u4_key_number.py --wiener-history 1.0 --dataset nomad_session2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

from neurochain.analysis.u4 import U4Config, compute_u4, default_sets
from neurochain.paths import results_dir


def _score(row: dict[str, Any]) -> str:
    return f"{row['r2']:.3f} [{row['r2_low']:.3f}, {row['r2_high']:.3f}]"


def show(title: str, summary: dict[str, Any], datasets: list[str]) -> None:
    """Print each decoder's score on the conventional chain and on U3's choice."""
    print(title)
    rows = {(r["dataset"], r["label"], r["decoder"]): r for r in summary["results"]}
    for name in datasets:
        head = summary["headline"][name]
        print(f"  {name}: conventional = {head['conventional']}; U3's choice = {head['winner']}")
        for decoder, entry in head["decoders"].items():
            base = rows[name, head["conventional"], decoder]
            best = rows[name, head["winner"], decoder]
            diff = entry["winner_minus_conventional"]
            delay = f", delay {best['delay_bins']} bins" if "delay_bins" in best else ""
            delay += f", penalty {best['penalty']}" if "penalty" in best else ""
            print(
                f"    {decoder:7s} R² {_score(base)} -> {_score(best)}"
                f"  difference {diff['difference']:+.3f} [{diff['low']:+.3f}, {diff['high']:+.3f}]"
                f"{delay}"
            )
    cross = summary.get("cross_session")
    if cross:
        print(f"  across sessions, {cross['days_apart']:.0f} days apart:")
        for r in cross["rows"]:
            print(
                f"    {r['decoder']:7s} fitted on {r['train']}, tested on {r['test']}: "
                f"R² {_score(r)}"
            )


def main() -> None:
    """Print the committed result, and a changed one if any option is given."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--dataset", default="nason", help="dataset to refit on (nason)")
    parser.add_argument("--kalman-delays", type=int, nargs="+", help="delays to choose from")
    parser.add_argument("--no-displacement", action="store_true", help="velocity-only state")
    parser.add_argument("--ridge-lags", type=int, help="bins of history for the ridge readout")
    parser.add_argument("--wiener-history", type=float, help="Wiener history, seconds (0.5)")
    parser.add_argument("--seed", type=int, help="seed for the GRU and the bootstrap (0)")
    parser.add_argument("--gru-hidden", type=int, help="hidden units in the GRU (64)")
    args = parser.parse_args()

    committed = json.loads((results_dir() / "u4_metrics.json").read_text())
    show("U4 key numbers (committed)", committed, list(committed["headline"]))

    config = U4Config()
    decoders: list[str] = []
    if args.kalman_delays or args.no_displacement:
        decoders.append("kalman")
        config = replace(
            config,
            kalman_delays=tuple(args.kalman_delays or config.kalman_delays),
            kalman_displacement=not args.no_displacement,
        )
    if args.ridge_lags:
        decoders.append("ridge")
        config = replace(config, ridge_lags=args.ridge_lags)
    if args.wiener_history:
        decoders.append("wiener")
        config = replace(config, wiener_history_s=args.wiener_history)
    if args.gru_hidden or args.seed is not None:
        decoders.append("gru")
        config = replace(
            config,
            gru_hidden=args.gru_hidden or config.gru_hidden,
            seed=config.seed if args.seed is None else args.seed,
        )
    if not decoders:
        return

    u3_summary = json.loads((results_dir() / "u3_metrics.json").read_text())
    sets, _, _ = default_sets(u3_summary)
    data, _ = sets[args.dataset]
    head = committed["headline"][args.dataset]
    changed = compute_u4(
        u3_summary,
        config=replace(config, decoders=tuple(decoders)),
        sets={args.dataset: (data, (head["conventional"], head["winner"]))},
    )
    changes = ", ".join(
        f"{k} = {v}" for k, v in vars(args).items() if v not in (None, False) and k != "dataset"
    )
    print()
    show(f"Changed: {changes}", changed, [args.dataset])


if __name__ == "__main__":
    main()
