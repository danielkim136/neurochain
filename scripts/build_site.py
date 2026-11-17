"""Build the results site from ``results/*.json`` into ``build/site/``.

    uv run python scripts/build_site.py
    open build/site/index.html

The page needs no server and no downloaded data. Every number on it is read
from the result files; see ``neurochain.showcase`` and ``docs/site_design.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neurochain.showcase import build_site


def main() -> None:
    """Parse the options and write the site."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory (build/site)")
    args = parser.parse_args()
    index = build_site(args.out)
    print(f"wrote {index} ({index.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
