# The results site: design notes

`make site` writes `build/site/index.html`: one self-contained page that a
recruiter, reviewer or another lab can open from a link, with interactive
versions of the project's main results. GitHub Actions publishes the same page
to GitHub Pages on every push that changes a result (`.github/workflows/pages.yml`).

## What it is for

The README holds every number, but as tables. The page answers the research
question in the order a reader asks it: what the chain costs, whether it
decodes as well, whether it survives stimulation and aging, and whether it was
actually built. A reader who stops after the first screen still gets the
answer: 0.48 µW per channel, the same GRU R² as the 19.6 µW chain, and a
bit-exact RTL implementation.

## The rule it keeps

Rule 1 of the project (never invent results) applies to the page too, so the
page cannot hold a typed-in number:

- `site/template.html` holds the prose, with `{{name}}` placeholders wherever a
  result is quoted.
- `neurochain.showcase.build_facts` reads each one from `results/*.json` and
  formats it. A placeholder with no fact stops the build (`KeyError`), so the
  prose cannot quote something the results do not contain.
- The charts draw arrays copied from the same files into a JSON block in the
  page. The JavaScript only selects, scales and draws. It never recomputes a
  result.

When a unit is rerun, `make site` picks up the new numbers with no edits.

## Choices

| Choice | Alternatives | Why |
|---|---|---|
| One static HTML file, no server | Streamlit, Dash, marimo | A link has to open instantly on a phone. A Python server costs money and goes to sleep, and in-browser Python takes seconds to load. |
| Hand-written SVG charts | Plotly, D3, Vega-Lite | About 500 lines with no dependency. A charting library would weigh more than the whole page (123 kB). |
| Charts select from precomputed results | Sliders that rerun the power model in JavaScript | A JavaScript copy of the power model would be a second implementation that could drift from `neurochain.power`. Changing a constant stays a `make u3-key` job. |
| One accent, the selected chain | A hue per series | Cobalt marks the selected 0.48 µW chain, in the hero specs, the tables and every figure. Everything else is a neutral ink step, and shapes or line weights tell series apart, so the eye lands on the result first. |

## Visual direction

The look was reworked with [unslop](https://github.com/heydamir/unslop) by
Damir Sokolovsky (CC BY 4.0), a checklist for taking the generated-template look
out of an interface. Its audit found headings at weight 700, uppercase tracked
labels, a stacked hero with no visual, rows of equal stat tiles, off-grid
spacing and four hues in one chart. The direction chosen from the product's own
world:

- **Product:** an open project that finds the lowest-power neural sensing chain
  that still decodes, and builds it in RTL.
- **Character:** exact, measured, plainly confident.
- **World:** chip datasheets and bench test reports: paper, black ink,
  numbered figures, test conditions.
- **Moment:** Figure 1, the Pareto plot, in the hero, with the selected chain
  as the one coloured mark.
- **Signature:** a mono "Conditions" line under every figure, and spec tables
  with Parameter / Value / Conditions columns, as a datasheet has.

Type is IBM Plex Sans with IBM Plex Mono for figures, weights 400 to 600.
Neutrals are unslop's "ink" ramp, spacing sits on a 4 px grid, and the only
motion is a 140 ms colour change on hover.

## Sections and their sources

| Section | Source |
|---|---|
| Where the power goes | `u3_metrics.json` → `configurations` (Nason baseline, winner, raw stream) |
| Figure 1, accuracy against power (hero) | `u3_metrics.json` → `configurations`, `datasets.*.pareto` |
| Decoders | `u4_metrics.json` → `headline`, `results`, `example` |
| Stimulation | `u5_metrics.json` → `in_vivo.trace`, `decoding.rows` (same array, GRU) |
| Aging | `u6_metrics.json` → `decoding.rows` (baseline), `key.cross_session`, `variations` |
| RTL | `u7_metrics.json` → `key`, `synthesis.sweep`, `example` |
| Groundwork | `u0` to `u2` metrics |

## Checks

`tests/test_showcase.py` builds the page from the committed results, checks
that no placeholder is left, that the embedded data parses, and that the RTL
example on the page is identical between Python and RTL.
