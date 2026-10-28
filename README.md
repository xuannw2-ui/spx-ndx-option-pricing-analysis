# SPX/NDX Option Pricing and eSSVI Analysis

This Python project studies SPX and NDX index options in two stages:

1. invert Black–Scholes prices to obtain implied volatility; and
2. fit eSSVI volatility surfaces and derive risk-neutral-density measures.

It also contains an independent implied-volatility robustness analysis and
scripts for producing paper-ready Excel, chart, and Word outputs.

## Data availability

Bloomberg course files and OptionMetrics datasets are not included. They may
be subject to course or vendor licensing restrictions. Obtain permission from
the data owner before sharing either raw data or row-level derived datasets.

The small Part 1 workbooks should be placed in `data/`:

```text
data/
├── Final Project--Part1--SPX--Data.xlsx
└── Final Project--Part1--NDX--Data.xlsx
```

The long-horizon scripts expect a separate data root with this layout:

```text
<data-root>/
├── OptionMetrics_SPX_NDX_2018/
│   ├── opprcd2018.csv
│   └── secprd2018.csv
├── ... annual folders through 2024 ...
└── OptionMetrics_SPX_NDX_auxiliary_2018_2024/
    ├── idxdvd.csv
    ├── zerocd.csv
    └── vix_2018_2024.csv
```

## Installation

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

Run the Part 1 Black–Scholes implied-volatility analysis:

```bash
python part1_analysis.py --data-dir data --output-dir outputs
```

Run the 2018–2024 eSSVI and risk-neutral-density analysis:

```bash
python long_horizon_essvi_analysis.py \
  --data-root "/path/to/optionmetrics-data"
```

Run the independent implied-volatility robustness analysis after the
long-horizon analysis:

```bash
python iv_robustness_analysis.py \
  --data-root "/path/to/optionmetrics-data"
```

The scripts write generated files beneath `outputs/`, which is intentionally
excluded from version control.

To update an existing paper document with generated results:

```bash
python update_paper_with_essvi_rnd.py \
  --source "/path/to/source-paper.docx"
```

## Method summary

- Calls and puts are handled separately.
- Quote midpoint is `(Bid + Ask) / 2`.
- Black–Scholes includes continuous dividend yield.
- Implied volatility uses Newton iterations with a Brent fallback.
- Missing or zero quotes remain missing rather than being replaced with zero.
- The eSSVI analysis uses monthly snapshots and standard target tenors.
- Risk-neutral-density statistics and robustness diagnostics are exported for
  downstream reporting.

## Tests

The unit tests do not require proprietary market data:

```bash
python -m unittest discover -s tests -v
```

## Repository contents

- `part1_analysis.py`: Black–Scholes pricing and Part 1 analysis.
- `essvi_rnd_analysis.py`: reusable eSSVI and density functions.
- `long_horizon_essvi_analysis.py`: 2018–2024 eSSVI/RND pipeline.
- `iv_robustness_analysis.py`: independent IV validation.
- `write_*_report.py`: report-generation scripts.
- `tests/`: unit tests for pricing and IV inversion.

## License and data notice

No license is granted for third-party market data. This repository publishes
analysis code only; users are responsible for complying with Bloomberg,
OptionMetrics, course, and institutional data-use terms.
