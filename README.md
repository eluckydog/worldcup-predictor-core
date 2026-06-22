# ⚽ 2026 World Cup Predictor — Unified Pipeline

🇬🇧 [`README.en.md`](./README.en.md) · 🇨🇳 [`README.zh.md`](./README.zh.md) · 🇯🇵 [`README.ja.md`](./README.ja.md) · 🇫🇷 [`README.fr.md`](./README.fr.md) · 🇩🇪 [`README.de.md`](./README.de.md) · 🇪🇸 [`README.es.md`](./README.es.md) · 🇸🇦 [`README.ar.md`](./README.ar.md) · 🇮🇳 [`README.hi.md`](./README.hi.md) · 🇰🇷 [`README.ko.md`](./README.ko.md)

**Dual-engine prediction system** combining Dixon-Coles Poisson regression with causal inference.  
**One command: DC → calibration → standings → stakes → MC → report**  
**100% open source · MIT license**

---

## Quick Start

```bash
pip install -r requirements.txt

# Full prediction report (72 matches, 2.5s)
python scripts/run.py

# With Monte Carlo simulation of total draws
python scripts/run.py --mc

# Save report to predictions/
python scripts/run.py --save --mc
```

## Current Performance (MD1+MD2, 40 matches)

| Metric | Value |
|--------|-------|
| Baseline accuracy | 52.5% (21/40) |
| Calibrated (+10% draw bonus) | 55.0% (22/40) |
| ≥0.60 confidence threshold | **62.5%** (10/16) |
| 2002-2022 MC backtest (288 matches) | 63.9% (+10%Δ vs 62.8% baseline) |
| Completed draw rate | **32.5%** (vs historical 24.6%) |

> MC projection: P50 = **23 draws / 32.0%** (100k trials) · Full docs: [`docs/MODELING.md`](docs/MODELING.md)

## Pipeline

```
python scripts/run.py
  ├─ data/matches.py           ← 72 matches (single source of truth)
  ├─ main.run_prediction()     ← DC Poisson + causal selector
  ├─ core/calibration.py       ← +10% uniform draw bonus
  ├─ core/stakes.py            ← MD3 group situation + bracket incentive
  ├─ core/monte_carlo.py       ← 100k trials for total draw projection
  └─ Full report → stdout / predictions/
```

## Project Structure

```
├── main.py              # CLI entry (single match)
├── core/
│   ├── engine_poisson.py # Dixon-Coles bivariate Poisson
│   ├── engine_causal.py  # Causal inference (Double-ML/DAG)
│   ├── selector.py       # Dual-engine gate
│   ├── monte_carlo.py    # 50k MC simulation per match
│   ├── calibration.py    # Uniform +10% draw calibration
│   ├── stakes.py         # MD3 group situation + bracket analysis
│   ├── bracket.py        # FIFA-compliant standings + R32 bracket
│   ├── fusion.py         # Odds fusion (4-source)
│   ├── bayesian.py       # Beta-Binomial belief tracking
│   └── irrationality.py  # Market anomaly detection
├── data/
│   ├── matches.py        # 72 match tuples (single source)
│   ├── worldcup.db       # 964 historical matches (1930-2022)
│   └── data_adapter.py   # SQLite query layer
├── scripts/
│   ├── run.py            # ★ ONE-COMMAND PIPELINE
│   ├── daily_predict.py  # Original daily runner (reads data/matches.py)
│   ├── full_analysis.py  # Full backtest suite
│   └── verify_odds.py    # Odds data validation
├── predictions/          # Daily prediction reports
├── docs/
│   └── MODELING.md       # Full modeling methodology (4 layers + draw theory)
└── README.md             # ← this file
```

## Qualification Status (after MD2, 40 of 72 matches)

| Stage | Status |
|-------|--------|
| Group winners locked | 12/12 (Mexico, Canada, Brazil, USA, Germany, Netherlands, Egypt, Spain, Norway, Argentina, Colombia, England) |
| Runners-up | S. Korea, Switzerland, Morocco, Australia, Côte d'Ivoire, Japan, Iran, Uruguay, France, Austria, Portugal, Ghana |
| Best 3rd-placed (8 qualify) | Sweden(F), Scotland(C), Paraguay(D), Cape Verde(H), Belgium(G), DR Congo(K), Czech R.(A), Ecuador(E) |
| MD3 remaining | 24 matches · 8 MD2 · 32 total |

## License

**MIT** — free for any use.
