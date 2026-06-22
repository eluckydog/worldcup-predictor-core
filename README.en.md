# ⚽ 2026 World Cup Daily Predictor

A **dual-engine fusion** of Poisson regression + causal inference + multi-source market odds.

**Predictions updated daily → [`predictions/`](./predictions/)**

### Quick Start

```bash
pip install -r requirements.txt

# Unified pipeline: DC → calibration → standings → stakes → MC → report
python scripts/run.py

# With Monte Carlo simulation (total draw projection)
python scripts/run.py --mc --save
```

### Model Performance Tracking

| Date | Round | Predicted | Correct | Accuracy |
|------|-------|-----------|---------|----------|
| 2026-06-18 | MD1 | 24 | 11 | 45.8% |
| 2026-06-22 | MD1+MD2 | 40 | 21 | **52.5%** |
| 2026-06-22 | MD1+MD2 (calibrated +10%) | 40 | 22 | **55.0%** |

| Historical MC backtest (2002-2022) | 288 matches | Baseline 62.8% | +10% Δ → **63.9%** |
| MC projection (2026 remaining 32) | 100k trials | P50 = 23 draws / **32.0%** | |

_Table updates after each matchday. Details in [`docs/MODELING.md`](docs/MODELING.md)._

**100% open source · MIT license**

---

## 📊 Qualification Status (after MD2)

| Stage | Status |
|-------|--------|
| Group winners locked | Mexico(A), Canada(B), Brazil(C), USA(D), Germany(E), Netherlands(F), Egypt(G), Spain(H), Norway(I), Argentina(J), Colombia(K), England(L) |
| Runners-up | S. Korea(A), Switzerland(B), Morocco(C), Australia(D), Côte d'Ivoire(E), Japan(F), Iran(G), Uruguay(H), France(I), Austria(J), Portugal(K), Ghana(L) |
| Best 3rd-placed (8 qualify) | Sweden(F), Scotland(C), Paraguay(D), Cape Verde(H), Belgium(G), DR Congo(K), Czech R.(A), Ecuador(E) |
| Matches remaining | 8 MD2 + 24 MD3 = 32 total |
| Can still qualify | Bosnia(1pt GD-3), Senegal(0pt GD-2), Jordan(0pt GD-2), Panama(0pt GD-1) |

➡ Full FIFA tiebreaker resolution (head-to-head mini-league) in [`core/bracket.py`](core/bracket.py).

---

## 🔮 MD2 + MD3 Predictions (32 upcoming matches)

_MD3 stakes analysis integrated: each match's group situation + knockout bracket incentive considered._

### MD2 (8 matches)

| Group | Home | Away | H | D | A | Calibrated | Pick |
|-------|------|------|---|---|---|-------------|------|
| I | Iraq | France | 40% | 25% | 35% | 36%/32%/32% | 🏠 Home |
| I | Norway | Senegal | 36% | 27% | 37% | 33%/34%/34% | ✈️ Away |
| J | Austria | Argentina | 34% | 22% | 44% | 31%/29%/40% | ✈️ Away |
| J | Jordan | Algeria | 46% | 29% | 25% | 42%/35%/23% | 🏠 Home |
| K | Portugal | Uzbekistan | 40% | 27% | 33% | 36%/34%/30% | 🏠 Home |
| K | Colombia | DR Congo | 43% | 28% | 29% | 39%/35%/26% | 🏠 Home |
| L | England | Ghana | 55% | 25% | 20% | 50%/32%/18% | 🏠 Home |
| L | Croatia | Panama | 44% | 28% | 28% | 40%/35%/25% | 🏠 Home |

### MD3 (24 matches) — with stakes analysis

| Group | Match | Scenario | Pick |
|-------|-------|----------|------|
| A | Czech Republic vs Mexico | Mexico locked 1st, Czech can't advance | 🏠 Home |
| A | South Africa vs South Korea | Korea needs win to advance | ✈️ Away |
| **B** | **Switzerland vs Canada** | **DRAW ADVANCES BOTH (both 4pts, tiebreaker)** | **🤝 Draw** |
| B | Qatar vs Bosnia and Herzegovina | Both eliminated if other result holds | ✈️ Away |
| C | Brazil vs Scotland | Brazil 4pts, Scotland 3pts. Brazil safe, Scotland needs help | 🏠 Home |
| **C** | **Morocco vs Haiti** | **Morocco 4pts, Haiti 0pts. Draw enough for Morocco** | **🤝 Draw** |
| D | USA vs Turkey | USA locked 1st, Turkey eliminated | 🏠 Home |
| D | Australia vs Paraguay | Both 3pts. Winner likely qualifies, loser fights 3rd | ✈️ Away |
| E | Germany vs Ecuador | Germany locked 1st, Ecuador out | 🏠 Home |
| E | Côte d'Ivoire vs Curaçao | CIV 3pts needs win to guarantee 2nd | 🏠 Home |
| F | Netherlands vs Tunisia | NED 4pts vs TUN 0pts | 🏠 Home |
| F | Sweden vs Japan | Both in contention (SWE 3pts, JPN 4pts) | 🏠 Home |
| **G** | **Belgium vs New Zealand** | **BEL 2pts, NZ 1pt. Draw helps both but need other result** | **🤝 Draw** |
| G | Iran vs Egypt | Iran (2pt) must win; Egypt (4pt) draw enough | ✈️ Away |
| H | Spain vs Uruguay | ESP 4pts, URU 2pts. Spain win = 1st | 🏠 Home |
| H | Saudi Arabia vs Cape Verde | Both can still qualify with win | ✈️ Away |
| I | France vs Norway | Both 3pts. Draw advances both (winner→3R1, runner-up→Spain) | 🏠 Home |
| I | Iraq vs Senegal | Both 0pts, must win to have any chance | 🏠 Home |
| J | Algeria vs Austria | Both can still qualify with win | ✈️ Away |
| J | Jordan vs Argentina | ARG locked 1st, JOR can qualify with win + help | ✈️ Away |
| K | Portugal vs Colombia | POR 1pt, COL 3pts. Portugal must win | 🏠 Home |
| K | Uzbekistan vs DR Congo | Both 0-1pts, winners could qualify as 3rd | 🏠 Home |
| L | Ghana vs Croatia | GHA 3pts, CRO 0pts. Ghana can secure 2nd with result | ✈️ Away |
| L | England vs Panama | ENG 3pts, PAN 0pts. England can lock 1st | 🏠 Home |

---

## ⚡ What Makes This Different

- **Dual-engine architecture** — Poisson and Causal models compete on every match; selector picks the best
- **Calibration-aware** — explicit draw calibration (+10% uniform bonus from 2002-2022 MC optimization)
- **Stakes analysis (LLM)** — MD3 group situations integrated (draw advances both, must-win, bracket incentive)
- **Monte Carlo simulation** — 100k trials for total draw projection; P50=23 draws / 32.0%
- **Multi-source odds fusion**  — 4 sources (500.com, JC SP, international, fallback) with juice removal
- **Transparent by design**  — every prediction with confidence score, engine selection, and full probability distribution
- **FIFA-compliant** — head-to-head tiebreakers, third-placed ranking, R32 bracket generation

---

## 🧠 Architecture

```
                           scripts/run.py
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                   ▼
        DC Predictions     Calibration        Stakes Analysis
        (main.py)         (+10% draw Δ)      (group scenario
              │                              + bracket incentive)
              ▼
        ┌─────────────────────────────────────┐
        │     Monte Carlo Simulator           │
        │  (100k trials, total draw forecast) │
        └─────────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │     Report (standings + qualifiers) │
        └─────────────────────────────────────┘
```

### Layer Architecture (see [`docs/MODELING.md`](docs/MODELING.md))

| Layer | Component | Purpose |
|-------|-----------|---------|
| 0 | DC Poisson | Dixon-Coles model, bivariate joint grid |
| 1 | Calibration | Uniform +10% draw probability boost |
| 2 | LLM Stakes | Group scenario analysis + bracket incentive |
| 3 | MC Simulation | 100k trial total draw projection |

## 🚀 All Commands

```bash
# Quick prediction (single match)
python main.py --home "Brazil" --away "Argentina"

# Full pipeline
python scripts/run.py --mc --save

# Traditional daily runner
python scripts/daily_predict.py --save
```

## 📁 Project Structure

```
├── main.py              # CLI entry (single match mode)
├── core/
│   ├── engine_poisson.py # Dixon-Coles bivariate Poisson
│   ├── engine_causal.py  # Causal inference (Double-ML/DAG)
│   ├── selector.py       # Dual-engine selector gate
│   ├── monte_carlo.py    # MC simulation engine
│   ├── calibration.py    # Uniform draw calibration (+10%)
│   ├── stakes.py         # MD3 group situation + bracket analysis
│   ├── bracket.py        # FIFA standings + R32 bracket
│   ├── fusion.py         # Multi-source odds fusion
│   └── bayesian.py       # Beta-Binomial belief tracking
├── data/
│   ├── matches.py        # 72 match tuples (single source of truth)
│   ├── worldcup.db       # 964 historical matches (1930-2022)
│   └── data_adapter.py   # SQLite query layer
├── scripts/
│   ├── run.py            # ★ Unified pipeline (recommended)
│   ├── daily_predict.py  # Original daily runner
│   └── full_analysis.py  # Full backtest suite
├── predictions/          # Daily prediction reports
├── docs/
│   └── MODELING.md       # Full modeling methodology
└── README.*.md           # Multi-language docs
```

## 🏆 Track Record (MD1+MD2)

| Metric | Value |
|--------|-------|
| Completed Matches | 40 |
| 1X2 Accuracy (baseline) | **52.5%** (21/40) |
| 1X2 Accuracy (calibrated +10%) | **55.0%** (22/40) |
| MD1 Accuracy | **45.8%** (11/24) |
| MD2 Accuracy | **62.5%** (10/16) |
| ≥0.60 confidence threshold | **62.5%** (10/16) |
| Historical MC backtest (288 matches) | 63.9% (+10% Δ) |
| Completed draw rate | **32.5%** (13/40) |

### Confidence Calibration

| Threshold | Matches | Accuracy |
|-----------|---------|----------|
| All 40 | 40 | 52.5% |
| ≥ 0.50 | 29 | 58.6% |
| ≥ 0.55 | 23 | 60.9% |
| **≥ 0.60** | **16** | **62.5%** |
| ≥ 0.65 | 11 | 54.5% |

## 💡 Draw Prediction Methodology

**Key insight:** Single-match "Draw pick" is rare (DC model dp 20-30% can't beat H or A). Draws are cumulative — the model predicts ~32% draw rate at tournament level, not individual draw picks.

See [`docs/MODELING.md`](docs/MODELING.md) for:
- Complete modeling methodology (4 layers)
- MC backtest validation (2002-2022, 288 matches)
- Historical draw rate analysis
- Calibration optimization

## ⚙️ Dependencies

- Python 3.10+
- `numpy`, `scipy`, `pandas`
- `sqlite3` (stdlib)
- `requests`
- `networkx`

## 📄 License

**MIT** — free for any use, including commercial.

---

> Built with historical data from [open-football](https://github.com/openfootball/world-cup). Live odds from 500.com.
