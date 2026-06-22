# ⚽ 2026 World Cup Daily Predictor

A **dual-engine fusion** of Poisson regression + causal inference + multi-source market odds.

**Predictions updated daily via GitHub Actions → [`predictions/`](./predictions/)**

- **[2026-06-22 Daily Report](predictions/2026-06-22_daily_report.md)** — 40 backtested matches + 32 upcoming predictions + qualification report

### Model Performance Tracking

| Date | Round | Predicted | Correct | Accuracy |
|------|-------|-----------|---------|----------|
| 2026-06-18 | MD1 | 24 | 11 | 45.8% |
| 2026-06-22 | MD1+MD2 | 40 | 21 | **52.5%** |

_This table updates after each matchday. At tournament end, it becomes a complete model validation report._

**100% open source · MIT license**

---

## 📊 Qualification Status (after MD2)

| Stage | Status |
|-------|--------|
| Group winners locked | 12/12 |
| Top third-placed (8 qualify) | Sweden (F), Scotland (C), Paraguay (D), Cape Verde (H), Belgium (G), DR Congo (K), Czech Republic (A), Ecuador (E) |
| Matches remaining | 32 (Group MD3) |

Full FIFA tiebreaker resolution (head-to-head mini-league) implemented in [`core/bracket.py`](core/bracket.py).

---

## 📊 MD1+MD2 Results — Prediction vs Actual

### Matchday 1 (11/24 correct, 45.8%)

| Match | Pred | Actual | Result |
|-------|------|--------|--------|
| Mexico vs South Africa | H | H | ✅ |
| South Korea vs Czech Republic | H | H | ✅ |
| Qatar vs Switzerland | H | D | ❌ |
| Canada vs Jordan | A | D | ❌ |
| Brazil vs Morocco | H | D | ❌ |
| Scotland vs Haiti | H | H | ✅ |
| USA vs Paraguay | H | H | ✅ |
| Australia vs Turkey | H | H | ✅ |
| Germany vs Curaçao | H | H | ✅ |
| Côte d'Ivoire vs Ecuador | A | H | ❌ |
| Sweden vs Tunisia | H | H | ✅ |
| Japan vs Netherlands | A | D | ❌ |
| Belgium vs Egypt | H | D | ❌ |
| Iran vs New Zealand | A | D | ❌ |
| Spain vs Cape Verde | H | D | ❌ |
| Saudi Arabia vs Uruguay | A | D | ❌ |
| France vs Senegal | H | H | ✅ |
| Norway vs Iraq | A | H | ❌ |
| Argentina vs Algeria | H | H | ✅ |
| Austria vs Jordan | A | H | ❌ |
| Portugal vs DR Congo | H | D | ❌ |
| Uzbekistan vs Colombia | A | A | ✅ |
| England vs Croatia | H | H | ✅ |
| Ghana vs Panama | A | H | ❌ |

> **Total: 11/24 correct (45.8%)**

### Matchday 2 (10/16 correct, 62.5%)

| Match | Pred | Actual | Result |
|-------|------|--------|--------|
| Mexico vs South Korea | H | H | ✅ |
| Czech Republic vs South Africa | H | D | ❌ |
| Canada vs Qatar | A | H | ❌ |
| Switzerland vs Bosnia and Herzegovina | A | H | ❌ |
| Brazil vs Haiti | H | H | ✅ |
| Scotland vs Morocco | A | A | ✅ |
| USA vs Australia | H | H | ✅ |
| Turkey vs Paraguay | A | A | ✅ |
| Germany vs Côte d'Ivoire | H | H | ✅ |
| Ecuador vs Curaçao | H | D | ❌ |
| Netherlands vs Sweden | H | H | ✅ |
| Tunisia vs Japan | A | A | ✅ |
| Belgium vs Iran | H | D | ❌ |
| New Zealand vs Egypt | H | A | ❌ |
| Spain vs Saudi Arabia | H | H | ✅ |
| Uruguay vs Cape Verde | H | D | ❌ |

> **Total: 10/16 correct (62.5%)**

### Combined: **21/40 correct (52.5%)**

---

## 🔮 MD3 Predictions (Upcoming)

| Match | Pick | Confidence |
|-------|------|------------|
| Mexico vs South Korea | H | 46% |
| Czech Republic vs South Africa | H | 49% |
| Canada vs Qatar | A | 42% |
| Switzerland vs Bosnia and Herzegovina | A | 41% |
| Brazil vs Scotland | H | 69% |
| Morocco vs Haiti | H | 35% |
| USA vs Australia | H | 44% |
| Paraguay vs Turkey | H | 42% |
| Germany vs Côte d'Ivoire | H | 62% |
| Ecuador vs Curaçao | H | 42% |
| Sweden vs Japan | H | 47% |
| Netherlands vs Tunisia | H | 76% |
| Belgium vs Iran | H | 51% |
| New Zealand vs Egypt | H | 40% |
| Spain vs Saudi Arabia | H | 79% |
| Uruguay vs Cape Verde | H | 45% |
| France vs Norway | H | 61% |
| Senegal vs Iraq | A | 60% |
| Argentina vs Austria | H | 44% |
| Algeria vs Jordan | A | 46% |
| Portugal vs Uzbekistan | H | 45% |
| DR Congo vs Colombia | A | 42% |

> **H = Home Win · D = Draw · A = Away Win**

---

## ⚡ What Makes This Different

Most football prediction projects pick one model and call it a day. This one doesn't:

- **Dual-engine architecture** — Poisson and Causal models compete on every match; the selector picks the best
- **Market-aware** — 4-source odds fusion (500.com, JC SP, international, fallback) with automatic juice removal and credibility weighting
- **Calibration-aware** — we measure confidence vs accuracy explicitly. MD1 showed 66.7% accuracy at ≥0.60 confidence threshold (9/24 matches), and we publish the full calibration table
- **Transparent by design** — every prediction comes with a confidence score, an engine selection reason, and a probability distribution. No black box
- **Daily automation** — GitHub Actions runs predictions twice a day with no human intervention

**This is an open-source engineering experiment.** 964 historical matches, 50k MC simulations per match, BPD irrationality detection, Bayesian belief tracking, FIFA-compliant qualification & bracket engine — all in one Python repo. Fork it, break it, make it better.

---

## 🧠 Architecture

```
┌──────────────────────────────────────────────────────┐
│                   main.py (CLI)                       │
├──────────────────────────────────────────────────────┤
│                    Selector                           │
│         (Poisson ↔ Causal dual-engine gate)          │
├──────────────────────────────────────────────────────┤
│   ┌─────────────────┐   ┌─────────────────────────┐  │
│   │ Engine: Poisson  │   │   Engine: Causal        │  │
│   │ (Dixon-Coles)    │   │   (Double-ML / DAG)    │  │
│   └────────┬─────────┘   └──────────┬──────────────┘  │
│            │                        │                  │
├────────────┴────────────────────────┴──────────────────┤
│                  Monte Carlo Simulator                  │
│               (50k trials per match)                    │
├────────────────────────────────────────────────────────┤
│   Odds Fusion           │   Bayesian Update            │
│   (Multi-source odds    │   (Beta-Binomial             │
│    → λ bias adjustment)  │    belief tracking)          │
├────────────────────────────────────────────────────────┤
│   Data: 964 historical matches (1930-2022) + live odds │
└────────────────────────────────────────────────────────┘
```

## ✨ Features

- 🎯 **Dual-engine selector** — dynamically chooses between Poisson (classic) and Causal models per match based on data quality
- 🔄 **50k Monte Carlo simulation** — conditional branching sampling for goal probability distributions
- 📊 **Multi-source odds fusion** — integrates 500.com average, JC SP, and international odds with automatic juice removal
- 🧠 **Bayesian belief tracking** — Beta-Binomial conjugate updates as the tournament progresses
- 🔍 **Irrationality detection** — BPD (Behavioral Pattern Decoder) for detecting market anomalies
- 📅 **Historical data** — 971 matches from every World Cup since 1930

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Single match prediction
python main.py --home "Brazil" --away "Argentina"

# With odds correction
python main.py --home "Mexico" --away "South Korea" --use-odds

# Full daily prediction
python scripts/daily_predict.py --save
```

## 📋 Example Output

```
Mexico vs South Korea | mode=auto
┌──────────────────────────────────────┐
│ 引擎选择: poisson(主0.85) + causal(辅0.15) │
│ 预期进球: 1.95 - 1.58                   │
│ 概率分布: H 46.0% / D 23.0% / A 31.0%  │
│ 最可能比分: 1:1 (p=10%)                 │
│ 置信度: 0.74                            │
└──────────────────────────────────────┘
```

## 📁 Project Structure

```
├── main.py              # CLI entry point
├── core/                # Prediction engine modules
│   ├── bayesian.py      # Bayesian belief tracking
│   ├── data_types.py    # Shared types & odds records
│   ├── engine_causal.py # Causal inference (Double-ML)
│   ├── engine_poisson.py# Dixon-Coles bivariate Poisson
│   ├── fusion.py        # Dual-engine fusion & odds
│   ├── irrationality.py # BPD pattern detector
│   ├── knox_client.py   # Knox.chat LLM integration
│   ├── monte_carlo.py   # 50k MC simulation
│   ├── pathway.py       # Entity-relation pathway scoring
│   ├── selector.py      # Dual-engine selector gate
│   └── team_resolver.py # Name resolution & aliases
├── data/
│   ├── data_adapter.py  # SQLite query layer
│   ├── importer.py      # Historical data parser
│   ├── odds_provider.py # Multi-source odds fetcher
│   ├── worldcup.db      # 971 matches (1930-2022)
│   └── wc2026.json      # Tournament group layout
├── scripts/
│   ├── daily_predict.py # Automated daily runner
│   ├── full_analysis.py # Full backtest suite
│   └── verify_odds.py   # Odds data validation
├── predictions/         # Daily prediction reports
├── README.md            # Landing page (bilingual)
├── README.en.md         # English docs
├── README.zh.md         # 中文文档
└── LICENSE              # MIT
```

## 🏆 Track Record (MD1+MD2)

| Metric | Value |
|--------|-------|
| Completed Matches | 40 |
| 1X2 Accuracy | **52.5%** (21/40) |
| MD1 Accuracy | **45.8%** (11/24) |
| MD2 Accuracy | **62.5%** (10/16) |
| Mid-confidence (0.50-0.69) | **58.6%** (17/29) |

> 📌 **Note:** Draw rate remains high (~50% of incorrects were draws).

### Confidence Calibration / Accuracy by Threshold (MD1 only)

Accuracy **improves significantly** when low-confidence predictions are excluded:

| Threshold | Matches Kept | Accuracy | Note |
|-----------|-------------|----------|------|
| All 24 | 24 | 45.8% | Baseline |
| ≥ 0.40 | 21 | 52.4% | Drops 3 (all wrong) |
| ≥ 0.50 | 18 | 50.0% | Drops 6 (2/6 correct) |
| ≥ 0.55 | 17 | 52.9% | Drops 7 (2/7 correct) |
| **≥ 0.60** | **9** | **66.7%** | Best threshold — 2/3 accuracy |
| ≥ 0.65 | 6 | 50.0% | Drops 18 (8/18 correct) |
| ≥ 0.70 | 4 | 25.0% | Worst — model was most confident on upsets |

> **Key insight (MD1):** Highest-confidence predictions (≥0.70) were the least accurate — the model was confidently wrong on Brazil-Morocco, Saudi-Uruguay, Japan-Netherlands upsets.

### Full MD1+MD2 Calibration (40 matches)

| Threshold | Matches Kept | Accuracy |
|-----------|-------------|----------|
| All 40 | 40 | 52.5% |
| ≥ 0.40 | 36 | 55.6% |
| ≥ 0.50 | 29 | 58.6% |
| ≥ 0.55 | 23 | 60.9% |
| ≥ 0.60 | 16 | **62.5%** |
| ≥ 0.65 | 11 | 54.5% |
| ≥ 0.70 | 6 | 33.3% |

> **Key insight (MD1+MD2):** The ≥0.60 threshold remains the best calibration point at 62.5% across 16 matches.

---

## 🧪 How the Selector Works

The selector evaluates **4 dimensions** per match:

| Dimension | What it checks |
|-----------|---------------|
| Historical data volume | Does this matchup have enough past games? |
| DAG coverage | Does the causal graph cover relevant factors? |
| Poisson fit | How well do goal distributions fit? |
| Structural breaks | Have recent results shifted dynamics? |

**Modes:** `classic` (Poisson only) · `causal-only` · `auto` (recommended) · `debug`

## 📈 How Odds Fusion Works

```
Raw Odds → Remove Juice → Normalize to Probabilities
→ Weight by source credibility → Fuse → Convert to λ bias
→ Adjust Poisson λ for each team → Run MC simulation
```

| Source | Vig | Weight |
|--------|-----|--------|
| 500.com average | ~6% | 0.94 (high) |
| JC SP (竞彩SP) | ~13% | 0.89 (low) |
| Foreign (international) | varies | varies |
| Simulated (fallback) | N/A | fallback only |

## 🔧 Dependencies

- Python 3.10+
- `numpy`, `scipy`, `pandas`
- `sqlite3` (stdlib)
- `requests`
- `networkx`

## 📄 License

**MIT** — free for any use, including commercial.

---

> Built with historical data from [open-football](https://github.com/openfootball/world-cup). Live odds from 500.com.
