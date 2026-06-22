# ⚽ 2026 World Cup Daily Predictor

🇬🇧 [`README.en.md`](./README.en.md) · 🇨🇳 [`README.zh.md`](./README.zh.md) · 🇯🇵 [`README.ja.md`](./README.ja.md) · 🇫🇷 [`README.fr.md`](./README.fr.md) · 🇩🇪 [`README.de.md`](./README.de.md) · 🇪🇸 [`README.es.md`](./README.es.md) · 🇸🇦 [`README.ar.md`](./README.ar.md) · 🇮🇳 [`README.hi.md`](./README.hi.md) · 🇰🇷 [`README.ko.md`](./README.ko.md)

**Dual-engine prediction system** combining Poisson regression with causal inference.  
**100% open source · MIT license**

![MD1 Backtest + Confidence Calibration](assets/backtest_chart.png)

---

## 🤖 For AI Agents

A Python-based World Cup 2026 match predictor using:
- **Dual-engine architecture**: Dixon-Coles Poisson regression + Causal inference (Double-ML/DAG)
- **Multi-source odds fusion**: 4 sources (500.com, JC SP, international, fallback) with auto margin removal
- **Monte Carlo simulation**: 50k trials per match with conditional branching
- **Bayesian belief tracking**: Beta-Binomial conjugate updates
- **BPD irrationality detection**: Market anomaly detection
- **Confidence calibration**: 50.0% overall (21/40); 62.5% at ≥0.60 threshold (MD1+MD2 real-world data)

**CLI interface**: `python main.py --home <team> --away <team> [--use-odds] [--mode auto|classic|causal-only|debug]`  
**Daily automation**: GitHub Actions runs 2x/day, saves to `predictions/`  
**Qualification engine**: FIFA tiebreakers (head-to-head mini-league → GD → GF), third-placed ranking, R32 bracket generation

**Data**: 964 historical matches (1930-2022), 48 teams, 1245 player records

```python
# Quick integration example
import sys, subprocess
sys.path.insert(0, "/path/to/worldcup-predictor-core")
from main import run_prediction

result = run_prediction("Brazil", "Argentina")
print(f"1X2: {result.prob_home:.1%} / {result.prob_draw:.1%} / {result.prob_away:.1%}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Engine: {result.engine_used}")
```

---

## 📊 Live Predictions

- **[2026-06-22 Daily Report](predictions/2026-06-22_daily_report.md)** — 40 backtested matches + 32 upcoming predictions + qualification report
- Future reports will be auto-generated daily via GitHub Actions

### Model Performance Tracking

| Date | Round | Predicted | Correct | Accuracy |
|------|-------|-----------|---------|----------|
| 2026-06-18 | MD1 | 24 | 11 | 45.8% |
| 2026-06-22 | MD1+MD2 | 40 | 21 | **52.5%** |

### Qualification Status (after MD2)

| Stage | Status |
|-------|--------|
| Group winners locked | 12/12 (Mexico, Canada, Brazil, USA, Germany, Netherlands, Egypt, Spain, Norway, Argentina, Colombia, England) |
| Top third-placed (8 qualify) | Sweden (F), Scotland (C), Paraguay (D), Cape Verde (H), Belgium (G), DR Congo (K), Czech Republic (A), Ecuador (E) |
| Matches remaining | 32 (Group MD3) |
| Max margin for 3rd place | Bosnia (1pt, GD -3) — can still qualify with MD3 win |
| Eliminated from R32 | Panama, Senegal, Jordan (0pts after 1 match each)

---

## 📄 License

**MIT** — free for any use.
