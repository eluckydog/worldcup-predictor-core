# ⚽ 2026 World Cup Daily Predictor

**Dual-engine prediction system** combining Poisson regression with causal inference.  
**100% open source · MIT license · [9 languages](#read--阅读)**

---

## 🤖 For AI Agents

A Python-based World Cup 2026 match predictor using:
- **Dual-engine architecture**: Dixon-Coles Poisson regression + Causal inference (Double-ML/DAG)
- **Multi-source odds fusion**: 4 sources (500.com, JC SP, international, fallback) with auto margin removal
- **Monte Carlo simulation**: 50k trials per match with conditional branching
- **Bayesian belief tracking**: Beta-Binomial conjugate updates
- **BPD irrationality detection**: Market anomaly detection
- **Confidence calibration**: Measured 66.7% accuracy at ≥0.60 threshold (MD1 real-world data)

**CLI interface**: `python main.py --home <team> --away <team> [--use-odds] [--mode auto|classic|causal-only|debug]`  
**Daily automation**: GitHub Actions runs 2x/day, saves to `predictions/`  
**Data**: 971 historical matches (1930-2022), 48 teams, 1245 player records

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

## 📖 Read / 阅读

| Language | File |
|----------|------|
| 🇬🇧 **English** (full docs) | [`README.en.md`](./README.en.md) |
| 🇨🇳 **中文** | [`README.zh.md`](./README.zh.md) |
| 🇯🇵 **日本語** | [`README.ja.md`](./README.ja.md) |
| 🇫🇷 **Français** | [`README.fr.md`](./README.fr.md) |
| 🇩🇪 **Deutsch** | [`README.de.md`](./README.de.md) |
| 🇪🇸 **Español** | [`README.es.md`](./README.es.md) |
| 🇸🇦 **العربية** | [`README.ar.md`](./README.ar.md) |
| 🇮🇳 **हिन्दी** | [`README.hi.md`](./README.hi.md) |
| 🇰🇷 **한국어** | [`README.ko.md`](./README.ko.md) |

---

## 📊 Live Predictions

**Daily prediction reports → [`predictions/`](./predictions/)**

---

## 📄 License

**MIT** — free for any use.
