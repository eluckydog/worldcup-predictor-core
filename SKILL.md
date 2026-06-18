---
name: 2026世界杯预测
description: "2026 FIFA World Cup daily predictor — dual-engine (Poisson + Causal Inference), multi-source odds fusion, Monte Carlo simulation, Bayesian belief tracking. Open source, MIT license, daily GitHub Actions updates."
license: MIT
version: 1.0.0
homepage: https://github.com/eluckydog/worldcup-predictor-core
---

# 2026 World Cup Predictor

Daily predictions for the 2026 FIFA World Cup using a dual-engine architecture.

## Quick Start

```bash
pip install -r requirements.txt
python main.py --home "Brazil" --away "Argentina" --use-odds
python scripts/daily_predict.py --save
```

## Architecture

main.py → Selector → Poisson / Causal Engine → MC Sim → Odds Fusion → Bayesian Update → Output

## Model Performance (MD1)

- 24 matches backtested, **45.8%** 1X2 accuracy (11/24)
- Mid-confidence threshold ≥0.60: **66.7%** (9 matches)
- 48 MD2 matches predicted, updated daily

## 9 Languages

🇬🇧 🇨🇳 🇯🇵 🇫🇷 🇩🇪 🇪🇸 🇸🇦 🇮🇳 🇰🇷

## License

MIT — free to fork, modify, and redistribute.
