# ⚽ 2026 World Cup Daily Predictor

**Fusion à double moteur** — régression de Poisson + inférence causale + moteur de qualification conforme FIFA + cotes multi-sources

**Prédictions mises à jour quotidiennement via GitHub Actions → [`predictions/`](./predictions/)**

**100% open source · Licence MIT**

---

## ⚡ Ce qui le rend différent

La plupart des projets de prédiction football utilisent un seul modèle. Pas celui-ci :

- **Architecture à double moteur** — Poisson et Causal s'affrontent sur chaque match ; le sélecteur choisit le meilleur
- **Conscient du marché** — Fusion de 4 sources de cotes (500.com, JC SP, internationales, fallback) avec élimination automatique de la marge et pondération par crédibilité
- **Calibration transparente** — Nous mesurons explicitement la relation confiance/précision. MD1 : 66.7% de précision au seuil ≥0.60 (9/24 matchs), tableau complet publié
- **Transparent par conception** — Chaque prédiction affiche un score de confiance, la raison du choix du moteur, et une distribution de probabilité. Pas de boîte noire
- **Automatisation quotidienne** — GitHub Actions exécute les prédictions 2 fois par jour, sans intervention humaine

**C'est une expérience d'ingénierie open source.** 971 matchs historiques, 50k simulations Monte Carlo par match, détection d'irrationalité BPD, suivi bayésien des croyances — tout dans un seul dépôt Python. Forkez-le, cassez-le, améliorez-le.

Documentation complète (anglais) : [`README.en.md`](./README.en.md)

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

## ✨ Fonctionnalités

- 🎯 **Sélecteur double moteur** — choisit dynamiquement entre Poisson et Causal selon la qualité des données
- 🔄 **50k simulations Monte Carlo** — échantillonnage par branchements conditionnels pour les distributions de buts
- 📊 **Fusion multi-sources** — cotes 500.com, JC SP, internationales avec élimination de la marge
- 🧠 **Suivi bayésien** — mise à jour par conjugaison Beta-Binomiale au fil du tournoi
- 🔍 **Détection d'irrationalité** — BPD pour repérer les anomalies de marché
- 📅 **Données historiques** — 971 matchs de toutes les Coupes du Monde depuis 1930

## 🚀 Démarrage rapide

```bash
pip install -r requirements.txt

# Prédiction d'un match
python main.py --home "Brazil" --away "Argentina"

# Avec correction par les cotes
python main.py --home "Mexico" --away "South Korea" --use-odds

# Prédiction quotidienne complète
python scripts/daily_predict.py --save
```

## 📄 Licence

**MIT** — utilisation libre, y compris commerciale.

---

> Données historiques : [open-football](https://github.com/openfootball/world-cup). Cotes : 500.com.
