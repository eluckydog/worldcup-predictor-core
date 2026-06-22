# ⚽ 2026 World Cup Daily Predictor

**Dual-Engine Fusion** — Poisson-Regression + kausale Inferenz + FIFA-konformes Qualifikations- & K.-o.-System + Multi-Quellen-Quoten

**Täglich aktualisiert via GitHub Actions → [`predictions/`](./predictions/)**

**100% Open Source · MIT Lizenz**

---

## ⚡ Was es anders macht

Die meisten Fußball-Prognoseprojekte setzen auf ein einziges Modell. Dieses nicht:

- **Dual-Engine-Architektur** — Poisson und kausales Modell treten auf jedes Spiel an; der Selektor wählt das beste
- **Marktbewusst** — Fusion von 4 Quoten-Quellen (500.com, JC SP, international, Fallback) mit automatischer Marge-Entfernung und Gewichtung nach Glaubwürdigkeit
- **Kalibrierung transparent** — Vertrauen vs. Genauigkeit wird explizit gemessen. MD1: 66,7% Genauigkeit ab ≥0,60 Konfidenz (9/24 Spiele)
- **Transparent by Design** — Jede Vorhersage enthält Konfidenzwert, Motorbegründung und Wahrscheinlichkeitsverteilung. Keine Blackbox
- **Tägliche Automatisierung** — GitHub Actions läuft 2x täglich ohne menschliches Eingreifen

**Dies ist ein Open-Source-Ingenieurexperiment.** 971 historische Spiele, 50k Monte-Carlo-Simulationen pro Spiel, BPD-Irrationalitätserkennung, Bayessche Glaubensverfolgung — alles in einem Python-Repo. Fork es, brich es, mach es besser.

Vollständige Dokumentation (Englisch) : [`README.en.md`](./README.en.md)

---

## 🧠 Architektur

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

## ✨ Funktionen

- 🎯 **Dual-Engine-Selektor** — wählt pro Spiel dynamisch zwischen Poisson und Causal basierend auf Datenqualität
- 🔄 **50k MC-Simulationen** — bedingte Verzweigung für Torwahrscheinlichkeitsverteilungen
- 📊 **Multi-Quellen-Quotenfusion** — 500.com, JC SP, internationale Quoten mit automatischer Margenbereinigung
- 🧠 **Bayessche Glaubensverfolgung** — Beta-Binomial-konjugierte Updates im Turnierverlauf
- 🔍 **Irrationalitätserkennung** — BPD zur Erkennung von Marktanomalien
- 📅 **Historische Daten** — 971 Spiele aller Weltmeisterschaften seit 1930

## 🚀 Schnellstart

```bash
pip install -r requirements.txt

# Einzelspiel-Vorhersage
python main.py --home "Brazil" --away "Argentina"

# Mit Quoten-Korrektur
python main.py --home "Mexico" --away "South Korea" --use-odds

# Tägliche Komplettvorhersage
python scripts/daily_predict.py --save
```

## 📄 Lizenz

**MIT** — frei für jede Nutzung, auch kommerziell.

---

> Historische Daten: [open-football](https://github.com/openfootball/world-cup). Quoten: 500.com.
