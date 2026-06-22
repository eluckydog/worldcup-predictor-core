# ⚽ 2026 World Cup Daily Predictor

**Fusión de doble motor** — regresión de Poisson + inferencia causal + motor de clasificación compatible con FIFA + cuotas multi-fuente

**Predicciones actualizadas a diario vía GitHub Actions → [`predictions/`](./predictions/)**

**100% código abierto · Licencia MIT**

---

## ⚡ Qué lo hace diferente

La mayoría de los proyectos de predicción de fútbol usan un solo modelo. Este no:

- **Arquitectura de doble motor** — Poisson y Causal compiten en cada partido; el selector elige el mejor
- **Consciente del mercado** — Fusión de 4 fuentes de cuotas (500.com, JC SP, internacionales, respaldo) con eliminación automática de margen y ponderación por credibilidad
- **Calibración transparente** — Medimos explícitamente la relación confianza/precisión. MD1: 66.7% de precisión en umbral ≥0.60 (9/24 partidos), tabla completa publicada
- **Transparente por diseño** — Cada predicción muestra puntuación de confianza, razón de selección del motor y distribución de probabilidad. Sin caja negra
- **Automatización diaria** — GitHub Actions ejecuta predicciones 2 veces al día sin intervención humana

**Esto es un experimento de ingeniería de código abierto.** 971 partidos históricos, 50k simulaciones Monte Carlo por partido, detección de irracionalidad BPD, seguimiento de creencias bayesiano — todo en un solo repositorio Python. Haz fork, rómpelo, mejóralo.

Documentación completa (inglés) : [`README.en.md`](./README.en.md)

---

## 🧠 Arquitectura

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

## ✨ Características

- 🎯 **Selector de doble motor** — elige dinámicamente entre Poisson y Causal según calidad de datos por partido
- 🔄 **50k simulaciones Monte Carlo** — muestreo condicional para distribuciones de probabilidad de goles
- 📊 **Fusión multi-fuente** — cuotas de 500.com, JC SP, internacionales con eliminación automática de margen
- 🧠 **Seguimiento bayesiano** — actualización conjugada Beta-Binomial a medida que avanza el torneo
- 🔍 **Detección de irracionalidad** — BPD para detectar anomalías del mercado
- 📅 **Datos históricos** — 971 partidos de todos los Mundiales desde 1930

## 🚀 Inicio rápido

```bash
pip install -r requirements.txt

# Predicción de un partido
python main.py --home "Brazil" --away "Argentina"

# Con corrección por cuotas
python main.py --home "Mexico" --away "South Korea" --use-odds

# Predicción diaria completa
python scripts/daily_predict.py --save
```

## 📄 Licencia

**MIT** — uso libre, incluido comercial.

---

> Datos históricos: [open-football](https://github.com/openfootball/world-cup). Cuotas: 500.com.
