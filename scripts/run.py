#!/usr/bin/env python3
"""
run.py — Unified prediction pipeline: DC → calibration → standings → stakes → MC → report

Usage:
    python scripts/run.py                          # Full report
    python scripts/run.py --mc                     # With MC simulation for total draws
    python scripts/run.py --mc --sims 100000       # Custom MC trials
    python scripts/run.py --save                   # Save report to predictions/
    python scripts/run.py --quick                  # Quick: MD2 + MD3 predictions only (no backtest)
    
Requires: data/matches.py (MATCHES), core/calibration, core/bracket, core/stakes, main
"""
import sys, os, re, json, time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from data.matches import MATCHES, completed, upcoming, by_group, by_matchday, group_list
from main import run_prediction
from core.calibration import calibrate, pick_outcome
from core.bracket import compute_standings, rank_third_placed, compute_qualification, render_qualifiers, render_standings, generate_bracket
from core.stakes import analyze_match, bracket_overview, render_stakes
from core.monte_carlo import normal_simulation, simulate_from_prediction, DEFAULT_TRIALS

CST = timezone(timedelta(hours=8))

# ── Config ──
DEFAULT_DELTA = 0.10


def parse_pred(output: str) -> dict:
    """Parse main.py prediction output into a dict."""
    d = {"hp": 0, "dp": 0, "ap": 0, "eh": 0, "ea": 0, "ml": "", "mlp": 0, "conf": 0, "engine": ""}
    for ln in output.split("\n"):
        if "引擎选择" in ln:
            d["engine"] = ln.strip()
        elif "预期进球" in ln:
            pts = ln.replace(":", "").replace("-", "").split()
            vals = []
            for p in pts:
                try:
                    vals.append(float(p))
                except (ValueError, TypeError):
                    continue
            if len(vals) >= 2:
                d["eh"], d["ea"] = vals[0], vals[1]
        elif "概率分布" in ln:
            ns = re.findall(r"(\d+\.?\d*)%", ln)
            if len(ns) >= 3:
                d["hp"], d["dp"], d["ap"] = float(ns[0]), float(ns[1]), float(ns[2])
        elif "最可能比分" in ln:
            m = re.search(r"(\d+:\d+)", ln)
            if m: d["ml"] = m.group(1)
            m = re.search(r"p=([\d.]+)%", ln)
            if m: d["mlp"] = float(m.group(1))
        elif "置信度" in ln:
            for p in ln.split():
                try:
                    v = float(p)
                    if 0 < v <= 1:
                        d["conf"] = v
                except (ValueError, TypeError):
                    continue
    return d


def render_outcome(hp: float, dp: float, ap: float, delta: float = DEFAULT_DELTA) -> str:
    """Render 1X2 with optional calibration indicator."""
    hp_c, dp_c, ap_c = calibrate(hp / 100, dp / 100, ap / 100, delta=delta)
    orig = "H" if hp > max(dp, ap) else ("D" if dp > max(hp, ap) else "A")
    cal = pick_outcome(hp_c, dp_c, ap_c)
    if orig == cal:
        names = {"H": "Home", "D": "Draw", "A": "Away"}
        return names[orig]
    return f"{names[orig]} → {names[cal]} (cal)"


def outcome_label(hp, dp, ap):
    if dp > max(hp, ap): return 'D'
    return 'H' if hp > ap else 'A'


# ── Single match DC + calibration ──

def predict_one(home: str, away: str, delta: float = DEFAULT_DELTA) -> dict:
    """Full prediction for a single match."""
    out = run_prediction(home, away, mode="auto", seed=42)
    d = parse_pred(out)
    hp_c, dp_c, ap_c = calibrate(d["hp"] / 100, d["dp"] / 100, d["ap"] / 100, delta=delta)
    d["cal"] = {"hp": hp_c * 100, "dp": dp_c * 100, "ap": ap_c * 100}
    d["pick_cal"] = pick_outcome(hp_c, dp_c, ap_c)
    return d


def predict_batch(matches: list, delta: float = DEFAULT_DELTA, verbose: bool = False) -> dict:
    """Batch predict all matches. Returns {(group,home,away,md): {pred}}."""
    results = {}
    for i, (g, h, a, hs, aw, md, src) in enumerate(matches):
        if verbose:
            print(f"  [{i+1}/{len(matches)}] {g} {h} vs {a}", file=sys.stderr)
        try:
            results[(g, h, a, md)] = predict_one(h, a, delta=delta)
        except Exception as e:
            if verbose:
                print(f"    ❌ {e}", file=sys.stderr)
    return results


# ── MC simulation for total draws ──

def mc_total_draws(predictions: dict, completed_list: list, n_sims: int = 50000, seed: int = 42) -> dict:
    """
    MC simulation for total draws across the entire tournament.
    Completed matches: fixed results. Remaining: sampled from calibrated probabilities.
    """
    # Map: (group,home,away,md) -> (hp_c, dp_c, ap_c)
    probs = {}
    for key, p in predictions.items():
        cal = p.get("cal", {})
        probs[key] = (cal.get("hp", 50) / 100, cal.get("dp", 25) / 100, cal.get("ap", 25) / 100)
    
    # Count draws in completed matches
    completed_draws = sum(1 for m in completed_list if m[3] == m[4])
    
    # Find remaining match keys
    remaining_keys = [(k, v) for k, v in probs.items()
                      if (k[0], k[1], k[2], k[3]) not in
                      {(m[0], m[1], m[2], m[5]) for m in completed_list}]
    # Actually just take all keys from predictions and check against completed
    # More reliable: match by (group, home, away, matchday)
    was_played = {(m[0], m[1], m[2], m[5]) for m in completed_list}
    remaining = {k: v for k, v in probs.items() if k not in was_played}
    
    rng = np.random.default_rng(seed)
    n_rem = len(remaining)
    outcomes = np.array([v for v in remaining.values()])  # shape (n, 3)
    
    draws = np.zeros(n_sims)
    for sim in range(n_sims):
        r = rng.random(n_rem)
        hits = np.argmax((r[:, None] < outcomes.cumsum(axis=1)).astype(int) > 0, axis=1)
        draws[sim] = completed_draws + int((hits == 1).sum())
    
    return {
        "completed_draws": completed_draws,
        "n_remaining": n_rem,
        "mean": float(draws.mean()),
        "p5": float(np.percentile(draws, 5)),
        "p25": float(np.percentile(draws, 25)),
        "p50": float(np.percentile(draws, 50)),
        "p75": float(np.percentile(draws, 75)),
        "p95": float(np.percentile(draws, 95)),
        "n_sims": n_sims,
    }


# ── Report generation ──

def generate_report(
    delta: float = DEFAULT_DELTA,
    with_mc: bool = False,
    mc_sims: int = DEFAULT_TRIALS,
    quick: bool = False,
):
    """Generate full prediction report."""
    now = datetime.now(CST)
    today_str = now.strftime("%Y-%m-%d")
    
    comp = completed()
    upcom = upcoming()
    total = len(MATCHES)
    
    lines = []
    lines.append(f"# 2026 World Cup — Prediction Report ({today_str})")
    lines.append(f"")
    lines.append(f"_Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}_")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Completed | {len(comp)} |")
    lines.append(f"| Upcoming | {len(upcom)} |")
    lines.append(f"| Draw rate (completed) | {sum(1 for m in comp if m[3]==m[4])/len(comp)*100:.1f}% |")
    lines.append(f"| Calibration delta | +{delta:.0%} |")
    lines.append(f"")
    lines.append(f"---")
    
    # ── Part 1: Backtest ──
    if not quick:
        lines.append(f"\n## ✅ Backtest ({len(comp)} completed)\n")
        lines.append(f"| Group | Match | Result | Calibration |")
        lines.append(f"|-------|-------|--------|-------------|")
        
        correct_raw = 0
        correct_cal = 0
        for g, h, a, hs, aw, md, src in comp:
            p = predict_one(h, a, delta=delta)
            raw = outcome_label(p["hp"], p["dp"], p["ap"])
            cal = p["pick_cal"]
            actual = "H" if hs > aw else ("D" if hs == aw else "A")
            raw_ok = raw == actual
            cal_ok = cal == actual
            if raw_ok: correct_raw += 1
            if cal_ok: correct_cal += 1
            raw_icon = "✅" if raw_ok else "❌"
            cal_icon = "✅" if cal_ok else "❌"
            cal_note = f"cal: {cal} {cal_icon}" if cal != raw else ""
            lines.append(f"| {g} | {h} vs {a} | {hs}-{aw} ({actual}) | {raw_icon} {raw} {cal_note} |")
        
        lines.append(f"\n**Baseline: {correct_raw}/{len(comp)} ({correct_raw/len(comp)*100:.1f}%)**")
        lines.append(f"**Calibrated (+{delta:.0%}∆): {correct_cal}/{len(comp)} ({correct_cal/len(comp)*100:.1f}%)**")
        if correct_cal > correct_raw:
            lines.append(f"_∆: +{correct_cal-correct_raw} more correct_")
    
    # ── Part 2: Predictions ──
    lines.append(f"\n---\n## 🔮 Upcoming Predictions\n")
    
    # Batch predict all upcoming
    print(f"  Running DC predictions for {len(upcom)} matches...", file=sys.stderr)
    t0 = time.time()
    all_preds = predict_batch(MATCHES, delta=delta, verbose=True)
    upcom_preds = {k: v for k, v in all_preds.items()
                   if (k[0], k[1], k[2], k[3]) in {(m[0], m[1], m[2], m[5]) for m in upcom}}
    elapsed = time.time() - t0
    print(f"  DC predictions done ({elapsed:.1f}s)", file=sys.stderr)
    
    # Group by matchday
    last_md = None
    for m in sorted(upcom, key=lambda x: (x[5], x[0])):
        g, h, a, _, _, md, src = m
        key = (g, h, a, md)
        p = upcom_preds.get(key, {})
        if not p:
            continue
        
        if md != last_md:
            lines.append(f"\n### Matchday {md}\n")
            lines.append(f"| Group | Home | Away | H | D | A | Calibrated | Pick |")
            lines.append(f"|-------|------|------|---|---|-------------|------|")
            last_md = md
        
        hp, dp, ap = p["hp"], p["dp"], p["ap"]
        cal = p["cal"]
        pick = p["pick_cal"]
        pick_name = {"H": "🏠", "D": "🤝", "A": "✈️"}.get(pick, pick)
        conf = f"{p['conf']:.2f}" if p['conf'] else "-"
        
        lines.append(f"| {g} | {h} | {a} | {hp:.0f}% | {dp:.0f}% | {ap:.0f}% "
                     f"| H{cal['hp']:.0f}%/D{cal['dp']:.0f}%/A{cal['ap']:.0f}% "
                     f"| {pick_name} (conf={conf}) |")
    
    # ── Part 3: Stakes Analysis for MD3 ──
    standings = compute_standings(MATCHES)
    lines.append(f"\n---\n## 🧠 MD3 Stakes Analysis\n")
    lines.append(f"{bracket_overview()}\n")
    
    md3_upcoming = [m for m in upcom if m[5] == 3]
    lines.append(f"### MD3 Draw Adjustments\n")
    lines.append(f"| Group | Match | Pts | Scenario | Bracket | Adj dp | Pick |")
    lines.append(f"|-------|-------|-----|----------|---------|--------|------|")
    lines.append(f"> Each MD3 match is analyzed for knockout incentive + group situation.")
    lines.append(f"> `Adj` = draw probability adjustment based on scenario.\n")
    
    for g, h, a, _, _, md, src in sorted(md3_upcoming, key=lambda x: x[0]):
        key = (g, h, a, md)
        p = all_preds.get(key, {})
        if not p:
            continue
        hp, dp, ap = p["hp"], p["dp"], p["ap"]
        cal = p["cal"]
        
        sa = analyze_match(g, h, a, standings)
        adj_dp = cal["dp"] / 100 + sa["adj"]
        # Re-distribute
        hp2 = (1 - adj_dp) * (cal["hp"] / 100) / (cal["hp"] / 100 + cal["ap"] / 100 + 1e-10)
        ap2 = (1 - adj_dp) * (cal["ap"] / 100) / (cal["hp"] / 100 + cal["ap"] / 100 + 1e-10)
        pick_stakes = "D" if adj_dp >= max(hp2, ap2) else ("H" if hp2 >= ap2 else "A")
        adj_str = f"{sa['adj']:+.0%}"
        pick_icon = {"H": "🏠", "D": "🤝", "A": "✈️"}.get(pick_stakes, pick_stakes)
        
        # Short scenario
        sc_short = sa["scenario"]
        if len(sc_short) > 35:
            sc_short = sc_short[:32] + "..."
        
        bracket_note = sa.get("bracket_note", "")
        if len(bracket_note) > 25:
            bracket_note = bracket_note[:22] + "..."
        
        lines.append(f"| {g} | {h} vs {a} | {sa['pts_h']}-{sa['pts_a']} | {sc_short} "
                     f"| {bracket_note} | {adj_str} | {pick_icon} |")
    
    # ── Part 4: Group Standings ──
    lines.append(f"\n---\n## 📊 Group Standings\n")
    lines.append(render_standings(standings))
    
    # ── Part 5: Qualification Picture ──
    qual = compute_qualification(MATCHES)
    lines.append(f"\n{render_qualifiers(qual)}")
    
    # ── Part 6: Draw Analysis ──
    n_draws_comp = sum(1 for m in comp if m[3] == m[4])
    exp_draws_rem = sum(
        all_preds.get((g, h, a, md), {}).get("cal", {}).get("dp", 25)
        for g, h, a, _, _, md, _ in upcom
    ) / 100
    total_exp = n_draws_comp + exp_draws_rem
    
    lines.append(f"\n---\n## 📈 Draw Analysis\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Completed draws | {n_draws_comp} / {len(comp)} ({n_draws_comp/len(comp)*100:.1f}%) |")
    lines.append(f"| Expected draws (remaining) | {exp_draws_rem:.1f} / {len(upcom)} |")
    lines.append(f"| **Expected total** | **{total_exp:.1f} / {total} ({total_exp/total*100:.1f}%)** |")
    lines.append(f"| Use `--mc` for MC simulation of total draws | |")
    
    if with_mc:
        print(f"  Running MC simulation ({mc_sims} trials)...", file=sys.stderr)
        mc = mc_total_draws(all_preds, comp, n_sims=mc_sims)
        lines.append(f"\n**MC Simulation ({mc['n_sims']:,} trials, {mc['n_remaining']} remaining):**\n")
        lines.append(f"| Percentile | Total draws | Rate |")
        lines.append(f"|-----------|-------------|------|")
        lines.append(f"| P5 | {mc['p5']:.0f} | {mc['p5']/total*100:.1f}% |")
        lines.append(f"| P25 | {mc['p25']:.0f} | {mc['p25']/total*100:.1f}% |")
        lines.append(f"| **P50 (median)** | **{mc['p50']:.0f}** | **{mc['p50']/total*100:.1f}%** |")
        lines.append(f"| P75 | {mc['p75']:.0f} | {mc['p75']/total*100:.1f}% |")
        lines.append(f"| P95 | {mc['p95']:.0f} | {mc['p95']/total*100:.1f}% |")
        lines.append(f"| Mean | {mc['mean']:.1f} | {mc['mean']/total*100:.1f}% |")
    
    lines.append(f"\n---\n"
                 f"*Pipeline: DC Poisson → uniform +{delta:.0%} calibration → LLM stakes → "
                 f"{'MC → ' if with_mc else ''}report*\n"
                 f"*Data: 964 historical matches (1930-2022). Not betting advice.*")
    
    return "\n".join(lines)


# ── CLI ──

if __name__ == "__main__":
    args = sys.argv[1:]
    delta = DEFAULT_DELTA
    with_mc = "--mc" in args
    mc_sims = int(args[args.index("--sims") + 1]) if "--sims" in args else DEFAULT_TRIALS
    save = "--save" in args
    quick = "--quick" in args
    
    report = generate_report(delta=delta, with_mc=with_mc, mc_sims=mc_sims, quick=quick)
    print(report)
    
    if save:
        today_str = datetime.now(CST).strftime("%Y-%m-%d")
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "predictions",
            f"{today_str}_pipeline_report.md",
        )
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📁 Saved to {save_path}", file=sys.stderr)
