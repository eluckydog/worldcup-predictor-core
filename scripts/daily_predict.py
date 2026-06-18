#!/usr/bin/env python3
"""
2026 World Cup — Daily Prediction Runner

Usage:
    python scripts/daily_predict.py          # Full daily report (today's completed + upcoming)
    python scripts/daily_predict.py --live    # Use live odds (if available)
    python scripts/daily_predict.py --save    # Save to predictions/ dir as markdown

Outputs predictions for all upcoming matches and backtest results for today's completed matches.
"""
import sys, os, re, json
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from main import run_prediction

CST = timezone(timedelta(hours=8))

# ── Match Database: 72 group stage matches ──
# (group, home, away, actual_home, actual_away, matchday, source)
# actual_home=None → not yet played
MATCHES = [
    # Group A
    ("A","Mexico","South Africa",2,0,1,"DB"),
    ("A","South Korea","Czech Republic",2,1,1,"DB"),
    ("A","Mexico","South Korea",None,None,2,"PENDING"),
    ("A","Czech Republic","South Africa",None,None,2,"PENDING"),
    ("A","South Africa","South Korea",None,None,3,"PENDING"),
    ("A","Czech Republic","Mexico",None,None,3,"PENDING"),
    # Group B
    ("B","Qatar","Switzerland",1,1,1,"DB"),
    ("B","Canada","Jordan",1,1,1,"DB"),
    ("B","Canada","Qatar",None,None,2,"PENDING"),
    ("B","Switzerland","Bosnia and Herzegovina",None,None,2,"PENDING"),
    ("B","Jordan","Switzerland",None,None,3,"PENDING"),
    ("B","Bosnia and Herzegovina","Canada",None,None,3,"PENDING"),
    # Group C
    ("C","Brazil","Morocco",1,1,1,"DB"),
    ("C","Scotland","Haiti",1,0,1,"DB"),
    ("C","Brazil","Scotland",None,None,2,"PENDING"),
    ("C","Morocco","Haiti",None,None,2,"PENDING"),
    ("C","Haiti","Scotland",None,None,3,"PENDING"),
    ("C","Morocco","Brazil",None,None,3,"PENDING"),
    # Group D
    ("D","USA","Paraguay",4,1,1,"DB"),
    ("D","Australia","Turkey",2,0,1,"DB"),
    ("D","USA","Australia",None,None,2,"PENDING"),
    ("D","Paraguay","Turkey",None,None,2,"PENDING"),
    ("D","Turkey","USA",None,None,3,"PENDING"),
    ("D","Paraguay","Australia",None,None,3,"PENDING"),
    # Group E
    ("E","Germany","Curaçao",7,1,1,"WEB"),
    ("E","Côte d'Ivoire","Ecuador",1,0,1,"WEB"),
    ("E","Germany","Côte d'Ivoire",None,None,2,"PENDING"),
    ("E","Ecuador","Curaçao",None,None,2,"PENDING"),
    ("E","Ecuador","Germany",None,None,3,"PENDING"),
    ("E","Curaçao","Côte d'Ivoire",None,None,3,"PENDING"),
    # Group F
    ("F","Sweden","Tunisia",5,1,1,"WEB"),
    ("F","Japan","Netherlands",2,2,1,"WEB"),
    ("F","Sweden","Japan",None,None,2,"PENDING"),
    ("F","Netherlands","Tunisia",None,None,2,"PENDING"),
    ("F","Netherlands","Sweden",None,None,3,"PENDING"),
    ("F","Tunisia","Japan",None,None,3,"PENDING"),
    # Group G
    ("G","Belgium","Egypt",1,1,1,"WEB"),
    ("G","Iran","New Zealand",2,2,1,"WEB"),
    ("G","Belgium","Iran",None,None,2,"PENDING"),
    ("G","New Zealand","Egypt",None,None,2,"PENDING"),
    ("G","Egypt","Iran",None,None,3,"PENDING"),
    ("G","New Zealand","Belgium",None,None,3,"PENDING"),
    # Group H
    ("H","Spain","Cape Verde",0,0,1,"WEB"),
    ("H","Saudi Arabia","Uruguay",1,1,1,"WEB"),
    ("H","Spain","Saudi Arabia",None,None,2,"PENDING"),
    ("H","Uruguay","Cape Verde",None,None,2,"PENDING"),
    ("H","Cape Verde","Saudi Arabia",None,None,3,"PENDING"),
    ("H","Uruguay","Spain",None,None,3,"PENDING"),
    # Group I
    ("I","France","Senegal",3,1,1,"WEB"),
    ("I","Norway","Iraq",4,1,1,"WEB"),
    ("I","France","Norway",None,None,2,"PENDING"),
    ("I","Senegal","Iraq",None,None,2,"PENDING"),
    ("I","Iraq","France",None,None,3,"PENDING"),
    ("I","Senegal","Norway",None,None,3,"PENDING"),
    # Group J
    ("J","Argentina","Algeria",3,0,1,"WEB"),
    ("J","Austria","Jordan",3,1,1,"WEB"),
    ("J","Argentina","Austria",None,None,2,"PENDING"),
    ("J","Algeria","Jordan",None,None,2,"PENDING"),
    ("J","Jordan","Argentina",None,None,3,"PENDING"),
    ("J","Algeria","Austria",None,None,3,"PENDING"),
    # Group K
    ("K","Portugal","DR Congo",1,1,1,"WEB"),
    ("K","Uzbekistan","Colombia",1,3,1,"WEB"),
    ("K","Portugal","Uzbekistan",None,None,2,"PENDING"),
    ("K","DR Congo","Colombia",None,None,2,"PENDING"),
    ("K","Colombia","Portugal",None,None,3,"PENDING"),
    ("K","DR Congo","Uzbekistan",None,None,3,"PENDING"),
    # Group L
    ("L","England","Croatia",4,2,1,"WEB"),
    ("L","Ghana","Panama",1,0,1,"WEB"),
    ("L","England","Ghana",None,None,2,"PENDING"),
    ("L","Croatia","Panama",None,None,2,"PENDING"),
    ("L","Panama","England",None,None,3,"PENDING"),
    ("L","Croatia","Ghana",None,None,3,"PENDING"),
]

def parse_pred(output):
    d = {"hp":0,"dp":0,"ap":0,"eh":0,"ea":0,"ml":"","mlp":0,"conf":0,"engine":""}
    for ln in output.split("\n"):
        if "引擎选择" in ln: d["engine"] = ln.strip()
        elif "预期进球" in ln:
            pts = ln.replace(":","").replace("-","").split()
            for p in pts:
                try:
                    v=float(p)
                    if d["eh"]==0: d["eh"]=v
                    elif d["ea"]==0: d["ea"]=v; break
                except: continue
        elif "概率分布" in ln:
            ns=re.findall(r"(\d+\.?\d*)%",ln)
            if len(ns)>=3:
                d["hp"],d["dp"],d["ap"]=float(ns[0]),float(ns[1]),float(ns[2])
        elif "最可能比分" in ln:
            m=re.search(r"(\d+:\d+)",ln)
            if m: d["ml"]=m.group(1)
            m=re.search(r"p=([\d.]+)%",ln)
            if m: d["mlp"]=float(m.group(1))
        elif "置信度" in ln:
            for p in ln.split():
                try:
                    v=float(p)
                    if 0<v<=1: d["conf"]=v
                except: continue
    return d

def render_1x2(hp, dp, ap):
    if hp > max(dp, ap): return f"Home ({hp:.0f}%)"
    if dp > max(hp, ap): return f"Draw ({dp:.0f}%)"
    return f"Away ({ap:.0f}%)"

def group_standings(completed):
    from collections import defaultdict
    standings = defaultdict(lambda: {"p":0,"gd":0,"gf":0,"ga":0,"w":0,"d":0,"l":0})
    for _,h,a,ha,aa,_,_ in completed:
        for tm in [h,a]:
            if tm not in standings: standings[tm] = {"p":0,"gd":0,"gf":0,"ga":0,"w":0,"d":0,"l":0}
        standings[h]["gf"]+=ha; standings[h]["ga"]+=aa
        standings[a]["gf"]+=aa; standings[a]["ga"]+=ha
        if ha>aa: standings[h]["p"]+=3; standings[h]["w"]+=1; standings[a]["l"]+=1
        elif ha==aa: standings[h]["p"]+=1; standings[a]["p"]+=1; standings[h]["d"]+=1; standings[a]["d"]+=1
        else: standings[a]["p"]+=3; standings[a]["w"]+=1; standings[h]["l"]+=1
        standings[h]["gd"]=standings[h]["gf"]-standings[h]["ga"]
        standings[a]["gd"]=standings[a]["gf"]-standings[a]["ga"]
    return dict(standings)

def main():
    save = "--save" in sys.argv
    use_odds = "--live" in sys.argv or "--use-odds" in sys.argv
    now = datetime.now(CST)
    today_str = now.strftime("%Y-%m-%d")

    completed = [m for m in MATCHES if m[3] is not None]
    upcoming = [m for m in MATCHES if m[3] is None]

    # Sort upcoming: today's matches first (by expected kickoff time)
    # We use MD order: earlier MD → earlier in day
    upcoming.sort(key=lambda m: (m[5], m[0]))

    lines = []
    lines.append(f"# 2026 World Cup — Daily Predictions ({today_str})")
    lines.append(f"")
    lines.append(f"_Generated at {now.strftime('%Y-%m-%d %H:%M %Z')}_")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Completed matches | {len(completed)} |")
    lines.append(f"| Upcoming matches | {len(upcoming)} |")
    lines.append(f"| Historical data | 971 matches (1930-2022) |")
    lines.append(f"")
    lines.append(f"---")

    # ── Part 1: Backtest (today's completed matches) ──
    lines.append(f"\n## ✅ Latest Results (Backtest)")
    lines.append(f"\n| Group | Match | Prediction | Actual | 1X2 | Confidence |")
    lines.append(f"|-------|-------|-----------|--------|-----|------------|")

    total_ok = 0
    for g,h,a,ha,aa,md,src in completed:
        try:
            out = run_prediction(h, a, mode="auto", seed=42)
            d = parse_pred(out)
        except Exception as e:
            lines.append(f"| {g} | {h} vs {a} | ❌ {e} | {ha}-{aa} | - | - |")
            continue

        hp,dp,ap = d["hp"],d["dp"],d["ap"]
        pred_1x2 = "H" if hp>max(dp,ap) else ("D" if dp>max(hp,ap) else "A")
        act_1x2 = "H" if ha>aa else ("D" if ha==aa else "A")
        ok = pred_1x2 == act_1x2
        if ok: total_ok += 1
        icon = "✅" if ok else "❌"
        lines.append(f"| {g} | {h} vs {a} | {render_1x2(hp,dp,ap)} ({hp:.0f}/{dp:.0f}/{ap:.0f}) | {ha}-{aa} | {icon} {pred_1x2}→{act_1x2} | {d['conf']:.2f} |")

    acc = total_ok/len(completed)*100 if completed else 0
    lines.append(f"\n**Backtest: {total_ok}/{len(completed)} correct ({acc:.1f}%)**")
    lines.append(f"")

    # ── Part 2: Upcoming predictions ──
    lines.append(f"\n## 🔮 Upcoming Match Predictions")
    lines.append(f"")

    last_md = None
    for g,h,a,_,_,md,src in upcoming:
        if last_md != md:
            lines.append(f"\n### Matchday {md}")
            last_md = md
        try:
            out = run_prediction(h, a, mode="auto", seed=42)
            d = parse_pred(out)
        except Exception as e:
            lines.append(f"\n- **[{g}]** {h} vs {a}: ❌ {e}")
            continue

        hp,dp,ap = d["hp"],d["dp"],d["ap"]
        tip = render_1x2(hp,dp,ap)
        lines.append(f"\n- **[{g}]** {h} vs {a}")
        lines.append(f"  - Expected goals: {d['eh']:.2f}–{d['ea']:.2f}")
        lines.append(f"  - Probability: H {hp:.0f}% / D {dp:.0f}% / A {ap:.0f}%")
        lines.append(f"  - Most likely: {d['ml']} (p={d['mlp']:.0f}%)")
        lines.append(f"  - Pick: {tip} (confidence: {d['conf']:.2f})")

    # ── Part 3: Group standings ──
    lines.append(f"\n## 📊 Group Standings")
    lines.append(f"")

    from collections import defaultdict
    st = group_standings(completed)
    for grp in sorted(set(m[0] for m in MATCHES)):
        g_teams = defaultdict(lambda: {"p":0,"gd":0,"gf":0,"ga":0,"w":0,"d":0,"l":0})
        for _,h,a,ha,aa,_,_ in completed:
            if _ != grp: continue
            for tm in [h,a]:
                if tm not in g_teams: g_teams[tm]={"p":0,"gd":0,"gf":0,"ga":0,"w":0,"d":0,"l":0}
            g_teams[h]["gf"]+=ha; g_teams[h]["ga"]+=aa
            g_teams[a]["gf"]+=aa; g_teams[a]["ga"]+=ha
            if ha>aa: g_teams[h]["p"]+=3; g_teams[h]["w"]+=1; g_teams[a]["l"]+=1
            elif ha==aa: g_teams[h]["p"]+=1; g_teams[a]["p"]+=1; g_teams[h]["d"]+=1; g_teams[a]["d"]+=1
            else: g_teams[a]["p"]+=3; g_teams[a]["w"]+=1; g_teams[h]["l"]+=1
            g_teams[h]["gd"]=g_teams[h]["gf"]-g_teams[h]["ga"]
            g_teams[a]["gd"]=g_teams[a]["gf"]-g_teams[a]["ga"]

        if g_teams:
            s = sorted(g_teams.items(), key=lambda x: (-x[1]["p"], -x[1]["gd"], -x[1]["gf"]))
            n_completed = sum(1 for m in MATCHES if m[0]==grp and m[3] is not None)
            lines.append(f"\n**Group {grp}** ({n_completed} matches played):")
            lines.append(f"| # | Team | P | W | D | L | GF | GA | GD | Pts |")
            lines.append(f"|---|------|---|---|---|---|----|----|----|-----|")
            for i,(tm,st2) in enumerate(s):
                lines.append(f"| {i+1} | {tm} | 1 | {st2['w']} | {st2['d']} | {st2['l']} | {st2['gf']} | {st2['ga']} | {st2['gd']:+d} | **{st2['p']}** |")

    lines.append(f"\n---")
    lines.append(f"\n*Data: 971 historical World Cup matches (1930-2022). Engine: Poisson + Causal dual-selector.*")
    lines.append(f"\n*[Source code](https://github.com/) | *Disclaimer: Statistical model, not betting advice.*")

    report = "\n".join(lines)

    # Output
    print(report)

    if save:
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions", f"{today_str}_daily_report.md")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📁 Saved to {save_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
