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
# Aligned to actual 2026 WC schedule (user's fixture text)
# (group, home, away, actual_home, actual_away, matchday, source)
# actual_home=None → not yet played
MATCHES = [
    # Group A
    ("A","Mexico","South Africa",2,0,1,"DB"),
    ("A","South Korea","Czech Republic",2,1,1,"DB"),
    ("A","Czech Republic","South Africa",1,1,2,"WEB"),
    ("A","Mexico","South Korea",1,0,2,"WEB"),
    ("A","Czech Republic","Mexico",None,None,3,"PENDING"),
    ("A","South Africa","South Korea",None,None,3,"PENDING"),
    # Group B  (Canada, Bosnia, Qatar, Switzerland)
    ("B","Canada","Bosnia and Herzegovina",1,1,1,"WEB"),
    ("B","Switzerland","Qatar",1,1,1,"WEB"),
    ("B","Switzerland","Bosnia and Herzegovina",4,1,2,"WEB"),
    ("B","Canada","Qatar",6,0,2,"WEB"),
    ("B","Switzerland","Canada",None,None,3,"PENDING"),
    ("B","Qatar","Bosnia and Herzegovina",None,None,3,"PENDING"),
    # Group C  (Brazil, Morocco, Haiti, Scotland)
    ("C","Morocco","Brazil",1,1,1,"WEB"),
    ("C","Haiti","Scotland",0,1,1,"WEB"),
    ("C","Scotland","Morocco",0,1,2,"WEB"),
    ("C","Brazil","Haiti",3,0,2,"WEB"),
    ("C","Brazil","Scotland",None,None,3,"PENDING"),
    ("C","Morocco","Haiti",None,None,3,"PENDING"),
    # Group D  (USA, Paraguay, Australia, Turkey)
    ("D","Paraguay","USA",1,4,1,"WEB"),
    ("D","Turkey","Australia",0,2,1,"WEB"),
    ("D","USA","Australia",2,0,2,"WEB"),
    ("D","Turkey","Paraguay",0,1,2,"WEB"),
    ("D","USA","Turkey",None,None,3,"PENDING"),
    ("D","Australia","Paraguay",None,None,3,"PENDING"),
    # Group E  (Germany, Curaçao, Ivory Coast, Ecuador)
    ("E","Curaçao","Germany",1,7,1,"WEB"),
    ("E","Ecuador","Côte d'Ivoire",0,1,1,"WEB"),
    ("E","Germany","Côte d'Ivoire",2,1,2,"WEB"),
    ("E","Ecuador","Curaçao",0,0,2,"WEB"),
    ("E","Germany","Ecuador",None,None,3,"PENDING"),
    ("E","Côte d'Ivoire","Curaçao",None,None,3,"PENDING"),
    # Group F  (Netherlands, Japan, Sweden, Tunisia)
    ("F","Japan","Netherlands",2,2,1,"WEB"),
    ("F","Sweden","Tunisia",5,1,1,"WEB"),
    ("F","Netherlands","Sweden",5,1,2,"WEB"),
    ("F","Tunisia","Japan",0,4,2,"WEB"),
    ("F","Netherlands","Tunisia",None,None,3,"PENDING"),
    ("F","Sweden","Japan",None,None,3,"PENDING"),
    # Group G  (Belgium, Egypt, Iran, New Zealand)
    ("G","Belgium","Egypt",1,1,1,"WEB"),
    ("G","New Zealand","Iran",2,2,1,"WEB"),
    ("G","Belgium","Iran",0,0,2,"WEB"),
    ("G","New Zealand","Egypt",1,3,2,"WEB"),
    ("G","Belgium","New Zealand",None,None,3,"PENDING"),
    ("G","Iran","Egypt",None,None,3,"PENDING"),
    # Group H  (Spain, Cape Verde, Saudi Arabia, Uruguay)
    ("H","Cape Verde","Spain",0,0,1,"WEB"),
    ("H","Uruguay","Saudi Arabia",1,1,1,"WEB"),
    ("H","Spain","Saudi Arabia",4,0,2,"WEB"),
    ("H","Uruguay","Cape Verde",2,2,2,"WEB"),
    ("H","Spain","Uruguay",None,None,3,"PENDING"),
    ("H","Saudi Arabia","Cape Verde",None,None,3,"PENDING"),
    # Group I  (France, Senegal, Iraq, Norway)
    ("I","France","Senegal",3,1,1,"WEB"),
    ("I","Norway","Iraq",4,1,1,"WEB"),
    ("I","Iraq","France",None,None,2,"PENDING"),
    ("I","Norway","Senegal",None,None,2,"PENDING"),
    ("I","France","Norway",None,None,3,"PENDING"),
    ("I","Iraq","Senegal",None,None,3,"PENDING"),
    # Group J  (Argentina, Algeria, Austria, Jordan)
    ("J","Algeria","Argentina",0,3,1,"WEB"),  # original had Argentina 3-0 Algeria → Algeria 0-3 Argentina
    ("J","Jordan","Austria",1,3,1,"WEB"),  # original had Austria 3-1 Jordan → Jordan 1-3 Austria
    ("J","Austria","Argentina",None,None,2,"PENDING"),
    ("J","Jordan","Algeria",None,None,2,"PENDING"),
    ("J","Algeria","Austria",None,None,3,"PENDING"),
    ("J","Jordan","Argentina",None,None,3,"PENDING"),
    # Group K  (Portugal, DR Congo, Uzbekistan, Colombia)
    ("K","Portugal","DR Congo",1,1,1,"WEB"),
    ("K","Uzbekistan","Colombia",1,3,1,"WEB"),
    ("K","Portugal","Uzbekistan",None,None,2,"PENDING"),
    ("K","Colombia","DR Congo",None,None,2,"PENDING"),
    ("K","Portugal","Colombia",None,None,3,"PENDING"),
    ("K","Uzbekistan","DR Congo",None,None,3,"PENDING"),
    # Group L  (England, Croatia, Ghana, Panama)
    ("L","Croatia","England",2,4,1,"WEB"),  # original had England 4-2 Croatia → Croatia 2-4 England
    ("L","Panama","Ghana",0,1,1,"WEB"),  # original had Ghana 1-0 Panama → Panama 0-1 Ghana
    ("L","England","Ghana",None,None,2,"PENDING"),
    ("L","Croatia","Panama",None,None,2,"PENDING"),
    ("L","Ghana","Croatia",None,None,3,"PENDING"),
    ("L","England","Panama",None,None,3,"PENDING"),
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
                except (ValueError, TypeError):
                    continue
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
                except (ValueError, TypeError):
                    continue
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
    lines.append(f"| Historical data | 964 matches (1930-2022) |")
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

    # -- Qualification & Bracket Report --
    from core.bracket import full_qualification_report
    qual_report = full_qualification_report(MATCHES)
    lines.append(f"\n---")
    lines.append(f"\n```")
    lines.append(qual_report)
    lines.append(f"```")

    lines.append(f"\n---")
    lines.append(f"\n*Data: 964 historical World Cup matches (1930-2022). Engine: Poisson + Causal dual-selector.*")
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
