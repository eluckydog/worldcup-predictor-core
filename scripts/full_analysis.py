#!/usr/bin/env python3
"""
2026 World Cup — 全量预测 + 回测分析
整合 DB 已有数据 + 最新网页确认的赛果（截至 2026-06-18 18:18 CST）
全部48队首轮赛果已确认。
"""
import sys, os, re
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from main import run_prediction

# Format: (group, home, away, actual_home, actual_away, matchday, source)
# actual_home=None → 未开赛

ALL_MATCHES = [
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
    ("C","Morocco","Brazil",None,None,3,"PENDING"),
    ("C","Haiti","Scotland",None,None,3,"PENDING"),
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

def parse_prediction(output):
    data = {"home_win_pct":0,"draw_pct":0,"away_win_pct":0,"exp_home":0,"exp_away":0,"ml_score":"","ml_pct":0,"confidence":0,"engine":""}
    for line in output.split("\n"):
        if "引擎选择" in line: data["engine"] = line.strip()
        elif "预期进球" in line:
            parts = line.replace(":","").replace("-","").split()
            for p in parts:
                try:
                    v=float(p)
                    if data["exp_home"]==0: data["exp_home"]=v
                    elif data["exp_away"]==0: data["exp_away"]=v; break
                except: continue
        elif "概率分布" in line:
            nums=re.findall(r"(\d+\.\d)%",line)
            if len(nums)>=3:
                data["home_win_pct"]=float(nums[0])
                data["draw_pct"]=float(nums[1])
                data["away_win_pct"]=float(nums[2])
        elif "最可能比分" in line:
            m=re.search(r"(\d+:\d+)",line)
            if m: data["ml_score"]=m.group(1)
            m=re.search(r"p=([\d.]+)%",line)
            if m: data["ml_pct"]=float(m.group(1))
        elif "置信度" in line:
            for p in line.split():
                try:
                    v=float(p)
                    if v>1: continue
                    data["confidence"]=v
                except: continue
    return data

def main():
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"=== 2026 世界杯 全量回测+预测 ===")
    print(f"分析时间: {now_str}")
    print(f"总比赛: {len(ALL_MATCHES)} (已完赛24, 待赛48)")

    completed=[m for m in ALL_MATCHES if m[3] is not None]
    upcoming=[m for m in ALL_MATCHES if m[3] is None]
    md2_today=[m for m in upcoming if m[5]==2 and m[0] in ('A','B')]  # Tonight's MD2

    # ── 回测 ──
    print(f"\n{'─'*90}")
    print(f"【第一部分：首轮回测 — {len(completed)}场完赛】")
    print(f"{'─'*90}")

    total_correct=0
    from collections import defaultdict
    grp_results=defaultdict(list)

    for g,home,away,ha,aa,md,src in completed:
        try:
            output=run_prediction(home,away,mode="auto",seed=42,sims=50000)
            d=parse_prediction(output)
        except Exception as e:
            print(f"  ❌ {home}vs{away}: {e}")
            continue

        hp,dp,ap=d["home_win_pct"],d["draw_pct"],d["away_win_pct"]
        pred="H" if hp>max(dp,ap) else ("D" if dp>max(hp,ap) else "A")
        act="H" if ha>aa else ("D" if ha==aa else "A")
        correct=pred==act
        if correct: total_correct+=1

        mark="✅" if correct else "❌"
        act_s=f"{ha}-{aa} [{src}]"
        exp_s=f"{d['exp_home']:.2f}-{d['exp_away']:.2f}"
        prob_s=f"H{hp:.0f}/D{dp:.0f}/A{ap:.0f}"
        print(f"  [{g}] {mark} {home:20s} {away:20s} {exp_s:10s} {prob_s:20s} Conf:{d['confidence']:.2f} 实际:{act_s:15s}")

        grp_results[g].append({"correct":correct,"confidence":d["confidence"],
                                "pred_1x2":pred,"act_1x2":act})

    acc=total_correct/len(completed)*100
    print(f"\n📊 首轮1X2总准确率: {total_correct}/{len(completed)} = {acc:.1f}%")

    for lbl,lo,hi in [("高(≥0.70)",0.70,1.0),("中(0.50-0.69)",0.50,0.70),("低(<0.50)",0,0.50)]:
        subset=[r for gr in grp_results.values() for r in gr if lo<=r["confidence"]<hi]
        if subset:
            c=sum(1 for r in subset if r["correct"])
            print(f"  置信度{lbl}: {c}/{len(subset)}={c/len(subset)*100:.0f}%")

    print(f"\n各小组首轮准确率:")
    for g in sorted(grp_results):
        gr=grp_results[g]
        c=sum(1 for r in gr if r["correct"])
        print(f"  Group {g}: {c}/{len(gr)}")

    # ── 今晚MD2预测 ──
    print(f"\n{'─'*90}")
    print(f"【第二部分：今晚MD2预测（6月19日凌晨）— {len(md2_today)}场】")
    print(f"{'─'*90}")

    for g,home,away,_,_,md,src in md2_today:
        try:
            output=run_prediction(home,away,mode="auto",seed=42,sims=50000)
            d=parse_prediction(output)
        except Exception as e:
            print(f"  ❌ {home}vs{away}: {e}")
            continue

        hp,dp,ap=d["home_win_pct"],d["draw_pct"],d["away_win_pct"]
        if hp>max(dp,ap): tip=f"🏆 主胜({home})"
        elif dp>max(hp,ap): tip="⚖️ 平局"
        else: tip=f"🏆 客胜({away})"

        print(f"\n  [{g}] {home} vs {away} (MD2)")
        print(f"    预期进球: {d['exp_home']:.2f}-{d['exp_away']:.2f}")
        print(f"    概率分布: H{hp:.0f}%/D{dp:.0f}%/A{ap:.0f}%")
        print(f"    最可能比分: {d['ml_score']}(p={d['ml_pct']:.0f}%)")
        print(f"    置信度: {d['confidence']:.2f}")
        print(f"    {tip}")
        print(f"    引擎: {d['engine']}")

    # ── 小组积分 ──
    print(f"\n{'─'*90}")
    print(f"【第三部分：小组积分榜（首轮后）】")
    print(f"{'─'*90}")

    for grp in sorted(set(m[0] for m in ALL_MATCHES)):
        gmatches=[m for m in completed if m[0]==grp]
        st={}
        for _,h,a,ha,aa,_,_ in gmatches:
            for tm in [h,a]:
                if tm not in st: st[tm]=[0,0,0,0,0]
            st[h][0]+=1
            st[h][1]+=ha; st[h][2]+=aa
            st[h][3]+=ha-aa
            st[h][4]+=3 if ha>aa else 1 if ha==aa else 0
            st[a][0]+=1
            st[a][1]+=aa; st[a][2]+=ha
            st[a][3]+=aa-ha
            st[a][4]+=3 if aa>ha else 1 if ha==aa else 0

        s=sorted(st.items(),key=lambda x:(-x[1][4],-x[1][3],-x[1][1]))
        print(f"\n  Group {grp}:")
        for i,(tm,d) in enumerate(s):
            w=(d[4]//3 if d[4]>=3 else 0) if any(
                (grp==g and ((h==tm and ha>aa) or (a==tm and aa>ha)))
                for g,h,a,ha,aa,_,_ in gmatches) else 0
            w2=sum(1 for g,h,a,ha,aa,_,_ in gmatches
                   if g==grp and ((h==tm and ha>aa) or (a==tm and aa>ha)))
            d2=sum(1 for g,h,a,ha,aa,_,_ in gmatches
                   if g==grp and ((h==tm and ha==aa) or (a==tm and ha==aa)))
            l=d[0]-w2-d2
            print(f"    {i+1}. {tm:20s} {d[0]}场 {w2}胜 {d2}平 {l}负 进{d[1]} 失{d[2]} 净{d[3]:+d} {d[4]}分")

    print(f"\n=== 完成: {now_str} ===")

if __name__=="__main__":
    main()
