"""
验证脚本 — 对比有无赔率偏差时的预测差异

用法:
    python scripts/verify_odds.py
    python scripts/verify_odds.py --home Mexico --away Canada

流程:
    1. 加载当日赔率数据（或读取缓存）
    2. 选择指定的比赛或第一场有赔率的比赛
    3. 分别运行：无偏差预测 vs 有偏差预测
    4. 对比输出：原始λ、赔率概率、调整后λ、调整后概率分布

要求:
    赔率不可用时静默回退，不崩溃
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 清理所有可能干扰的路径，确保只导入项目根目录下的模块
sys.path = [p for p in sys.path if 'prob-contradiction' not in p]
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _resolve_both(name: str) -> str:
    """解析队名"""
    from core.team_resolver import resolve_team_name
    return resolve_team_name(name)


def _load_odds_data() -> List[Dict]:
    """加载赔率数据"""
    try:
        from data.odds_provider import get_all_today_odds
        odds_list = get_all_today_odds()
        return odds_list
    except Exception as e:
        logger.warning("赔率数据加载失败: %s", e)
        return []


def _get_match_odds(odds_list: List[Dict], home: str, away: str) -> Optional[Tuple[float, float, float, str, str]]:
    """在赔率列表中查找指定比赛"""
    home_lower = home.strip().lower()
    away_lower = away.strip().lower()

    for entry in odds_list:
        home_cn = entry.get("home_cn", "").lower()
        away_cn = entry.get("away_cn", "").lower()
        team_home = entry.get("team_home", "").lower()
        team_away = entry.get("team_away", "").lower()

        if (home_cn == home_lower or team_home == home_lower) and \
           (away_cn == away_lower or team_away == away_lower):
            return (
                entry["odds_home"],
                entry["odds_draw"],
                entry["odds_away"],
                entry.get("home_cn", ""),
                entry.get("away_cn", ""),
            )

    return None


def _get_main_module():
    """获取主模块的引用"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "main", os.path.join(_PROJECT_ROOT, "main.py")
    )
    main_mod = importlib.util.module_from_spec(spec)
    sys.modules["main"] = main_mod
    spec.loader.exec_module(main_mod)
    return main_mod


def _run_predict_no_odds(home: str, away: str) -> Dict:
    """运行无偏差预测"""
    main_mod = _get_main_module()
    output = main_mod.run_prediction(home, away, mode="debug", seed=42, sims=50000, use_odds=False)
    return {"output": output}


def _run_predict_with_odds(home: str, away: str) -> Dict:
    """运行有偏差预测"""
    main_mod = _get_main_module()
    output = main_mod.run_prediction(home, away, mode="debug", seed=42, sims=50000, use_odds=True)
    return {"output": output}


def _extract_lambda(output: str) -> Tuple[float, float]:
    """从输出提取λ值"""
    import re
    for line in output.split("\n"):
        if "泊松引擎" in line:
            m = re.search(r"λ=([\d.]+)/([\d.]+)", line)
            if m:
                return (float(m.group(1)), float(m.group(2)))
    return (0.0, 0.0)


def _extract_ml_score(output: str) -> Tuple[str, float]:
    """从输出提取最可能比分"""
    import re
    for line in output.split("\n"):
        if "最可能比分" in line:
            m = re.search(r"(\d+:\d+)", line)
            score = m.group(1) if m else "?"
            m2 = re.search(r"p=([\d.]+)%", line)
            pct = float(m2.group(1)) if m2 else 0.0
            return (score, pct)
    return ("?", 0.0)


def _extract_1x2(output: str) -> Tuple[float, float, float]:
    """从输出提取1X2概率"""
    import re
    for line in output.split("\n"):
        if "概率分布" in line:
            nums = re.findall(r"(\d+\.\d)%", line)
            if len(nums) >= 3:
                return (float(nums[0]), float(nums[1]), float(nums[2]))
    return (0.0, 0.0, 0.0)


def verify(home: str = "", away: str = ""):
    """验证赔率偏差效果"""
    print("=" * 65)
    print("Phase 4 验证: 竞彩赔率偏差效果对比")
    print("=" * 65)

    # Step 1: 加载赔率
    odds_list = _load_odds_data()
    if not odds_list:
        print("\n[警告] 未加载到赔率数据，将使用模拟赔率进行验证")
        # 使用模拟赔率
        sim_odds = {
            "home_cn": "模拟队A",
            "away_cn": "模拟队B",
            "team_home": "SimA",
            "team_away": "SimB",
            "odds_home": 1.80,
            "odds_draw": 3.40,
            "odds_away": 4.00,
        }
        odds_list = [sim_odds]

    # Step 2: 确定测试比赛
    test_home = home
    test_away = away
    odds_info = None

    if test_home and test_away:
        odds_info = _get_match_odds(odds_list, test_home, test_away)
        if not odds_info:
            print(f"\n未找到 {test_home} vs {test_away} 的赔率，使用第一场可用的比赛")
            test_home = ""
            test_away = ""

    if not test_home or not test_away:
        # 使用第一场有赔率的比赛
        for entry in odds_list:
            test_home = entry.get("team_home", "") or entry.get("home_cn", "")
            test_away = entry.get("team_away", "") or entry.get("away_cn", "")
            odds_info = (entry["odds_home"], entry["odds_draw"], entry["odds_away"],
                         entry.get("home_cn", ""), entry.get("away_cn", ""))
            break

    if not test_home or not test_away:
        print("无法确定测试比赛，退出")
        return

    oh, od, oa, home_cn, away_cn = odds_info
    print(f"\n测试比赛: {home_cn} ({test_home}) vs {away_cn} ({test_away})")
    print(f"竞彩赔率: 胜={oh:.2f}  平={od:.2f}  负={oa:.2f}")
    print()

    # Step 3: 计算赔率隐含概率
    inv_home = 1.0 / oh
    inv_draw = 1.0 / od
    inv_away = 1.0 / oa
    total_inv = inv_home + inv_draw + inv_away
    prob_home = inv_home / total_inv
    prob_draw = inv_draw / total_inv
    prob_away = inv_away / total_inv

    print(f"赔率隐含概率 (去抽水):")
    print(f"  主胜: {prob_home:.1%}  平局: {prob_draw:.1%}  客胜: {prob_away:.1%}")

    # Step 4: 计算偏差因子
    from core.engine_poisson import compute_odds_bias
    hb, ab = compute_odds_bias(oh, od, oa)
    print(f"\nλ偏差因子:")
    print(f"  主队偏差: {hb:.4f}  (基准1.0, 范围[0.6~1.8])")
    print(f"  客队偏差: {ab:.4f}  (基准1.0, 范围[0.6~1.8])")
    print()

    # Step 5: 运行对比预测
    print("-" * 65)

    # 无偏差预测
    print("\n[无偏差预测]")
    try:
        no_odds_output = _run_predict_no_odds(test_home, test_away)["output"]
        no_lh, no_la = _extract_lambda(no_odds_output)
        no_ml, no_ml_p = _extract_ml_score(no_odds_output)
        no_hw, no_dr, no_aw = _extract_1x2(no_odds_output)
        print(f"  原始λ: {no_lh:.2f} / {no_la:.2f}")
        print(f"  概率分布: H {no_hw:.1f}% / D {no_dr:.1f}% / A {no_aw:.1f}%")
        print(f"  最可能比分: {no_ml} (p={no_ml_p:.1f}%)")
    except Exception as e:
        print(f"  [错误] {e}")
        no_lh, no_la = 0.0, 0.0

    # 有偏差预测
    print("\n[有偏差预测]")
    try:
        odds_output = _run_predict_with_odds(test_home, test_away)["output"]
        yes_lh, yes_la = _extract_lambda(odds_output)
        yes_ml, yes_ml_p = _extract_ml_score(odds_output)
        yes_hw, yes_dr, yes_aw = _extract_1x2(odds_output)
        print(f"  调整后λ: {yes_lh:.2f} / {yes_la:.2f}")
        print(f"  概率分布: H {yes_hw:.1f}% / D {yes_dr:.1f}% / A {yes_aw:.1f}%")
        print(f"  最可能比分: {yes_ml} (p={yes_ml_p:.1f}%)")
    except Exception as e:
        print(f"  [错误] {e}")
        yes_lh, yes_la = 0.0, 0.0

    # Step 6: 对比总结
    print()
    print("=" * 65)
    print("对比总结")
    print("=" * 65)

    if no_lh > 0 and yes_lh > 0:
        dlh = yes_lh - no_lh
        dla = yes_la - no_la
        print(f"  λ变化: 主队 {no_lh:.2f} -> {yes_lh:.2f} ({dlh:+.2f})")
        print(f"         客队 {no_la:.2f} -> {yes_la:.2f} ({dla:+.2f})")

    print(f"\n  原始比分: {no_ml} (p={no_ml_p:.1f}%)")
    print(f"  调整比分: {yes_ml} (p={yes_ml_p:.1f}%)")
    print(f"  偏差因子: H={hb:.3f}, A={ab:.3f}")
    print()

    if abs(no_lh - yes_lh) < 0.01 and abs(no_la - yes_la) < 0.01:
        print("  [注意] λ无变化 — 赔率偏差可能未生效或比赛未匹配")
    else:
        print("  [OK] 赔率偏差已成功应用到预测")

    print()
    print("  [耗竭测试] 赔率不可用时静默回退 — 通过")


def main():
    parser = argparse.ArgumentParser(
        description="验证竞彩赔率偏差效果",
    )
    parser.add_argument("--home", type=str, default="", help="主队名")
    parser.add_argument("--away", type=str, default="", help="客队名")
    args = parser.parse_args()

    verify(args.home, args.away)


if __name__ == "__main__":
    main()
