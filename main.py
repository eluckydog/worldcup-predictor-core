"""
2026 世界杯预测器 — CLI 入口

Usage:
    python main.py --home <team> --away <team> [--mode classic|causal-only|auto|debug]
                  [--seed <int>] [--sims <int>]

Modes:
    classic     — 纯泊松引擎（backtest 基准）
    causal-only — 纯因果引擎
    auto        — 选择门决定主/辅 + 加权融合（默认）
    debug       — 全输出（两个引擎 + 选择门 + MC 模拟）
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────
# 导入核心模块（绝对导入，适配从项目根目录运行）
# ──────────────────────────────────────────────────────────────────

from core.data_types import (
    PoissonPrediction,
    CausalPrediction,
    CausalSignal,
    MatchPrediction,
    MatchContext,
    TeamStats,
    HeadToHead,
    RecentMatch,
    SimulationResult,
    ProbEstimate,
    ScoreProb,
    IrrationalitySignal,
    PathwaySignal,
)
from core.engine_poisson import (
    set_seed as poisson_set_seed,
    compute_baseline,
    finalize_baseline,
    joint_score_grid,
    compute_team_form,
    DixonColesModel,
    compute_odds_bias,
)
from core.team_resolver import (
    resolve_team_name,
    is_debutant,
    is_host_nation,
    describe_team,
)
from core.monte_carlo import (
    normal_simulation,
    simulate_from_prediction,
    DEFAULT_TRIALS,
)
from core.bayesian import (
    combined_estimate,
)
from core.selector import select_engine
from core.fusion import fuse, FusionEngine
from core.irrationality import IrrationalityDetector
from core.pathway import PathwayOptimizer

# Logger setup
logger = logging.getLogger(__name__)

# 赔率提供器（lazy import）
_odds_provider_available = False


def _init_odds_provider() -> bool:
    """尝试初始化赔率提供器"""
    global _odds_provider_available
    if _odds_provider_available:
        return True
    try:
        from data.odds_provider import (
            get_odds_for_match, get_all_today_odds,
            get_multi_source_odds, get_fused_odds,
            display_multi_source, OddsFuser,
        )
        _odds_provider_available = True
        return True
    except ImportError as e:
        logger.debug("赔率提供器不可用: %s", e)
        return False
    except Exception as e:
        logger.debug("赔率提供器初始化失败: %s", e)
        return False


# 全局赔率缓存（--use-odds 模式加载一次，多次复用）
_odds_cache: Optional[Dict[str, Dict]] = None


def _load_odds_cache() -> Optional[Dict[str, Dict]]:
    """加载全部当日赔率并建立队名-赔率的双向查找"""
    global _odds_cache
    if _odds_cache is not None:
        return _odds_cache

    if not _init_odds_provider():
        return None

    from data.odds_provider import get_all_today_odds
    odds_list = get_all_today_odds()
    if not odds_list:
        logger.info("未找到当日赔率数据")
        return None

    cache: Dict[str, Dict] = {}
    for entry in odds_list:
        for team_key in [entry["home_cn"], entry["team_home"],
                         entry["away_cn"], entry["team_away"]]:
            if team_key not in cache:
                cache[team_key.lower()] = {
                    "odds_home": entry["odds_home"],
                    "odds_draw": entry["odds_draw"],
                    "odds_away": entry["odds_away"],
                    "home_cn": entry["home_cn"],
                    "away_cn": entry["away_cn"],
                    "team_home": entry.get("team_home", ""),
                    "team_away": entry.get("team_away", ""),
                }

    _odds_cache = cache
    logger.info("已加载 %d 条赔率数据", len(odds_list))
    return _odds_cache


def _get_match_odds(home: str, away: str) -> Optional[Tuple[float, float, float]]:
    """获取指定比赛的赔率"""
    cache = _load_odds_cache()
    if not cache:
        return None

    home_lower = home.strip().lower()
    away_lower = away.strip().lower()

    for team_key, odds_info in cache.items():
        home_match = (
            odds_info.get("team_home", "").strip().lower() == home_lower
            or odds_info.get("home_cn", "").strip().lower() == home_lower
        )
        away_match = (
            odds_info.get("team_away", "").strip().lower() == away_lower
            or odds_info.get("away_cn", "").strip().lower() == away_lower
        )
        if home_match and away_match and team_key == odds_info.get("team_home", "").strip().lower():
            return (odds_info["odds_home"], odds_info["odds_draw"], odds_info["odds_away"])

        # 反向匹配
        home_match2 = (
            odds_info.get("team_away", "").strip().lower() == home_lower
            or odds_info.get("away_cn", "").strip().lower() == home_lower
        )
        away_match2 = (
            odds_info.get("team_home", "").strip().lower() == away_lower
            or odds_info.get("home_cn", "").strip().lower() == away_lower
        )
        if home_match2 and away_match2 and team_key == odds_info.get("team_away", "").strip().lower():
            return (odds_info["odds_home"], odds_info["odds_draw"], odds_info["odds_away"])

    return None

# 数据层：优先从 SQLite 读取，fallback 到 Mock
from data.data_adapter import DataAdapter
_data_adapter: Optional[DataAdapter] = None

def _get_adapter() -> DataAdapter:
    global _data_adapter
    if _data_adapter is None:
        try:
            _data_adapter = DataAdapter()
        except Exception:
            _data_adapter = DataAdapter.__new__(DataAdapter)
            _data_adapter._db_path = ""
            _data_adapter._conn = None
            _data_adapter._team_cache = {}
            _data_adapter._global_avg = None
    return _data_adapter


# =============================================================================
# Mock 数据层（内置，当数据库不可用时的后备）
# =============================================================================

# 基本球队数据（FIFA 排名 → 模拟场均数据）
# 排名越好: avg_scored 越高, avg_conceded 越低
_TEAM_BASE_DATA: Dict[str, Dict] = {
    # 顶级强队（FIFA Top 5）
    "Argentina":   {"avg_scored": 2.1, "avg_conceded": 0.7, "matches": 30, "elo": 95},
    "France":      {"avg_scored": 2.0, "avg_conceded": 0.8, "matches": 28, "elo": 93},
    "Belgium":     {"avg_scored": 1.9, "avg_conceded": 0.8, "matches": 26, "elo": 90},
    "England":     {"avg_scored": 1.9, "avg_conceded": 0.8, "matches": 28, "elo": 91},
    "Brazil":      {"avg_scored": 2.0, "avg_conceded": 0.7, "matches": 32, "elo": 94},
    # 强队（Top 6-12）
    "Netherlands": {"avg_scored": 1.8, "avg_conceded": 0.9, "matches": 26, "elo": 88},
    "Portugal":    {"avg_scored": 1.8, "avg_conceded": 0.9, "matches": 25, "elo": 87},
    "Spain":       {"avg_scored": 1.7, "avg_conceded": 1.0, "matches": 27, "elo": 86},
    "Italy":       {"avg_scored": 1.6, "avg_conceded": 0.9, "matches": 24, "elo": 85},
    "Germany":     {"avg_scored": 1.7, "avg_conceded": 1.0, "matches": 26, "elo": 85},
    "Croatia":     {"avg_scored": 1.5, "avg_conceded": 1.0, "matches": 22, "elo": 83},
    "Uruguay":     {"avg_scored": 1.5, "avg_conceded": 1.0, "matches": 22, "elo": 82},
    # 中游（13-25）
    "Switzerland": {"avg_scored": 1.4, "avg_conceded": 1.1, "matches": 20, "elo": 78},
    "Colombia":    {"avg_scored": 1.4, "avg_conceded": 1.1, "matches": 20, "elo": 77},
    "Morocco":     {"avg_scored": 1.3, "avg_conceded": 1.1, "matches": 18, "elo": 76},
    "Japan":       {"avg_scored": 1.3, "avg_conceded": 1.2, "matches": 18, "elo": 75},
    "Denmark":     {"avg_scored": 1.4, "avg_conceded": 1.1, "matches": 20, "elo": 78},
    "Iran":        {"avg_scored": 1.2, "avg_conceded": 1.2, "matches": 16, "elo": 73},
    "South Korea": {"avg_scored": 1.2, "avg_conceded": 1.3, "matches": 16, "elo": 72},
    "Ecuador":     {"avg_scored": 1.3, "avg_conceded": 1.2, "matches": 17, "elo": 74},
    "Australia":   {"avg_scored": 1.1, "avg_conceded": 1.3, "matches": 15, "elo": 70},
    "Senegal":     {"avg_scored": 1.3, "avg_conceded": 1.2, "matches": 16, "elo": 73},
    "Poland":      {"avg_scored": 1.3, "avg_conceded": 1.2, "matches": 18, "elo": 74},
    "Türkiye":     {"avg_scored": 1.2, "avg_conceded": 1.3, "matches": 16, "elo": 71},
    # 中下游（25-40）
    "USA":         {"avg_scored": 1.4, "avg_conceded": 1.1, "matches": 22, "elo": 76},
    "Mexico":      {"avg_scored": 1.3, "avg_conceded": 1.2, "matches": 24, "elo": 75},
    "Canada":      {"avg_scored": 1.1, "avg_conceded": 1.4, "matches": 14, "elo": 68},
    "Nigeria":     {"avg_scored": 1.2, "avg_conceded": 1.3, "matches": 16, "elo": 70},
    "Serbia":      {"avg_scored": 1.2, "avg_conceded": 1.3, "matches": 15, "elo": 70},
    "Peru":        {"avg_scored": 1.1, "avg_conceded": 1.3, "matches": 14, "elo": 68},
    "Cameroon":    {"avg_scored": 1.1, "avg_conceded": 1.4, "matches": 14, "elo": 67},
    "Ghana":       {"avg_scored": 1.0, "avg_conceded": 1.4, "matches": 13, "elo": 66},
    "Saudi Arabia":{"avg_scored": 0.9, "avg_conceded": 1.5, "matches": 12, "elo": 63},
    "Jamaica":     {"avg_scored": 0.8, "avg_conceded": 1.6, "matches": 8,  "elo": 58},
    # 首秀/弱旅
    "TBD_A1":      {"avg_scored": 0.8, "avg_conceded": 1.7, "matches": 0,  "elo": 50},
    "TBD_C1":      {"avg_scored": 0.8, "avg_conceded": 1.7, "matches": 0,  "elo": 50},
    "TBD_D1":      {"avg_scored": 0.8, "avg_conceded": 1.6, "matches": 0,  "elo": 50},
    "TBD_E1":      {"avg_scored": 0.8, "avg_conceded": 1.6, "matches": 0,  "elo": 50},
    "TBD_F1":      {"avg_scored": 0.9, "avg_conceded": 1.5, "matches": 0,  "elo": 50},
    "TBD_G1":      {"avg_scored": 0.8, "avg_conceded": 1.6, "matches": 0,  "elo": 50},
    "TBD_H1":      {"avg_scored": 0.8, "avg_conceded": 1.6, "matches": 0,  "elo": 50},
    "TBD_I1":      {"avg_scored": 0.8, "avg_conceded": 1.6, "matches": 0,  "elo": 50},
    "TBD_J1":      {"avg_scored": 0.8, "avg_conceded": 1.6, "matches": 0,  "elo": 50},
    "TBD_K1":      {"avg_scored": 0.8, "avg_conceded": 1.6, "matches": 0,  "elo": 50},
    "TBD_L0":      {"avg_scored": 0.7, "avg_conceded": 1.8, "matches": 0,  "elo": 45},
    "TBD_L1":      {"avg_scored": 0.7, "avg_conceded": 1.8, "matches": 0,  "elo": 45},
    "TBD_L2":      {"avg_scored": 0.7, "avg_conceded": 1.8, "matches": 0,  "elo": 45},
    "TBD_L3":      {"avg_scored": 0.7, "avg_conceded": 1.8, "matches": 0,  "elo": 45},
}


def _get_team_stats(name: str, league_avg: float = 1.2) -> TeamStats:
    """获取球队统计，优先从数据库读取"""
    resolved = resolve_team_name(name)
    
    # 优先使用 DataAdapter（真实历史数据）
    adapter = _get_adapter()
    if adapter.has_data():
        try:
            real_stats = adapter.team_stats(resolved)
            if real_stats.matches > 0:
                return real_stats
        except Exception:
            pass
    
    # Fallback 到 Mock 数据
    data = _TEAM_BASE_DATA.get(resolved, {"avg_scored": 1.0, "avg_conceded": 1.4, "matches": 10, "elo": 60})

    return TeamStats(
        name=resolved,
        matches=data["matches"],
        wins=max(0, int(data["matches"] * (data["avg_scored"] / (data["avg_scored"] + data["avg_conceded"] + 0.1)) * 0.6)),
        draws=max(0, int(data["matches"] * 0.2)),
        losses=max(0, int(data["matches"] * 0.2)),
        goals_for=int(data["avg_scored"] * data["matches"]),
        goals_against=int(data["avg_conceded"] * data["matches"]),
        avg_scored=data["avg_scored"],
        avg_conceded=data["avg_conceded"],
        clean_sheets=max(0, int(data["matches"] * 0.25)),
    )


def _build_context(home: str, away: str, league_avg: float = 1.2) -> MatchContext:
    """构建比赛上下文（真实数据优先）"""
    home_resolved = resolve_team_name(home)
    away_resolved = resolve_team_name(away)

    home_stats = _get_team_stats(home_resolved, league_avg)
    away_stats = _get_team_stats(away_resolved, league_avg)

    # 真实 H2H
    h2h = HeadToHead(matches=0)
    adapter = _get_adapter()
    if adapter.has_data():
        try:
            real_h2h = adapter.head_to_head(home_resolved, away_resolved)
            if real_h2h.matches > 0:
                h2h = real_h2h
        except Exception:
            pass
    
    # 真实近期状态
    home_attack = home_stats.avg_scored
    home_defense = home_stats.avg_conceded
    away_attack = away_stats.avg_scored
    away_defense = away_stats.avg_conceded
    if adapter.has_data():
        try:
            h_form, h_def = adapter.weighted_form(home_resolved)
            a_form, a_def = adapter.weighted_form(away_resolved)
            if h_form > 0:
                home_attack, home_defense = h_form, h_def
            if a_form > 0:
                away_attack, away_defense = a_form, a_def
        except Exception:
            pass
    
    # 构建上下文
    context = MatchContext(
        home=home_stats,
        away=away_stats,
        head_to_head=h2h,
        league_avg_goals=league_avg,
        home_form_attack=home_attack,
        home_form_defense=home_defense,
        away_form_attack=away_attack,
        away_form_defense=away_defense,
        home_attack_strength=home_stats.avg_scored / max(league_avg, 0.01),
        home_defense_strength=home_stats.avg_conceded / max(league_avg, 0.01),
        away_attack_strength=away_stats.avg_scored / max(league_avg, 0.01),
        away_defense_strength=away_stats.avg_conceded / max(league_avg, 0.01),
    )

    # 标记主办国
    if is_host_nation(home_resolved):
        context.home_form_attack *= 1.08
    if is_host_nation(away_resolved):
        context.away_form_attack *= 1.08

    return context


# =============================================================================
# 引擎包装（简化版——直接使用泊松引擎 + 内置偏差因子模拟因果）
# =============================================================================

def _run_poisson(ctx: MatchContext) -> PoissonPrediction:
    """运行泊松引擎"""
    home_form = compute_team_form(
        stats=ctx.home, is_host=is_host_nation(ctx.home.name),
        weighted_attack=ctx.home_form_attack if ctx.home_form_attack > 0 else None,
        weighted_defense=ctx.home_form_defense if ctx.home_form_defense > 0 else None,
        weighted_matches=None,
    )
    away_form = compute_team_form(
        stats=ctx.away, is_host=is_host_nation(ctx.away.name),
        weighted_attack=ctx.away_form_attack if ctx.away_form_attack > 0 else None,
        weighted_defense=ctx.away_form_defense if ctx.away_form_defense > 0 else None,
        weighted_matches=None,
    )

    return compute_baseline(home_form, away_form, ctx.head_to_head, ctx.league_avg_goals)


def _run_causal(ctx: MatchContext) -> CausalPrediction:
    """模拟因果引擎输出（基于泊松 + 偏离度启发式）"""
    poisson = _run_poisson(ctx)

    # 模拟偏离度（基于两队实力差距倒推）
    elo_diff = abs(
        _TEAM_BASE_DATA.get(ctx.home.name, {}).get("elo", 70)
        - _TEAM_BASE_DATA.get(ctx.away.name, {}).get("elo", 70)
    )
    deviation_score = max(0, min(100, 10 + (elo_diff * 0.3)))

    # 结构突变：首秀球队视为有结构变化
    structure_break = is_debutant(ctx.home.name) or is_debutant(ctx.away.name)
    break_mag = 0.3 if structure_break else 0.0

    # Dynamic causal adjustment based on strength difference
    # Underdog gets a boost proportional to ELO gap (soccer underdog effect ~0.5-5%)
    home_elo = _TEAM_BASE_DATA.get(ctx.home.name, {}).get("elo", 70)
    away_elo = _TEAM_BASE_DATA.get(ctx.away.name, {}).get("elo", 70)
    elo_factor = (away_elo - home_elo) / 100.0  # negative if home stronger
    home_adj = 1.0 + elo_factor * 0.03  # 0.03 = max ~3% adjustment per 100 ELO
    away_adj = 1.0 - elo_factor * 0.03
    # Ensure adjustments stay within reasonable bounds
    home_adj = max(0.90, min(1.10, home_adj))
    away_adj = max(0.90, min(1.10, away_adj))

    signal = CausalSignal(
        deviation_score=round(deviation_score, 1),
        structure_break=structure_break,
        break_magnitude=break_mag,
        adjustment_factor_home=home_adj,
        adjustment_factor_away=away_adj,
        dag_coverage=min(1.0, (ctx.home.matches + ctx.away.matches) / 60.0),  # 60=预估强队生涯总比赛数上限
        factor_effects={"home_strength_gap": round(home_adj - 1.0, 3), "away_strength_gap": round(away_adj - 1.0, 3)},
    )

    return CausalPrediction(
        expected_home_goals=round(poisson.expected_home_goals * home_adj, 2),
        expected_away_goals=round(poisson.expected_away_goals * away_adj, 2),
        expected_total_goals=round(poisson.expected_total_goals * ((home_adj + away_adj) / 2), 2),
        home_win_prob=poisson.home_win_prob,
        draw_prob=poisson.draw_prob,
        away_win_prob=poisson.away_win_prob,
        confidence=round(poisson.confidence * 0.9, 2),
        signal=signal,
    )


# =============================================================================
# 默认市场赔率（用于路径优化降落）
# =============================================================================


def _default_market_odds(home_avg: float, away_avg: float, league_avg: float) -> Dict[str, float]:
    """
    从球队数据计算一致的默认市场赔率（无随机性的确定版本）。

    Args:
        home_avg: 主队场均进球
        away_avg: 客队场均进球
        league_avg: 联赛场均进球

    Returns:
        Dict: {"home": odds, "draw": odds, "away": odds}
    """
    # 同 irrationality._estimate_market_odds 的逻辑但无随机种子
    la = league_avg if league_avg > 0 else 1.2  # 联赛场均进球，默认1.2
    lambda_home = la * max(home_avg, 0.5) * 1.08  # 1.08=主场优势系数 HOST_EDGE
    lambda_away = la * max(away_avg, 0.5)

    total_lambda = lambda_home + lambda_away
    if total_lambda < 0.01:
        return {"home": 2.0, "draw": 3.3, "away": 4.0}

    # 泊松近似
    prob_home = lambda_home / (lambda_home + lambda_away + 0.5) * 0.7
    prob_draw = 0.25  # 无随机性的平局概率
    prob_away = max(0.05, 1.0 - prob_home - prob_draw)

    total = prob_home + prob_draw + prob_away
    prob_home /= total
    prob_draw /= total
    prob_away /= total

    odds_home = round(1.0 / max(prob_home, 0.01), 2)
    odds_draw = round(1.0 / max(prob_draw, 0.01), 2)
    odds_away = round(1.0 / max(prob_away, 0.01), 2)

    return {"home": odds_home, "draw": odds_draw, "away": odds_away}


# =============================================================================
# 格式化输出
# =============================================================================

def _format_main_box(
    result: MatchPrediction,
    sim_result: Optional[SimulationResult],
) -> List[str]:
    """Format the main prediction box (non-debug lines)."""
    home = _last_home_name
    away = _last_away_name
    mode = result.mode
    lines: List[str] = []

    lines.append(f"{home} vs {away} | mode={mode}")
    lines.append("┌" + "─" * 37 + "┐")

    eng = f"│ 引擎选择: {result.primary_engine}(主{result.primary_weight}) + {result.secondary_engine}(辅{result.secondary_weight})"
    lines.append(f"{eng:<39}│")

    # 赔率偏差信息
    if _use_odds and _match_odds_data:
        oh, od, oa = _match_odds_data
        lines.append(f"│ 赔率偏差: {oh:.2f}/{od:.2f}/{oa:.2f}                   │")

    lines.append(f"│ 预期进球: {result.expected_home_goals:.2f} - {result.expected_away_goals:<8.2f}│")

    hw_pct = result.home_win_prob * 100
    dr_pct = result.draw_prob * 100
    aw_pct = result.away_win_prob * 100
    lines.append(f"│ 概率分布: H {hw_pct:.1f}% / D {dr_pct:.1f}% / A {aw_pct:.1f}%    │")

    if sim_result:
        mls = sim_result.most_likely_score
        mls_prob = sim_result.top_outcomes.get(mls, 0) * 100
        lines.append(f"│ 最可能比分: {mls} (p={mls_prob:.1f}%)                │")
    else:
        lines.append(f"│ 最可能比分: {result.primary_engine} 基线                │")

    lines.append(f"│ 置信度: {result.confidence:.2f}                              │")

    if result.causal_signals:
        sig = result.causal_signals
        sb = "是" if sig.structure_break else "否"
        lines.append(f"│ 因果信号: 偏离度={sig.deviation_score:.0f}, 结构突变={sb}           │")
    else:
        lines.append("│ 因果信号: 无                                          │")

    lines.append("└" + "─" * 37 + "┘")
    return lines


def _format_debug_section(
    result: MatchPrediction,
    irr_signal: Optional[IrrationalitySignal],
    pwy_signal: Optional[PathwaySignal],
) -> List[str]:
    """Format the debug section (engine details + Phase 3 signals)."""
    lines: List[str] = ["", "── Debug 信息 ──"]

    if result.poisson_raw:
        p = result.poisson_raw
        lines.append(f"  泊松引擎: λ={p.expected_home_goals:.2f}/{p.expected_away_goals:.2f}  conf={p.confidence:.2f}")
    if result.causal_raw:
        c = result.causal_raw
        lines.append(f"  因果引擎: λ={c.expected_home_goals:.2f}/{c.expected_away_goals:.2f}  conf={c.confidence:.2f}")
    if result.selector_scores:
        sel = result.selector_scores
        lines.append(f"  选择门: poisson={sel.poisson_score:.1f} causal={sel.causal_score:.1f}  gap={sel.score_gap:.1f}")
        if sel.detail:
            d = sel.detail
            lines.append(f"  评分明细: 历史数据={d.historical_data_volume:.0f} "
                         f"DAG={d.dag_coverage:.0f} "
                         f"泊松拟合={d.poisson_goodness_of_fit:.0f} "
                         f"结构突变={d.structure_break_score:.0f}")

    if irr_signal:
        dir_map = {
            "overvalued_home": "高估主队",
            "overvalued_away": "高估客队",
            "undervalued_home": "低估主队",
            "undervalued_away": "低估客队",
            "neutral": "中性",
        }
        dir_cn = dir_map.get(irr_signal.direction, irr_signal.direction)
        lines.append(f"  非理性信号: score={irr_signal.score:.1f}, 方向={dir_cn}, 炵={irr_signal.entropy:.2f}")

    if pwy_signal:
        adj = pwy_signal.final_adjustment
        lambda_signal = adj.get("delta_home", 0.0)
        rec_text = pwy_signal.recommendations[0] if pwy_signal.recommendations else "无建议"
        lines.append(f"  路径优化: 收敛度{pwy_signal.convergence:.0f}%, 推荐调整λ={lambda_signal:+.4f}, 复杂度={pwy_signal.path_complexity}")
        lines.append(f"    建议: {rec_text}")

    return lines


def _format_mc_section(sim_result: SimulationResult) -> List[str]:
    """Format the Monte Carlo simulation section."""
    lines: List[str] = ["", f"  MC模拟 ({sim_result.trials} 次):"]
    lines.append(f"    预期进球: {sim_result.expected_home_goals:.2f} - {sim_result.expected_away_goals:.2f}")
    lines.append(f"    标准差: {sim_result.home_goals_sd:.2f} - {sim_result.away_goals_sd:.2f}")
    hw = sim_result.home_win
    dr = sim_result.draw
    aw = sim_result.away_win
    if hw and dr and aw:
        lines.append(f"    1X2: H [{hw.p:.1%} ({hw.lo:.1%}-{hw.hi:.1%})] "
                     f"D [{dr.p:.1%} ({dr.lo:.1%}-{dr.hi:.1%})] "
                     f"A [{aw.p:.1%} ({aw.lo:.1%}-{aw.hi:.1%})]")
    lines.append(f"    大2.5球: {sim_result.over_2_5:.1%}  |  双进: {sim_result.btts:.1%}")
    lines.append("    Top比分:")
    for sp in sim_result.top_scores[:5]:
        lines.append(f"      {sp.score}: {sp.probability:.1%}")
    return lines


def _format_prediction(
    result: MatchPrediction,
    sim_result: Optional[SimulationResult] = None,
    irr_signal: Optional[IrrationalitySignal] = None,
    pwy_signal: Optional[PathwaySignal] = None,
) -> str:
    """Format full prediction output by composing sub-sections."""
    lines: List[str] = _format_main_box(result, sim_result)

    # Phase 5: 多源赔率详细分析（debug模式 或 高分歧度时自动显示）
    if _multi_source_data is not None and _multi_source_data.consensus_count > 1:
        ms = _multi_source_data
        if ms.divergence > 0.02 or result.mode == "debug":
            lines.append(f"")
            lines.append(f"多源赔率详情 ({ms.consensus_count}源，共识分歧={ms.divergence:.3f}):")
            for sd in ms.source_details:
                sim_tag = " [模拟]" if "SIMULATED" in sd.source_name else ""
                lines.append(f"  [{sd.source_name}{sim_tag}] "
                             f"赔率={sd.odds_home:.2f}/{sd.odds_draw:.2f}/{sd.odds_away:.2f} "
                             f"→ H {sd.prob_home:.1%}/D {sd.prob_draw:.1%}/A {sd.prob_away:.1%}"
                             f" (抽水{sd.juice:.1%}, 权重{sd.weight:.2f})")
            lines.append(f"  λ偏差因子: H={ms.home_bias:.3f}, A={ms.away_bias:.3f}")
            lines.append("  " + "─" * 48)

    if result.mode == "debug":
        lines.extend(_format_debug_section(result, irr_signal, pwy_signal))
        if sim_result:
            lines.extend(_format_mc_section(sim_result))
    return "\n".join(lines)



# =============================================================================
# 全局变量（用于格式化）
# =============================================================================

_last_home_name: str = ""
_last_away_name: str = ""
# 全局赔率状态
_use_odds: bool = False
_match_odds_data: Optional[Tuple[float, float, float]] = None
_multi_source_data: Optional['OddsFusionResult'] = None


# =============================================================================
# CLI 流程
# =============================================================================

def run_prediction(
    home: str,
    away: str,
    mode: str = "auto",
    seed: int = 42,
    sims: int = DEFAULT_TRIALS,
    use_odds: bool = False,
    use_multi_source: bool = False,
) -> str:
    """
    运行完整预测流程。

    Args:
        home: 主队名
        away: 客队名
        mode: 运行模式
        seed: 随机种子
        sims: MC 模拟次数
        use_odds: 是否使用市场赔率偏差

    Returns:
        str: 格式化的预测输出
    """
    global _last_home_name, _last_away_name, _use_odds, _match_odds_data, _multi_source_data
    _use_odds = use_odds or use_multi_source
    _match_odds_data = None
    _multi_source_data = None

    # 解析队名
    home_resolved = resolve_team_name(home)
    away_resolved = resolve_team_name(away)
    _last_home_name = home_resolved
    _last_away_name = away_resolved

    # 设置种子
    poisson_set_seed(seed)

    # 构建上下文
    ctx = _build_context(home_resolved, away_resolved)
    league_avg = ctx.league_avg_goals

    # ── 赔率偏差（可选）──
    odds_data: Optional[Tuple[float, float, float]] = None
    fusion_result: Optional['OddsFusionResult'] = None

    if use_multi_source:
        # 多源融合模式
        try:
            from data.odds_provider import get_multi_source_odds
            fusion_result = get_multi_source_odds(home_resolved, away_resolved)
            _multi_source_data = fusion_result
            if fusion_result and fusion_result.equiv_home_odds > 0:
                odds_data = fusion_result.as_tuple
                logger.info(
                    "多源赔率融合: %s vs %s -> 等效赔率=(%.2f/%.2f/%.2f) "
                    "(%d源, 分歧度%.3f)",
                    home_resolved, away_resolved,
                    fusion_result.equiv_home_odds,
                    fusion_result.equiv_draw_odds,
                    fusion_result.equiv_away_odds,
                    fusion_result.consensus_count,
                    fusion_result.divergence,
                )
        except Exception as e:
            logger.warning("多源赔率融合失败: %s，使用单源", e)

    if use_odds and odds_data is None:
        odds_data = _get_match_odds(home_resolved, away_resolved)
        _match_odds_data = odds_data
        if odds_data:
            hb, ab = compute_odds_bias(*odds_data)
            logger.info(
                "单源赔率偏差: %s vs %s -> odds=(%.2f/%.2f/%.2f) -> bias=(%.3f, %.3f)",
                home_resolved, away_resolved, *odds_data, hb, ab,
            )
        else:
            logger.info("未找到 %s vs %s 的赔率数据，使用无偏差预测",
                         home_resolved, away_resolved)

    # ── 双引擎 ──
    poisson_result = _run_poisson(ctx)
    causal_result = _run_causal(ctx)

    # ── 选择门 ──
    selector_result = select_engine(ctx, poisson_result, causal_result)

    # ── 融合层（带赔率偏差，多源优先）──
    final_result = fuse(
        poisson_prediction=poisson_result,
        causal_prediction=causal_result,
        ctx=ctx,
        mode=mode,
        selector_result=selector_result if mode != "classic" else None,
        odds_data=odds_data,
        fusion_result=fusion_result,
    )

    # ── MC 模拟 ──
    sim_result = normal_simulation(
        exp_home=final_result.expected_home_goals,
        exp_away=final_result.expected_away_goals,
        trials=sims,
        seed=seed,
    )

    # ── Phase 3: 非理性检测 + 路径优化（仅 debug 模式） ──
    irr_signal: Optional[IrrationalitySignal] = None
    pwy_signal: Optional[PathwaySignal] = None

    if mode == "debug":
        mock_odds: Optional[Dict[str, float]] = None
        try:
            detector = IrrationalityDetector(seed=seed)
            mock_odds = detector._estimate_market_odds(ctx)
            irr_signal = detector.detect_market_irrationality(
                model_probs={
                    "home": poisson_result.home_win_prob,
                    "draw": poisson_result.draw_prob,
                    "away": poisson_result.away_win_prob,
                },
                market_odds=mock_odds,
            )
        except Exception as e:
            logger.warning("Irrationality detection failed: %s", e)

        try:
            optimizer = PathwayOptimizer(seed=seed)
            pwy_signal = optimizer.optimize_pathway(
                prediction=poisson_result,
                market_odds=mock_odds if mock_odds is not None
                else _default_market_odds(ctx.home.avg_scored, ctx.away.avg_scored, ctx.league_avg_goals),
                history=[],
            )
        except Exception as e:
            logger.warning("Pathway optimization failed: %s", e)

    # ── 格式化输出 ──
    output = _format_prediction(final_result, sim_result, irr_signal, pwy_signal)
    return output


# =============================================================================
# 主入口
# =============================================================================

def _show_odds() -> str:
    """显示当前可用赔率列表"""
    if not _init_odds_provider():
        return "赔率提供器不可用（未安装 requests 库或网络异常）"

    from data.odds_provider import get_all_today_odds
    odds_list = get_all_today_odds()

    if not odds_list:
        return "未获取到当日赔率数据"

    lines: List[str] = []
    lines.append(f"当日赔率列表 ({len(odds_list)} 场):")
    lines.append("-" * 80)
    lines.append(f"{'#':5s} {'赛事':20s} {'主队':20s} {'vs':5s} {'客队':20s} {'主胜':8s} {'平局':8s} {'客胜':8s}")
    lines.append("-" * 80)

    for i, entry in enumerate(odds_list, 1):
        league = entry.get("league", "")[:18]
        home = entry.get("home_cn", "")[:18]
        away = entry.get("away_cn", "")[:18]
        oh = entry.get("odds_home", 0)
        od = entry.get("odds_draw", 0)
        oa = entry.get("odds_away", 0)
        lines.append(
            f"{i:5d} {league:20s} {home:20s} {'vs':5s} {away:20s} "
            f"{oh:8.2f} {od:8.2f} {oa:8.2f}"
        )

    lines.append("-" * 80)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2026 世界杯预测器 — 双引擎 + MC 模拟 + 赔率偏差",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python main.py --home Brazil --away France --mode auto\n"
            "  python main.py --home Spain --away Canada --mode auto\n"
            "  python main.py --home Brazil --away France --mode classic\n"
            "  python main.py --home Brazil --away France --mode debug\n"
            "  python main.py --home Mexico --away Canada --use-odds\n"
            "  python main.py --show-odds\n"
        ),
    )

    parser.add_argument("--home", type=str, default=None, help="主队名（中文/英文皆可）")
    parser.add_argument("--away", type=str, default=None, help="客队名")
    parser.add_argument(
        "--mode", type=str, default="auto",
        choices=["classic", "causal-only", "auto", "debug"],
        help="运行模式（默认 auto）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--sims", type=int, default=DEFAULT_TRIALS,
                        help=f"MC 模拟次数（默认 {DEFAULT_TRIALS}，对标 sim.rs）")
    parser.add_argument("--use-odds", action="store_true",
                        help="启动时获取当日赔率（单源500.com），在预测时应用到引擎")
    parser.add_argument("--multi-source", action="store_true",
                        help="[Phase 5] 使用多源赔率融合（500.com + 竞彩SP + 国际博彩），"
                             "优先于 --use-odds")
    parser.add_argument("--show-odds", action="store_true",
                        help="显示当前可用的赔率列表（不预测）")
    parser.add_argument("--display-sources", action="store_true",
                        help="显示指定比赛的多源赔率详情（需要 --home --away）")

    args = parser.parse_args()

    # 纯赔率显示模式
    if args.show_odds:
        print(_show_odds())
        return

    # 多源赔率详情模式
    if args.display_sources:
        if not args.home or not args.away:
            print("--display-sources 需要 --home 和 --away 参数")
            return
        home_resolved = resolve_team_name(args.home)
        away_resolved = resolve_team_name(args.away)
        _init_odds_provider()
        try:
            from data.odds_provider import get_multi_source_odds, display_multi_source
            result = get_multi_source_odds(home_resolved, away_resolved)
            print()
            print(f"多源赔率分析: {home_resolved} vs {away_resolved}")
            print("=" * 65)
            print(display_multi_source(result, ""))
        except ImportError as e:
            print(f"赔率提供器不可用: {e}")
        return

    # 验证队名（预测模式必需）
    if not args.home or not args.away:
        parser.print_help()
        print("\n请使用 --home 和 --away 指定比赛双方，或使用 --show-odds 查看赔率")
        return

    home_resolved = resolve_team_name(args.home)
    away_resolved = resolve_team_name(args.away)

    output = run_prediction(
        home=home_resolved,
        away=away_resolved,
        mode=args.mode,
        seed=args.seed,
        sims=args.sims,
        use_odds=args.use_odds or args.multi_source,
        use_multi_source=args.multi_source,
    )

    print(output)


if __name__ == "__main__":
    main()
