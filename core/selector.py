"""
选择门 — 决定每场比赛谁当主力引擎

评分逻辑（源自重构方案 v3.0）：
- 泊松得分 = w1 × 历史数据量 + w2 × 泊松拟合优度
- 因果得分 = w1 × DAG覆盖度 + w2 × 异常检测分数

四维评分体系：
1. 历史数据量(0-100)：两队历史记录充足度
2. DAG覆盖度(0-100)：因果引擎能观察到的特征比例
3. 泊松拟合优度(0-100)：Poisson分布假设的适切性
4. 结构突变分数(0-100)：因果引擎检测到的异常信号

选择决策：
- score gap > 30: 主0.85 / 辅0.15
- score gap 10-30: 主0.70 / 辅0.30
- score gap < 10: 均分0.50 / 0.50
"""

from typing import Optional

from .data_types import (
    MatchContext,
    PoissonPrediction,
    CausalPrediction,
    CausalSignal,
    SelectorResult,
    SelectorScore,
)


# =============================================================================
# 常量
# =============================================================================

# 泊松维度权重
WEIGHT_HISTORICAL: float = 0.5  # 历史数据量权重
WEIGHT_POISSON_FIT: float = 0.5  # 泊松拟合优度权重

# 因果维度权重
WEIGHT_DAG: float = 0.4  # DAG覆盖度权重
WEIGHT_ANOMALY: float = 0.6  # 异常检测分数权重

# 权重阈值
GAP_LARGE: float = 30.0  # 差距大
GAP_SMALL: float = 10.0  # 差距小

# 主权重映射
W_PRIMARY_LARGE: float = 0.85   # 差距大时主权重
W_SECONDARY_LARGE: float = 0.15
W_PRIMARY_DEFAULT: float = 0.70  # 默认主权重
W_SECONDARY_DEFAULT: float = 0.30
W_PRIMARY_EVEN: float = 0.50    # 差距小时均分
W_SECONDARY_EVEN: float = 0.50


# =============================================================================
# 评分函数
# =============================================================================

def score_historical_volume(ctx: MatchContext) -> float:
    """
    历史数据量评分(0-100)。

    判断逻辑：
    - 两队都有 ≥20 场历史数据 → 高分(≥80)
    - 其中一队 <5 场 → 低分(≤30)
    - 首秀球队 → 极低分(<10)

    泊松引擎依赖历史数据做均值收缩，数据少时估计不可靠。
    """
    home_m = ctx.home.matches
    away_m = ctx.away.matches

    # 取较小值（短板效应）
    min_matches = min(home_m, away_m)

    if min_matches >= 20:
        return 90.0 + min((min_matches - 20) * 0.2, 10.0)  # 90-100
    elif min_matches >= 10:
        return 60.0 + (min_matches - 10) * 3.0  # 60-90
    elif min_matches >= 5:
        return 30.0 + (min_matches - 5) * 6.0  # 30-60
    elif min_matches >= 1:
        return 10.0 + min_matches * 4.0  # 14-30
    else:
        return 5.0  # 首秀


def score_dag_coverage(deviation_score: float, dag_coverage: float) -> float:
    """
    DAG覆盖度评分(0-100)，转换为因果引擎的优势分数。

    判断逻辑：
    - DAG覆盖度高(>0.8) + 偏离度适中 → 高分(因果能捕获异常因素)
    - DAG覆盖度低(<0.4) → 低分(数据不足以支撑因果推断)
    - 偏离度极高(>80) → 加分(因果在极端情况下更有价值)
    """
    # 基础分来自 DAG 覆盖度
    base = dag_coverage * 70.0

    # 偏离度加分：因果引擎善于处理异常
    anomaly_bonus = deviation_score * 0.3

    score = base + anomaly_bonus
    return min(100.0, max(0.0, score))


def score_poisson_goodness_of_fit(
    prediction: PoissonPrediction,
    ctx: MatchContext,
) -> float:
    """
    泊松拟合优度评分(0-100)。

    判断逻辑（使用代理指标）：
    - 两队比赛数充足（≥10）→ 泊松假设更合理
    - 进攻/防守强度不极端 → 泊松拟合好
    - 近期表现稳定 → 泊松适用
    - 首秀/极端情况 → 泊松假设可能不适切

    注意：完全准确的拟合优度需要卡方检验（需要真实比分数据）。
    Phase 1 用代理指标，Phase 2 接入真实数据后完善。
    """
    home_m = ctx.home.matches
    away_m = ctx.away.matches
    min_matches = min(home_m, away_m)

    # 比赛数评分
    if min_matches >= 15:
        match_score = 80.0 + min((min_matches - 15) * 0.5, 20.0)  # 80-100
    elif min_matches >= 8:
        match_score = 50.0 + (min_matches - 8) * 5.0  # 50-80
    elif min_matches >= 3:
        match_score = 20.0 + (min_matches - 3) * 10.0  # 20-50
    else:
        match_score = 10.0  # 数据太少，泊松不可靠

    # 进攻/防守极端度惩罚（方差大 → 泊松拟合差）
    home_ratio = _safe_div(ctx.home.avg_scored, max(ctx.home.avg_conceded, 0.01))
    away_ratio = _safe_div(ctx.away.avg_scored, max(ctx.away.avg_conceded, 0.01))

    extreme_penalty = 0.0
    if home_ratio > 3.0:
        extreme_penalty += 10.0
    if away_ratio > 3.0:
        extreme_penalty += 10.0
    if home_ratio < 0.33:
        extreme_penalty += 5.0
    if away_ratio < 0.33:
        extreme_penalty += 5.0

    return max(0.0, match_score - extreme_penalty)


def score_structure_break(
    causal_prediction: CausalPrediction,
) -> float:
    """
    结构突变分数(0-100)。

    判断逻辑：
    - 有结构突变 → 高分(因果引擎能捕获变化)
    - 突变幅度大 → 更高分
    - 无突变 → 低分(泊松的稳定假设未被挑战)

    注意：这个分数不是"好/坏"评判，而是"因果引擎对此比赛的价值"。
    结构突变越高，因果引擎越有价值。
    """
    signal = causal_prediction.signal
    if signal is None:
        return 0.0

    if not signal.structure_break:
        return 10.0  # 无突变，因果价值低

    magnitude = signal.break_magnitude
    if magnitude > 0.7:
        return 90.0
    elif magnitude > 0.5:
        return 70.0 + (magnitude - 0.5) * 100.0
    elif magnitude > 0.3:
        return 50.0 + (magnitude - 0.3) * 150.0
    else:
        return 30.0


# =============================================================================
# 决策函数
# =============================================================================

def select_engine(
    ctx: MatchContext,
    poisson_prediction: Optional[PoissonPrediction] = None,
    causal_prediction: Optional[CausalPrediction] = None,
) -> SelectorResult:
    """
    选择门主函数：评估两个引擎的适用性并决定谁当主力。

    Args:
        ctx: 比赛上下文
        poisson_prediction: 泊松引擎输出（可为 None，则用代理指标评分）
        causal_prediction: 因果引擎输出（可为 None，则用默认值）

    Returns:
        SelectorResult: 选择决策（含评分明细）
    """
    # === Step 1: 四个维度评分 ===

    # 1a. 历史数据量
    hist_vol = score_historical_volume(ctx)

    # 1b. DAG覆盖度（从因果预测中取信号，或从 ctx 推断）
    if causal_prediction and causal_prediction.signal:
        dag_cov_raw = causal_prediction.signal.dag_coverage
    else:
        # fallback: 简单估计
        dag_cov_raw = _estimate_dag_coverage(ctx)

    # 偏离度
    dev_score = causal_prediction.signal.deviation_score if (
        causal_prediction and causal_prediction.signal
    ) else 0.0

    dag_cov_score = score_dag_coverage(dev_score, dag_cov_raw)

    # 1c. 泊松拟合优度
    if poisson_prediction:
        poisson_fit = score_poisson_goodness_of_fit(poisson_prediction, ctx)
    else:
        poisson_fit = score_poisson_goodness_of_fit(PoissonPrediction(), ctx)

    # 1d. 结构突变
    if causal_prediction:
        str_break = score_structure_break(causal_prediction)
    else:
        str_break = 0.0

    # === Step 2: 加权计算引擎总分 ===

    poisson_score = WEIGHT_HISTORICAL * hist_vol + WEIGHT_POISSON_FIT * poisson_fit
    causal_score = WEIGHT_DAG * dag_cov_score + WEIGHT_ANOMALY * str_break

    # 归一化到 0-100
    poisson_score = max(0.0, min(100.0, poisson_score))
    causal_score = max(0.0, min(100.0, causal_score))

    # === Step 3: 选择决策 ===

    gap = poisson_score - causal_score
    is_poisson_primary = poisson_score >= causal_score

    if abs(gap) > GAP_LARGE:
        primary_w = W_PRIMARY_LARGE
        secondary_w = W_SECONDARY_LARGE
    elif abs(gap) > GAP_SMALL:
        primary_w = W_PRIMARY_DEFAULT
        secondary_w = W_SECONDARY_DEFAULT
    else:
        primary_w = W_PRIMARY_EVEN
        secondary_w = W_SECONDARY_EVEN

    if is_poisson_primary:
        primary_engine = "poisson"
        secondary_engine = "causal"
    else:
        primary_engine = "causal"
        secondary_engine = "poisson"
        # 注意：当因果为主时，primary_w / secondary_w 含义翻转但数值不变

    detail = SelectorScore(
        historical_data_volume=round(hist_vol, 1),
        dag_coverage=round(dag_cov_score, 1),
        poisson_goodness_of_fit=round(poisson_fit, 1),
        structure_break_score=round(str_break, 1),
    )

    return SelectorResult(
        poisson_score=round(poisson_score, 1),
        causal_score=round(causal_score, 1),
        primary_engine=primary_engine,
        secondary_engine=secondary_engine,
        primary_weight=primary_w,
        secondary_weight=secondary_w,
        score_gap=round(abs(gap), 1),
        detail=detail,
    )


# =============================================================================
# 内部帮助函数
# =============================================================================

def _estimate_dag_coverage(ctx: MatchContext) -> float:
    """从上下文粗略估计 DAG 覆盖度（无因果预测时使用）"""
    checks = 0
    total = 6  # 基础因子数

    if ctx.home_attack_strength > 0 or ctx.home.avg_scored > 0:
        checks += 1
    if ctx.away_attack_strength > 0 or ctx.away.avg_scored > 0:
        checks += 1
    if ctx.home_defense_strength > 0 or ctx.home.avg_conceded > 0:
        checks += 1
    if ctx.away_defense_strength > 0 or ctx.away.avg_conceded > 0:
        checks += 1
    if len(ctx.home_recent) > 0 or ctx.home.matches > 0:
        checks += 1
    if len(ctx.away_recent) > 0 or ctx.away.matches > 0:
        checks += 1

    return min(1.0, checks / total)


def _safe_div(a: float, b: float) -> float:
    """安全除法"""
    return a / b if b != 0 else float("inf")
