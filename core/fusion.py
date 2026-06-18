"""
融合层 — 加权合并泊松引擎和因果引擎的输出

支持四种模式：
- classic（纯泊松）：忽略因果引擎，完全输出泊松结果
- causal-only（纯因果）：忽略泊松引擎，完全输出因果结果
- auto（默认，推荐）：由选择门决定主/辅权重，加权融合
- debug：全量输出两个引擎和选择门的所有信息

融合逻辑（源自重构方案 v3.0）：
    最终λ_home = primary.λ_home × w_primary + secondary.λ_home × w_secondary
    最终conf   = primary.conf × w_primary + secondary.conf × w_secondary

Phase 4 增强：
    FusionEngine 类封装完整流程，支持可选的市场赔率偏差（odds_data）。
    赔率数据在泊松预测前应用，不侵入核心预测逻辑。
"""

from typing import Optional, Tuple

from .data_types import (
    OddsFusionResult,
    PoissonPrediction,
    CausalPrediction,
    CausalSignal,
    SelectorResult,
    MatchPrediction,
    MatchContext,
)
from .selector import select_engine
from .engine_poisson import DixonColesModel, compute_odds_bias


def fuse(
    poisson_prediction: Optional[PoissonPrediction],
    causal_prediction: Optional[CausalPrediction],
    ctx: Optional[MatchContext] = None,
    mode: str = "auto",
    selector_result: Optional[SelectorResult] = None,
    odds_data: Optional[Tuple[float, float, float]] = None,
    fusion_result: Optional[OddsFusionResult] = None,
) -> MatchPrediction:
    """
    融合两个引擎的预测输出。

    如果未提供 selector_result，会自动调用选择门进行评估。

    Args:
        poisson_prediction: 泊松引擎预测（classic 模式必需）
        causal_prediction: 因果引擎预测（causal-only 模式必需）
        ctx: 比赛上下文（仅在需要运行时选择门时使用）
        mode: 运行模式
            "classic" - 纯泊松
            "causal-only" - 纯因果
            "auto" - 选择门决定（默认）
            "debug" - 全输出
        selector_result: 选择门结果（可选，不提供则自动计算）
        odds_data: 单源赔率 (主胜赔率, 平赔率, 客胜赔率) — 向后兼容
        fusion_result: 多源赔率融合结果 (OddsFusionResult) — Phase 5
                       优先级高于 odds_data，同时提供时fusion_result生效

    Returns:
        MatchPrediction: 最终预测

    Raises:
        ValueError: 当所需引擎预测为 None 时
    """
    mode = mode.lower().strip()

    # 确定使用的赔率数据（fusion_result 优先级高于 odds_data）
    effective_odds = None
    effective_odds_desc = ""

    if fusion_result is not None and fusion_result.equiv_home_odds > 0:
        effective_odds = (fusion_result.equiv_home_odds,
                          fusion_result.equiv_draw_odds,
                          fusion_result.equiv_away_odds)
        effective_odds_desc = (
            f"多源融合({fusion_result.consensus_count}源, "
            f"分歧度{fusion_result.divergence:.3f})"
        )
    elif odds_data is not None:
        effective_odds = odds_data
        effective_odds_desc = "单源"

    # 如果提供了赔率数据，重新生成偏差后的泊松预测
    if effective_odds is not None and poisson_prediction is not None and ctx is not None:
        odds_home, odds_draw, odds_away = effective_odds
        hb, ab = compute_odds_bias(odds_home, odds_draw, odds_away)

        biased_home = poisson_prediction.expected_home_goals * hb
        biased_away = poisson_prediction.expected_away_goals * ab

        from .engine_poisson import finalize_baseline
        biased_pp = finalize_baseline(
            poisson_prediction.expected_home_goals,
            poisson_prediction.expected_away_goals,
            sample=0.0,
            home_bias=hb,
            away_bias=ab,
        )

        if mode == "debug":
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                "赔率偏差[%s]: odds=(%.2f, %.2f, %.2f) → bias=(%.3f, %.3f) "
                "λ: (%.2f, %.2f) → (%.2f, %.2f)",
                effective_odds_desc,
                odds_home, odds_draw, odds_away, hb, ab,
                poisson_prediction.expected_home_goals,
                poisson_prediction.expected_away_goals,
                biased_home, biased_away,
            )
            if fusion_result and fusion_result.consensus_count > 1:
                for sd in fusion_result.source_details:
                    logger.info("  [%s] %.2f/%.2f/%.2f (权重%.2f)",
                                sd.source_name, sd.odds_home, sd.odds_draw,
                                sd.odds_away, sd.weight)

        poisson_prediction = biased_pp

    # --- 模式分发 ---

    if mode == "classic":
        return _classic_mode(poisson_prediction)
    elif mode == "causal-only":
        return _causal_only_mode(causal_prediction)
    elif mode == "debug":
        return _debug_mode(poisson_prediction, causal_prediction, ctx, selector_result)
    elif mode == "auto":
        return _auto_mode(poisson_prediction, causal_prediction, ctx, selector_result)
    else:
        raise ValueError(f"未知模式: {mode}，支持: classic, causal-only, auto, debug")


# =============================================================================
# 各模式实现
# =============================================================================

def _classic_mode(
    poisson_prediction: Optional[PoissonPrediction],
) -> MatchPrediction:
    """
    纯泊松模式：忽略因果引擎，完全输出泊松结果。
    用于 backtest 基准验证——对比"加了因果补强后，预测准确率提升了多少"。
    """
    if poisson_prediction is None:
        raise ValueError("classic 模式需要 poisson_prediction")

    return MatchPrediction(
        mode="classic",
        primary_engine="poisson",
        primary_weight=1.0,
        secondary_engine="causal",
        secondary_weight=0.0,
        expected_home_goals=poisson_prediction.expected_home_goals,
        expected_away_goals=poisson_prediction.expected_away_goals,
        expected_total_goals=poisson_prediction.expected_total_goals,
        home_win_prob=poisson_prediction.home_win_prob,
        draw_prob=poisson_prediction.draw_prob,
        away_win_prob=poisson_prediction.away_win_prob,
        confidence=poisson_prediction.confidence,
        poisson_raw=poisson_prediction,
    )


def _causal_only_mode(
    causal_prediction: Optional[CausalPrediction],
) -> MatchPrediction:
    """
    纯因果模式：忽略泊松引擎，完全输出因果结果。
    """
    if causal_prediction is None:
        raise ValueError("causal-only 模式需要 causal_prediction")

    return MatchPrediction(
        mode="causal-only",
        primary_engine="causal",
        primary_weight=1.0,
        secondary_engine="poisson",
        secondary_weight=0.0,
        expected_home_goals=causal_prediction.expected_home_goals,
        expected_away_goals=causal_prediction.expected_away_goals,
        expected_total_goals=causal_prediction.expected_total_goals,
        home_win_prob=causal_prediction.home_win_prob,
        draw_prob=causal_prediction.draw_prob,
        away_win_prob=causal_prediction.away_win_prob,
        confidence=causal_prediction.confidence,
        causal_raw=causal_prediction,
        causal_signals=causal_prediction.signal,
    )


def _auto_mode(
    poisson_prediction: Optional[PoissonPrediction],
    causal_prediction: Optional[CausalPrediction],
    ctx: Optional[MatchContext],
    selector_result: Optional[SelectorResult],
) -> MatchPrediction:
    """
    自动模式（默认）：
    1. 运行选择门
    2. 按选择门权重加权融合
    """
    if poisson_prediction is None or causal_prediction is None:
        raise ValueError("auto 模式需要 poisson_prediction 和 causal_prediction")

    # 运行选择门（如果未提供）
    if selector_result is None:
        if ctx is None:
            raise ValueError("auto 模式需要 ctx 或 selector_result")
        selector_result = select_engine(ctx, poisson_prediction, causal_prediction)

    # === 加权融合 ===

    primary_name = selector_result.primary_engine
    secondary_name = selector_result.secondary_engine
    w_primary = selector_result.primary_weight
    w_secondary = selector_result.secondary_weight

    # 获取主/辅引擎的输出
    if primary_name == "poisson":
        primary_out = poisson_prediction
        secondary_out = causal_prediction
    else:
        primary_out = causal_prediction
        secondary_out = poisson_prediction

    # 加权合并 λ_home, λ_away
    fused_lh = primary_out.expected_home_goals * w_primary + secondary_out.expected_home_goals * w_secondary
    fused_la = primary_out.expected_away_goals * w_primary + secondary_out.expected_away_goals * w_secondary
    fused_total = fused_lh + fused_la

    # 加权合并概率
    fused_hw = primary_out.home_win_prob * w_primary + secondary_out.home_win_prob * w_secondary
    fused_dr = primary_out.draw_prob * w_primary + secondary_out.draw_prob * w_secondary
    fused_aw = primary_out.away_win_prob * w_primary + secondary_out.away_win_prob * w_secondary

    # 加权合并置信度
    fused_conf = primary_out.confidence * w_primary + secondary_out.confidence * w_secondary

    # 归一化概率
    prob_sum = fused_hw + fused_dr + fused_aw
    if prob_sum > 0:
        fused_hw /= prob_sum
        fused_dr /= prob_sum
        fused_aw /= prob_sum

    # 取信号的偏离度、结构突变信息
    signal = causal_prediction.signal if causal_prediction else None

    return MatchPrediction(
        mode="auto",
        primary_engine=selector_result.primary_engine,
        primary_weight=round(w_primary, 2),
        secondary_engine=selector_result.secondary_engine,
        secondary_weight=round(w_secondary, 2),
        selector_scores=selector_result,
        expected_home_goals=_round2(fused_lh),
        expected_away_goals=_round2(fused_la),
        expected_total_goals=_round2(fused_total),
        home_win_prob=_round2(fused_hw),
        draw_prob=_round2(fused_dr),
        away_win_prob=_round2(fused_aw),
        confidence=_round2(fused_conf),
        poisson_raw=poisson_prediction,
        causal_raw=causal_prediction,
        causal_signals=signal,
    )


def _debug_mode(
    poisson_prediction: Optional[PoissonPrediction],
    causal_prediction: Optional[CausalPrediction],
    ctx: Optional[MatchContext],
    selector_result: Optional[SelectorResult],
) -> MatchPrediction:
    """
    Debug 模式：输出自动模式结果 + 两个引擎的原始值 + 选择门完整明细。

    与 auto 模式相同，但始终在 poisson_raw/causal_raw/selector_scores 中
    包含完整数据。
    """
    result = _auto_mode(poisson_prediction, causal_prediction, ctx, selector_result)
    result.mode = "debug"
    return result


# =============================================================================
# FusionEngine 类（Phase 4 — 赔率偏差集成）
# =============================================================================


class FusionEngine:
    """
    融合引擎 — 封装完整预测流程，支持赔率偏差集成。

    与 fuse() 函数相同，但提供了 predict_match() 方法，
    在融合前自动应用赔率偏差。
    """

    def __init__(self, mode: str = "auto"):
        """
        Args:
            mode: 运行模式 (classic/causal-only/auto/debug)
        """
        self._mode = mode
        self._odds_data: Optional[Tuple[float, float, float]] = None

    def with_odds(self, odds_data: Optional[Tuple[float, float, float]]) -> "FusionEngine":
        """
        设置市场赔率数据。

        Args:
            odds_data: (主胜赔率, 平赔率, 客胜赔率) 或 None

        Returns:
            self（链式调用）
        """
        self._odds_data = odds_data
        return self

    def predict_match(
        self,
        poisson_prediction: Optional[PoissonPrediction],
        causal_prediction: Optional[CausalPrediction],
        ctx: Optional[MatchContext] = None,
        mode: Optional[str] = None,
        selector_result: Optional[SelectorResult] = None,
    ) -> MatchPrediction:
        """
        运行完整融合预测。

        Args:
            poisson_prediction: 泊松引擎预测
            causal_prediction: 因果引擎预测
            ctx: 比赛上下文
            mode: 覆盖构造时的模式
            selector_result: 选择门结果（可选）

        Returns:
            MatchPrediction
        """
        use_mode = mode if mode is not None else self._mode

        return fuse(
            poisson_prediction=poisson_prediction,
            causal_prediction=causal_prediction,
            ctx=ctx,
            mode=use_mode,
            selector_result=selector_result,
            odds_data=self._odds_data,
        )


# =============================================================================
# 内部工具
# =============================================================================

def _round2(x: float) -> float:
    return round(x * 100.0) / 100.0
