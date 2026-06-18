"""
贝叶斯信念更新引擎 — 融合先验与观测数据

用于持续学习：随着比赛进行，更新球队的真实强弱参数。

使用 Beta-Binomial 共轭先验（球队胜率/赢盘率信念更新）：
- update_belief(prior_alpha, prior_beta, observations_success, observations_total) → posterior
- combined_estimate(fusion_prediction, sim_result) → 后验概率
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .data_types import (
    MatchPrediction,
    PoissonPrediction,
    SimulationResult,
    ProbEstimate,
)

# 从 engine_poisson 复用 wilson_confidence_interval
from .engine_poisson import wilson_confidence_interval


# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class PosteriorState:
    """
    贝叶斯后验状态。

    Beta(a, b) 分布：
    - a = alpha (成功次数)
    - b = beta  (失败次数)
    - mean = a / (a + b)
    - variance = a*b / ((a+b)^2 * (a+b+1))
    """
    alpha: float = 1.0  # 后验 α（先验 α + 成功数）
    beta: float = 1.0   # 后验 β（先验 β + 失败数）
    posterior_mean: float = 0.5
    posterior_var: float = 0.083
    hdi_low: float = 0.0
    hdi_high: float = 0.0


# =============================================================================
# Beta-Binomial 共轭更新
# =============================================================================

def beta_binomial_update(
    prior_alpha: float,
    prior_beta: float,
    observations_success: int,
    observations_total: int,
) -> PosteriorState:
    """
    Beta-Binomial 共轭先验更新。

    P(θ | data) ∝ P(data | θ) * P(θ)
    = Beta(α + Σx, β + n - Σx)

    用于更新球队的胜率/赢盘率信念。
    例如：α=1, β=1 是无信息均匀先验；
    20场比赛10胜 → α=11, β=11 → mean=0.5

    Args:
        prior_alpha: 先验 Beta 分布 α 参数
        prior_beta: 先验 Beta 分布 β 参数
        observations_success: 观测到的成功次数（如胜场数）
        observations_total: 观测总次数

    Returns:
        PosteriorState: 后验分布参数和 HDI
    """
    if observations_total < 0:
        raise ValueError(f"observations_total 不能为负: {observations_total}")
    if observations_success < 0:
        raise ValueError(f"observations_success 不能为负: {observations_success}")
    if observations_success > observations_total:
        raise ValueError(
            f"observations_success ({observations_success}) > "
            f"observations_total ({observations_total})"
        )

    posterior_a = prior_alpha + observations_success
    posterior_b = prior_beta + (observations_total - observations_success)

    n = posterior_a + posterior_b
    mean = posterior_a / n if n > 0 else 0.5
    var = (posterior_a * posterior_b) / (n * n * (n + 1)) if n > 0 else 0.0

    # 近似 HDI（使用正态近似，Beta 近似正态当 a,b 较大时）
    # 更精确：用 scipy.stats.beta.interval，此处为无依赖简化版
    std = math.sqrt(var)
    hdi_low = max(0.0, mean - 1.96 * std)
    hdi_high = min(1.0, mean + 1.96 * std)

    return PosteriorState(
        alpha=posterior_a,
        beta=posterior_b,
        posterior_mean=round(mean, 4),
        posterior_var=round(var, 6),
        hdi_low=round(hdi_low, 4),
        hdi_high=round(hdi_high, 4),
    )


# =============================================================================
# 先验构造
# =============================================================================

def weak_uniform_prior() -> Tuple[float, float]:
    """
    弱信息均匀先验：Beta(1, 1)，相当于 "我什么都不知道"。

    Returns:
        (alpha, beta)
    """
    return (1.0, 1.0)


def skeptical_prior(strength: float = 0.5, pseudo_observations: int = 10) -> Tuple[float, float]:
    """
    有偏见的先验：用 pseudo_observations 个观测将 belief 拉向 strength。

    例如 skeptical_prior(0.55, 20) 意味着 "我认为这支球队大概有55%胜率，
    但不是很确定，所以我用20个假观测来抵消极端数据"。

    Args:
        strength: 先验信念的中心位置 (0-1)
        pseudo_observations: 信念强度（越大越顽固）

    Returns:
        (alpha, beta)
    """
    if not (0.0 < strength < 1.0):
        raise ValueError(f"strength 必须在 (0,1) 之间: {strength}")
    if pseudo_observations <= 0:
        raise ValueError(f"pseudo_observations 必须 > 0: {pseudo_observations}")

    alpha = strength * pseudo_observations
    beta = (1.0 - strength) * pseudo_observations
    return (alpha, beta)


def debutant_prior() -> Tuple[float, float]:
    """
    首秀球队先验：Beta(2, 8)，均值 0.2（首秀球队赢球概率低）。

    Returns:
        (alpha, beta)
    """
    return (2.0, 8.0)


def host_nation_prior() -> Tuple[float, float]:
    """
    东道主先验：Beta(6, 4)，均值 0.6（东道主有主场优势）。

    Returns:
        (alpha, beta)
    """
    return (6.0, 4.0)


# =============================================================================
# 融合引擎输出与贝叶斯更新
# =============================================================================

def combined_estimate(
    fusion_prediction: MatchPrediction,
    sim_result: Optional[SimulationResult] = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    historical_wins: int = 0,
    historical_total: int = 0,
) -> PosteriorState:
    """
    融合引擎预测 + MC 模拟 + 历史数据 → 后验概率。

    这是一个两层融合：
    1. 将引擎预测概率（home_win_prob）视为观测数据
    2. 用 MC 模拟的置信区间宽度调整观测的"有效样本量"
    3. 加上历史胜负记录做贝叶斯更新

    Args:
        fusion_prediction: 融合层输出的最终预测
        sim_result: MC 模拟结果（可选，用于置信度校准）
        prior_alpha: 先验 α
        prior_beta: 先验 β
        historical_wins: 历史胜场数（主队角度）
        historical_total: 历史比赛总数

    Returns:
        PosteriorState: 后验状态
    """
    # Step 1: 从 MC 模拟结果获取有效样本（置信区间宽/窄 → 有效样本少/多）
    effective_n = 50  # 默认：50 个等效观测

    if sim_result is not None and sim_result.home_win is not None:
        ci = sim_result.home_win
        # CI 越窄 → 有效样本越大
        ci_width = ci.hi - ci.lo
        if ci_width > 0.001:
            # Wilson CI 近似：宽度 ≈ 2*z*√(p(1-p)/n_eff)
            p = sim_result.home_win.p
            if 0 < p < 1:
                z = 1.96
                n_eff = (z * z * p * (1 - p)) / ((ci_width / 2) ** 2)
                effective_n = max(10, min(1000, int(n_eff)))
        # 也考虑融合置信度
        effective_n = int(effective_n * (0.5 + 0.5 * fusion_prediction.confidence))

    # Step 2: 将引擎预测概率转化为等效的"胜场"
    p = fusion_prediction.home_win_prob
    predicted_wins = p * effective_n
    predicted_total = effective_n

    # Step 3: 合并所有观测
    total_wins = historical_wins + int(round(predicted_wins))
    total_games = historical_total + predicted_total

    # Step 4: 贝叶斯更新
    return beta_binomial_update(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        observations_success=total_wins,
        observations_total=total_games,
    )


# =============================================================================
# 连续信念更新（跟踪球队）
# =============================================================================

@dataclass
class TeamBelief:
    """
    一支球队的贝叶斯信念状态。
    """
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    confidence: float = 0.0

    def update_with_match(
        self,
        won: bool,
        weight: float = 1.0,
    ) -> "TeamBelief":
        """
        用一场比赛结果更新信念。

        Args:
            won: 是否赢球
            weight: 这场比赛的信息权重（淘汰赛权重更高）

        Returns:
            self（链式调用）
        """
        if won:
            self.alpha += weight
        else:
            self.beta += weight

        n = self.alpha + self.beta
        self.confidence = min(1.0, n / (n + 5.0))

        return self

    def win_probability(self) -> float:
        """当前信念下的赢球概率（后验均值）"""
        n = self.alpha + self.beta
        return self.alpha / n if n > 0 else 0.5

    def credible_interval(self) -> Tuple[float, float]:
        """近似可信区间"""
        n = self.alpha + self.beta
        if n <= 0:
            return (0.0, 1.0)
        p = self.alpha / n
        var = (self.alpha * self.beta) / (n * n * (n + 1))
        std = math.sqrt(var)
        return (
            max(0.0, round(p - 1.96 * std, 4)),
            min(1.0, round(p + 1.96 * std, 4)),
        )


@dataclass
class BeliefTracker:
    """
    跟踪所有球队的贝叶斯信念。
    可用于模拟锦标赛过程中球队实力的动态更新。
    """
    teams: Dict[str, TeamBelief] = field(default_factory=dict)
    default_prior_alpha: float = 1.0
    default_prior_beta: float = 1.0

    def register_team(self, name: str, prior_alpha: Optional[float] = None,
                      prior_beta: Optional[float] = None) -> None:
        """注册一支球队"""
        if name not in self.teams:
            self.teams[name] = TeamBelief(
                name=name,
                alpha=prior_alpha or self.default_prior_alpha,
                beta=prior_beta or self.default_prior_beta,
                confidence=0.0,
            )

    def record_match(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        weight: float = 1.0,
    ) -> None:
        """
        记录一场比赛结果。

        Args:
            home: 主队名
            away: 客队名
            home_goals: 主队进球
            away_goals: 客队进球
            weight: 比赛权重
        """
        self.register_team(home)
        self.register_team(away)

        if home_goals > away_goals:
            self.teams[home].update_with_match(True, weight)
            self.teams[away].update_with_match(False, weight)
        elif away_goals > home_goals:
            self.teams[home].update_with_match(False, weight)
            self.teams[away].update_with_match(True, weight)
        else:
            # 平局：各算半个胜场和一个负场（从信息角度看）
            pass

    def get_team(self, name: str) -> Optional[TeamBelief]:
        """获取球队信念"""
        return self.teams.get(name)

    def summary(self) -> Dict[str, Dict]:
        """输出所有球队信念摘要"""
        return {
            name: {
                "alpha": round(t.alpha, 2),
                "beta": round(t.beta, 2),
                "win_prob": round(t.win_probability(), 4),
                "ci": t.credible_interval(),
                "confidence": round(t.confidence, 3),
            }
            for name, t in sorted(self.teams.items())
        }
