"""
Dixon-Coles 泊松引擎 — 从 Rust 项目 model.rs 移植到 Python

移植说明：
- 保持与原 Rust 项目行为一致（固定种子兼容）
- 使用 numpy 替代 Rust 原生计算
- 所有常量、公式、概率计算与 Rust 源码对齐

原 Rust 文件：backend/src/model.rs
"""

import math
from typing import Optional, Tuple

import numpy as np

from .data_types import (
    TeamStats,
    TeamForm,
    PoissonPrediction,
    HeadToHead,
    MatchContext,
)


# =============================================================================
# 常量（与原 Rust 代码完全对齐）
# =============================================================================

# Dixon-Coles 低比分相关修正参数
DIXON_COLES_RHO: float = -0.06

# 主场优势（世界杯在中立场地，但主办国有轻微加持）
HOST_EDGE: float = 1.08

# 首秀球队参数
DEBUTANT_STRENGTH: float = 0.55
DEBUTANT_OPPONENT_BOOST: float = 1.25

# 形态衰减半衰期（年），约 2 个世界杯周期
FORM_HALF_LIFE_YEARS: float = 8.0

# 预测锚定年份
PREDICTION_YEAR: int = 2026

# 联合概率网格大小（与原 Rust N=10 一致）
JOINT_GRID_SIZE: int = 10

# 回归到均值的有效比赛数阈值
REGRESSION_SAMPLE_THRESHOLD: float = 10.0

# 置信度计算参数
CONFIDENCE_SAMPLE_CAP: float = 15.0
CONFIDENCE_BASE: float = 0.35
CONFIDENCE_SAMPLE_WEIGHT: float = 0.5
CONFIDENCE_SPREAD_WEIGHT: float = 0.2
CONFIDENCE_MIN: float = 0.2
CONFIDENCE_MAX: float = 0.95

# 2026 世界杯东道主
HOST_NATIONS: frozenset = frozenset({"USA", "Mexico", "Canada"})


# =============================================================================
# 基础数学函数（对齐 Rust 行为）
# =============================================================================

def poisson_pmf(lambda_: float, k: int) -> float:
    """
    泊松分布概率质量函数：P(X = k) = e^{-λ} * λ^k / k!
    使用 math.lgamma 替代 for 循环算 factorial，行为与 Rust poisson_pmf 一致。
    """
    if k < 0:
        return 0.0
    log_p = -lambda_ + k * math.log(lambda_) - math.lgamma(k + 1)
    return math.exp(log_p)


def dixon_coles_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    """
    Dixon-Coles 低比分修正因子 τ(i, j | λ_h, λ_a, ρ)
    对齐 Rust dixon_coles_tau 的实现。
    """
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    elif i == 0 and j == 1:
        return 1.0 + lh * rho
    elif i == 1 and j == 0:
        return 1.0 + la * rho
    elif i == 1 and j == 1:
        return 1.0 - rho
    else:
        return 1.0


def joint_score_grid(lh: float, la: float) -> np.ndarray:
    """
    计算联合比分概率网格 P(home=i, away=j) for i,j ∈ 0..N
    对齐 Rust joint_score_grid 的行为：
    1. 先算独立泊松 * Dixon-Coles τ
    2. 归一化使总概率 = 1.0
    3. 返回 (N+1) × (N+1) 的 numpy 数组
    """
    N = JOINT_GRID_SIZE
    grid = np.zeros((N + 1, N + 1), dtype=np.float64)
    total = 0.0

    for i in range(N + 1):
        p_i = poisson_pmf(lh, i)
        for j in range(N + 1):
            p_j = poisson_pmf(la, j)
            tau = dixon_coles_tau(i, j, lh, la, DIXON_COLES_RHO)
            cell = p_i * p_j * max(tau, 0.0)
            grid[i, j] = cell
            total += cell

    if total > 0.0:
        grid /= total

    return grid


def strength(team_avg: float, league_avg: float, eff_matches: float) -> float:
    """
    计算球队的攻防强度，向均值收缩。
    对齐 Rust strength 函数的行为。
    """
    if league_avg <= 0.0:
        return 1.0

    raw = team_avg / league_avg
    n = max(eff_matches, 0.0)
    w = n / (n + REGRESSION_SAMPLE_THRESHOLD)
    return max(0.25, min(3.0, raw * w + 1.0 * (1.0 - w)))


# =============================================================================
# 球队形态计算
# =============================================================================

def compute_team_form(
    stats: TeamStats,
    is_host: bool,
    weighted_attack: Optional[float] = None,
    weighted_defense: Optional[float] = None,
    weighted_matches: Optional[float] = None,
) -> TeamForm:
    """
    计算球队的加权攻防形态。

    如果提供加权值（来自外部数据源的指数衰减），则使用加权值；
    否则回退到全量平均统计。

    Args:
        stats: 球队统计
        is_host: 是否为2026东道主
        weighted_attack: 加权场均进球（由外部数据层提供）
        weighted_defense: 加权场均失球
        weighted_matches: 有效比赛数（权重和）

    Returns:
        TeamForm: 球队形态
    """
    if (weighted_attack is not None
            and weighted_defense is not None
            and weighted_matches is not None):
        attack = weighted_attack
        defense = weighted_defense
        eff_matches = weighted_matches
    elif stats.matches > 0:
        attack = stats.avg_scored
        defense = stats.avg_conceded
        eff_matches = float(stats.matches)
    else:
        attack = stats.avg_scored
        defense = stats.avg_conceded
        eff_matches = 0.0

    return TeamForm(
        attack=attack,
        defense=defense,
        eff_matches=eff_matches,
        is_host=is_host,
    )


def compute_baseline(
    home_form: TeamForm,
    away_form: TeamForm,
    h2h: HeadToHead,
    league_avg: float,
    home_bias: float = 1.0,
    away_bias: float = 1.0,
) -> PoissonPrediction:
    """
    计算泊松基线预测。对齐 Rust compute_baseline 的逻辑。

    步骤：
    1. 计算进攻/防守强度（均值收缩）
    2. 产生初始 λ_home, λ_away
    3. 混入直接 H2H 数据（如果双方交过手）
    4. 东道主加成
    5. 从联合概率网格生成完整概率
    """
    # Step 1: 攻防强度（相对联赛均值）
    home_attack_str = strength(home_form.attack, league_avg, home_form.eff_matches)
    home_defense_str = strength(home_form.defense, league_avg, home_form.eff_matches)
    away_attack_str = strength(away_form.attack, league_avg, away_form.eff_matches)
    away_defense_str = strength(away_form.defense, league_avg, away_form.eff_matches)

    # Step 2: 预期进球（世界杯中立场地，无通用主场优势）
    exp_home = league_avg * home_attack_str * away_defense_str
    exp_away = league_avg * away_attack_str * home_defense_str

    # Step 3: 混入 H2H 数据
    if h2h.matches > 0:
        h2h_home = h2h.a_goals / h2h.matches
        h2h_away = h2h.b_goals / h2h.matches
        w = min(h2h.matches / (h2h.matches + 4.0), 0.5)
        exp_home = exp_home * (1.0 - w) + h2h_home * w
        exp_away = exp_away * (1.0 - w) + h2h_away * w

    # Step 4: 东道主加成
    if home_form.is_host:
        exp_home *= HOST_EDGE
    if away_form.is_host:
        exp_away *= HOST_EDGE

    # Step 5: 完整概率（应用偏差因子）
    sample = min(home_form.eff_matches, away_form.eff_matches)
    return finalize_baseline(exp_home, exp_away, sample, home_bias, away_bias)


def finalize_baseline(
    exp_home: float,
    exp_away: float,
    sample: float,
    home_bias: float = 1.0,
    away_bias: float = 1.0,
) -> PoissonPrediction:
    """
    从预期进球参数生成完整基线预测。
    对齐 Rust finalize_baseline 的行为。
    既可被 compute_baseline 调用，也可被调整后的预期进球值调用。
    """
    # 应用赔率偏差因子
    adj_home = exp_home * home_bias
    adj_away = exp_away * away_bias

    # 裁剪防止数值问题
    adj_home = max(0.05, min(6.0, adj_home))
    adj_away = max(0.05, min(6.0, adj_away))
    exp_total = adj_home + adj_away

    # 从 Dixon-Coles 联合网格计算所有概率
    grid = joint_score_grid(adj_home, adj_away)

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    over_2_5 = 0.0
    btts = 0.0

    N = JOINT_GRID_SIZE
    for i in range(N + 1):
        for j in range(N + 1):
            p = grid[i, j]
            if p > 0:
                if i > j:
                    home_win += p
                elif i < j:
                    away_win += p
                else:
                    draw += p

                if i + j >= 3:
                    over_2_5 += p
                if i >= 1 and j >= 1:
                    btts += p

    # 模式得分（最可能比分）
    flat_idx = int(np.argmax(grid))
    pred_home = flat_idx // (N + 1)
    pred_away = flat_idx % (N + 1)

    # 置信度：样本量 + 概率分布离散度
    sample_factor = min(max(sample / (sample + CONFIDENCE_SAMPLE_CAP), 0.0), 0.7)
    spread = abs(max(home_win, away_win) - draw)
    confidence = max(
        CONFIDENCE_MIN,
        min(
            CONFIDENCE_MAX,
            CONFIDENCE_BASE + sample_factor * CONFIDENCE_SAMPLE_WEIGHT + spread * CONFIDENCE_SPREAD_WEIGHT,
        ),
    )

    return PoissonPrediction(
        expected_home_goals=_round2(adj_home),
        expected_away_goals=_round2(adj_away),
        expected_total_goals=_round2(exp_total),
        home_win_prob=_round2(home_win),
        draw_prob=_round2(draw),
        away_win_prob=_round2(away_win),
        prob_over_2_5=_round2(over_2_5),
        prob_btts=_round2(btts),
        predicted_home_score=pred_home,
        predicted_away_score=pred_away,
        predicted_total_goals=pred_home + pred_away,
        confidence=_round2(confidence),
    )


# =============================================================================
# 公开预测接口
# =============================================================================

def predict_match(
    ctx: MatchContext,
    home_bias: float = 1.0,
    away_bias: float = 1.0,
) -> PoissonPrediction:
    """
    对一场比赛运行泊松引擎，返回完整预测。

    这是 engine_poisson.py 的主要公开入口。

    Args:
        ctx: 比赛上下文（含两队统计、H2H、联赛均值等）
        home_bias: 主队λ偏差因子（来自市场赔率），默认1.0无偏差
        away_bias: 客队λ偏差因子

    Returns:
        PoissonPrediction: 完整的泊松基线预测
    """
    # 判断东道主
    home_is_host = ctx.home.name in HOST_NATIONS
    away_is_host = ctx.away.name in HOST_NATIONS

    # 构建加权形态
    home_form = compute_team_form(
        stats=ctx.home,
        is_host=home_is_host,
        weighted_attack=ctx.home_form_attack if ctx.home_form_attack > 0 else None,
        weighted_defense=ctx.home_form_defense if ctx.home_form_defense > 0 else None,
        weighted_matches=None,
    )
    away_form = compute_team_form(
        stats=ctx.away,
        is_host=away_is_host,
        weighted_attack=ctx.away_form_attack if ctx.away_form_attack > 0 else None,
        weighted_defense=ctx.away_form_defense if ctx.away_form_defense > 0 else None,
        weighted_matches=None,
    )

    return compute_baseline(
        home_form, away_form, ctx.head_to_head, ctx.league_avg_goals,
        home_bias=home_bias, away_bias=away_bias,
    )


def predict_from_raw(
    home_name: str,
    away_name: str,
    home_avg_scored: float,
    home_avg_conceded: float,
    home_matches: int,
    away_avg_scored: float,
    away_avg_conceded: float,
    away_matches: int,
    league_avg: float,
    h2h_matches: int = 0,
    h2h_home_goals_per_game: float = 0.0,
    h2h_away_goals_per_game: float = 0.0,
    home_is_host: bool = False,
    away_is_host: bool = False,
) -> PoissonPrediction:
    """
    快捷接口：直接从原始统计计算，无需构建完整 MatchContext。

    用于轻量级调用和测试。

    Args:
        home_name: 主队名
        away_name: 客队名
        home_avg_scored: 主队场均进球
        home_avg_conceded: 主队场均失球
        home_matches: 主队比赛数
        away_avg_scored: 客队场均进球
        away_avg_conceded: 客队场均失球
        away_matches: 客队比赛数
        league_avg: 联赛场均进球
        h2h_matches: 交锋次数
        h2h_home_goals_per_game: 交锋中场均进球（主队角度）
        h2h_away_goals_per_game: 交锋中场均进球（客队角度）
        home_is_host: 是否为2026东道主
        away_is_host: 是否为2026东道主

    Returns:
        PoissonPrediction
    """
    home_form = TeamForm(
        attack=home_avg_scored,
        defense=home_avg_conceded,
        eff_matches=float(home_matches),
        is_host=home_is_host,
    )
    away_form = TeamForm(
        attack=away_avg_scored,
        defense=away_avg_conceded,
        eff_matches=float(away_matches),
        is_host=away_is_host,
    )

    h2h = HeadToHead(
        matches=h2h_matches,
        a_goals=int(round(h2h_home_goals_per_game * h2h_matches)),
        b_goals=int(round(h2h_away_goals_per_game * h2h_matches)),
    )

    return compute_baseline(home_form, away_form, h2h, league_avg, home_bias=1.0, away_bias=1.0)


def debutant_prediction(
    proven_name: str,
    proven_avg_scored: float,
    proven_avg_conceded: float,
    proven_matches: int,
    proven_is_home: bool,
    league_avg: float,
    debutant_label: str = "debutant",
) -> PoissonPrediction:
    """
    首秀球队的特殊预测。

    成熟球队保留真实数据，首秀球队用联赛均值折价。
    对齐 Rust build_context_one_sided 的首秀下狗先验逻辑。

    Args:
        proven_name: 有历史记录的球队名
        proven_avg_scored: 场均进球
        proven_avg_conceded: 场均失球
        proven_matches: 比赛数
        proven_is_home: 是否为名义主队
        league_avg: 联赛场均进球
        debutant_label: 首秀球队标签

    Returns:
        PoissonPrediction
    """
    home_stats = TeamStats(
        name=proven_name if proven_is_home else debutant_label,
        matches=proven_matches if proven_is_home else 0,
        avg_scored=proven_avg_scored if proven_is_home else league_avg,
        avg_conceded=proven_avg_conceded if proven_is_home else league_avg,
    )
    away_stats = TeamStats(
        name=debutant_label if proven_is_home else proven_name,
        matches=0 if proven_is_home else proven_matches,
        avg_scored=league_avg if proven_is_home else proven_avg_scored,
        avg_conceded=league_avg if proven_is_home else proven_avg_conceded,
    )

    home_form = compute_team_form(home_stats, is_host=False)
    away_form = compute_team_form(away_stats, is_host=False)

    h2h = HeadToHead()
    baseline = compute_baseline(home_form, away_form, h2h, league_avg)

    # 应用首秀下狗先验
    if proven_is_home:
        adj_home = baseline.expected_home_goals * DEBUTANT_OPPONENT_BOOST
        adj_away = baseline.expected_away_goals * DEBUTANT_STRENGTH
    else:
        adj_home = baseline.expected_home_goals * DEBUTANT_STRENGTH
        adj_away = baseline.expected_away_goals * DEBUTANT_OPPONENT_BOOST

    result = finalize_baseline(adj_home, adj_away, float(proven_matches))
    result.confidence = max(0.22, min(0.75, result.confidence * 0.55))
    return result


def wilson_confidence_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float, float]:
    """
    Wilson score 置信区间（用于置信度报告）。
    对齐 Rust sim.rs 中的 wilson 函数。

    Args:
        successes: 成功次数
        trials: 总试验次数
        z: Z-score（默认 1.96 对应 95% 置信区间）

    Returns:
        (p, lo, hi): 点估计、95%下限、95%上限
    """
    if trials <= 0:
        return (0.0, 0.0, 1.0)

    n = float(trials)
    p = successes / n
    z2 = z * z

    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))) / denom

    return (
        _round3(p),
        _round3(max(0.0, centre - half)),
        _round3(min(1.0, centre + half)),
    )


# =============================================================================
# DixonColesModel 类（Phase 4 — 赔率偏差集成）
# =============================================================================


class DixonColesModel:
    """
    Dixon-Coles 泊松模型类封装。

    提供不可变的 with_odds_bias() 方法和 predict_match() 方法。
    封装 predict_match 函数，添加赔率偏差因子支持。
    """

    def __init__(self, ctx: Optional[MatchContext] = None):
        """
        Args:
            ctx: 比赛上下文（可选，可在predict时传入）
        """
        self._ctx = ctx
        self._home_bias: float = 1.0
        self._away_bias: float = 1.0

    def with_odds_bias(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
    ) -> "DixonColesModel":
        """
        返回一个新的model副本，λ已根据市场赔率调整。

        转换逻辑:
            1. 去除庄家抽水(overround): 将1/odds归一化为概率
            2. 将主胜概率映射为主队λ偏差因子:
               - 基准: 主胜概率40%对应偏差1.0
               - 每±10%主胜概率 → λ偏差±0.25
               - 钳制范围 [0.6, 1.8]
            3. 同样的逻辑映射客胜概率到客队λ偏差因子
            4. 返回深拷贝的model，λ值乘以对应偏差因子
            5. 不修改原model（不可变模式）

        Args:
            odds_home: 主胜赔率（如1.42表示墨西哥胜赔1.42）
            odds_draw: 平局赔率
            odds_away: 客胜赔率

        Returns:
            DixonColesModel: 新的model副本（带偏差因子）
        """
        home_bias, away_bias = compute_odds_bias(odds_home, odds_draw, odds_away)

        # 返回副本（不可变模式）
        clone = DixonColesModel(ctx=self._ctx)
        clone._home_bias = home_bias
        clone._away_bias = away_bias
        return clone

    def predict_match(
        self,
        ctx: Optional[MatchContext] = None,
        home_bias: Optional[float] = None,
        away_bias: Optional[float] = None,
    ) -> PoissonPrediction:
        """
        运行泊松引擎预测，应用偏差因子。

        Args:
            ctx: 比赛上下文（覆盖初始化时传入的值）
            home_bias: 主队λ偏差（覆盖with_odds_bias设置的值）
            away_bias: 客队λ偏差

        Returns:
            PoissonPrediction
        """
        match_ctx = ctx if ctx is not None else self._ctx
        if match_ctx is None:
            raise ValueError("DixonColesModel.predict_match 需要 MatchContext")

        hb = home_bias if home_bias is not None else self._home_bias
        ab = away_bias if away_bias is not None else self._away_bias

        return predict_match(match_ctx, home_bias=hb, away_bias=ab)

    @property
    def home_bias(self) -> float:
        """当前主队偏差因子"""
        return self._home_bias

    @property
    def away_bias(self) -> float:
        """当前客队偏差因子"""
        return self._away_bias

    def __repr__(self) -> str:
        return (
            f"DixonColesModel(home_bias={self._home_bias:.3f}, "
            f"away_bias={self._away_bias:.3f})"
        )


# =============================================================================
# 内部工具
# =============================================================================

def _round2(x: float) -> float:
    """四舍五入到2位小数（对齐 Rust round2）"""
    return round(x * 100.0) / 100.0


# =============================================================================
# 赔率偏差计算（Phase 4 — 竞彩赔率接入）
# =============================================================================

def compute_odds_bias(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> Tuple[float, float]:
    """
    从市场赔率计算泊松λ偏差因子。

    转换逻辑:
        1. 去除庄家抽水(overround)：将1/odds归一化为概率
        2. 主胜概率 → 主队λ偏差:
           - 基准: 主胜概率40%对应偏差1.0
           - 每±10%主胜概率 → λ偏差±0.25
           - 钳制范围 [0.6, 1.8]
        3. 同样逻辑映射客胜概率 → 客队λ偏差

    Args:
        odds_home: 主胜赔率（如1.42）
        odds_draw: 平局赔率
        odds_away: 客胜赔率

    Returns:
        Tuple[float, float]: (home_bias, away_bias)
    """
    if odds_home <= 0 or odds_draw <= 0 or odds_away <= 0:
        return (1.0, 1.0)

    # 1. 去除庄家抽水：归一化赔率倒数
    inv_home = 1.0 / odds_home
    inv_draw = 1.0 / odds_draw
    inv_away = 1.0 / odds_away
    total_inv = inv_home + inv_draw + inv_away

    if total_inv <= 0:
        return (1.0, 1.0)

    prob_home = inv_home / total_inv
    prob_away = inv_away / total_inv

    # 2. 映射到λ偏差因子
    # 基准: 40%主胜概率 → 偏差1.0
    # 每±10% → ±0.25
    home_bias = 1.0 + (prob_home - 0.40) / 0.10 * 0.25
    away_bias = 1.0 + (prob_away - 0.40) / 0.10 * 0.25

    # 钳制范围 [0.6, 1.8]
    home_bias = max(0.6, min(1.8, home_bias))
    away_bias = max(0.6, min(1.8, away_bias))

    return (home_bias, away_bias)


def _round3(x: float) -> float:
    """四舍五入到3位小数（对齐 Rust round3）"""
    return round(x * 1000.0) / 1000.0


def set_seed(seed: int) -> None:
    """
    设置 numpy 随机种子，确保确定性运行。
    注意：engine_poisson 本身是纯数学计算（不涉及随机数），
    此函数为外部 MC 模拟等做接口预留。
    """
    np.random.seed(seed)
