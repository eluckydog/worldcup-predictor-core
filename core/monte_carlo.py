"""
蒙特卡洛模拟引擎 — 2026 世界杯比分分布模拟

对齐对方 Rust sim.rs 的三层条件分支：
- normal_simulation  — 从 Dixon-Coles 联合概率网格逆变换采样
- conditional_simulation — 给定 λ_home/λ_away 直接模拟比分
- intervention_simulation — 给定因果调整后的 λ 模拟"假如何因子不同"

所有模拟使用固定种子保证确定性（对齐 Rust 的 SEED=0x5DEE_CE66_D00D_1234 行为）。
Wilson 置信区间复用 engine_poisson.wilson_confidence_interval。
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_types import (
    ProbEstimate,
    ScoreProb,
    SimulationResult,
)
from .engine_poisson import joint_score_grid, JOINT_GRID_SIZE, wilson_confidence_interval


# =============================================================================
# 常量
# =============================================================================

DEFAULT_TRIALS: int = 50_000  # 对标 sim.rs 默认 50000 次 MC 模拟
DEFAULT_SEED: int = 42  # 固定种子保证确定性（对标 sim.rs 的 SEED=0x5DEE_CE66_D00D_1234）
JOINT_N: int = JOINT_GRID_SIZE  # 10, 与 joint_score_grid 对齐
TOP_SCORES_COUNT: int = 10  # 输出前 10 种最可能比分


# =============================================================================
# 核心模拟函数
# =============================================================================

def _build_grid_cdf(
    exp_home: float,
    exp_away: float,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, float]:
    """
    从 Dixon-Coles 联合概率网格构建逆变换采样用的 CDF。

    Args:
        exp_home: 主场预期进球
        exp_away: 客场预期进球

    Returns:
        (grid, cells, cdf):
            grid: (N+1)×(N+1) 概率网格
            cells: [(i, j), ...] 每个网格座标
            cdf:   [p0, p0+p1, ...] 累积分布
    """
    grid = joint_score_grid(exp_home, exp_away)
    n = JOINT_N

    cells: List[Tuple[int, int]] = []
    cdf_vals: List[float] = []
    acc = 0.0

    for i in range(n + 1):
        for j in range(n + 1):
            p = float(grid[i, j])
            if p > 0.0:
                acc += p
                cells.append((i, j))
                cdf_vals.append(acc)

    total = cdf_vals[-1] if cdf_vals else 1.0
    cdf_arr = np.array(cdf_vals, dtype=np.float64)

    return grid, cells, cdf_arr, total


def _sample_from_cdf(
    rng: np.random.RandomState,
    cells: List[Tuple[int, int]],
    cdf: np.ndarray,
    total: float,
    trials: int,
) -> np.ndarray:
    """
    从 CDF 批量采样，返回 (trials, 2) 的比分数组。

    Args:
        rng: numpy RandomState
        cells: [(i,j), ...]
        cdf: 累积概率数组
        total: 总概率（接近 1.0）
        trials: 采样数

    Returns:
        shape (trials, 2) 的整数比分数组
    """
    u = rng.uniform(0.0, total, size=trials)
    # 二分查找 CDF
    idx = np.searchsorted(cdf, u, side="right")
    idx = np.clip(idx, 0, len(cells) - 1)

    scores = np.array([cells[i] for i in idx], dtype=np.int32)
    return scores


def _compute_statistics(
    scores: np.ndarray,
    trials: int,
    exp_home: float,
    exp_away: float,
) -> SimulationResult:
    """
    从采样比分计算完整统计量。

    直接对齐 sim.rs 的输出字段。
    """
    t = float(trials)
    home_scores = scores[:, 0]
    away_scores = scores[:, 1]

    # --- 计数 ---
    hw_count = int(np.sum(home_scores > away_scores))
    dr_count = int(np.sum(home_scores == away_scores))
    aw_count = int(np.sum(home_scores < away_scores))

    sum_h = float(np.sum(home_scores))
    sum_a = float(np.sum(away_scores))
    sum_h2 = float(np.sum(home_scores.astype(np.float64) ** 2))
    sum_a2 = float(np.sum(away_scores.astype(np.float64) ** 2))

    total_goals = home_scores + away_scores
    o15 = int(np.sum(total_goals >= 2))
    o25 = int(np.sum(total_goals >= 3))
    o35 = int(np.sum(total_goals >= 4))
    btts_count = int(np.sum((home_scores >= 1) & (away_scores >= 1)))
    hcs = int(np.sum(away_scores == 0))  # home clean sheet = away failed to score
    acs = int(np.sum(home_scores == 0))

    # --- 预期进球与标准差 ---
    mean_h = sum_h / t
    mean_a = sum_a / t
    var_h = max(0.0, sum_h2 / t - mean_h * mean_h)
    var_a = max(0.0, sum_a2 / t - mean_a * mean_a)
    sd_h = math.sqrt(var_h)
    sd_a = math.sqrt(var_a)

    # --- 1X2 Wilson CI ---
    def _wilson(successes: int, n: int) -> ProbEstimate:
        p, lo, hi = wilson_confidence_interval(successes, n, z=1.96)
        return ProbEstimate(p=round(p, 3), lo=round(lo, 3), hi=round(hi, 3))

    hw_ci = _wilson(hw_count, trials)
    dr_ci = _wilson(dr_count, trials)
    aw_ci = _wilson(aw_count, trials)

    # --- 排名分布 ---
    # 统计每种比分出现次数
    unique_scores: Dict[Tuple[int, int], int] = {}
    for idx in range(trials):
        key = (int(scores[idx, 0]), int(scores[idx, 1]))
        unique_scores[key] = unique_scores.get(key, 0) + 1

    sorted_scores = sorted(unique_scores.items(), key=lambda x: -x[1])
    top_scores_list: List[ScoreProb] = []
    top_outcomes_dict: Dict[str, float] = {}

    for (h, a), cnt in sorted_scores[:TOP_SCORES_COUNT]:
        prob = cnt / t
        score_str = f"{h}:{a}"
        top_scores_list.append(ScoreProb(score=score_str, probability=round(prob, 4)))
        top_outcomes_dict[score_str] = round(prob, 4)

    most_likely = top_scores_list[0].score if top_scores_list else "0:0"

    # --- 样本列表（轻量，仅保留 top_scores 路由） ---
    sample_list: List[Tuple[int, int]] = [
        (int(scores[i, 0]), int(scores[i, 1])) for i in range(min(trials, 500))
    ]

    return SimulationResult(
        trials=trials,
        lambda_home=exp_home,
        lambda_away=exp_away,
        mode="normal",
        expected_home_goals=round(mean_h, 2),
        expected_away_goals=round(mean_a, 2),
        home_goals_sd=round(sd_h, 2),
        away_goals_sd=round(sd_a, 2),
        home_win=hw_ci,
        draw=dr_ci,
        away_win=aw_ci,
        over_1_5=round(o15 / t, 3),
        over_2_5=round(o25 / t, 3),
        over_3_5=round(o35 / t, 3),
        btts=round(btts_count / t, 3),
        home_clean_sheet=round(hcs / t, 3),
        away_clean_sheet=round(acs / t, 3),
        most_likely_score=most_likely,
        top_scores=top_scores_list,
        samples=sample_list,
        top_outcomes=top_outcomes_dict,
    )


# =============================================================================
# 三层模拟接口
# =============================================================================

def normal_simulation(
    exp_home: float,
    exp_away: float,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> SimulationResult:
    """
    【普通模拟】从 Dixon-Coles 联合概率网格逆变换采样。

    这是最标准的模拟路径——使用 joint_score_grid 的完整修正网格，
    保持主/客场泊松的 Dixon-Coles 相关性。

    Args:
        exp_home: 主场预期进球
        exp_away: 客场预期进球
        trials: 模拟次数（默认 50000，对齐 sim.rs）
        seed: 随机种子（默认 42，确定性运行）

    Returns:
        SimulationResult: 完整模拟结果
    """
    rng = np.random.RandomState(seed)
    grid, cells, cdf, total = _build_grid_cdf(exp_home, exp_away)
    scores = _sample_from_cdf(rng, cells, cdf, total, trials)
    result = _compute_statistics(scores, trials, exp_home, exp_away)
    result.mode = "normal"
    return result


def conditional_simulation(
    lambda_home: float,
    lambda_away: float,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> SimulationResult:
    """
    【条件模拟】给定 λ_home / λ_away 直接模拟比分分布。

    与 normal_simulation 行为相同（都从联合网格采样），
    但语义上表示为"已知基线 λ 后各比分出现的概率"，
    用于 causal 引擎提供调整后的 λ 时的后续模拟。

    Args:
        lambda_home: 主队泊松 λ
        lambda_away: 客队泊松 λ
        trials: 模拟次数
        seed: 随机种子

    Returns:
        SimulationResult
    """
    result = normal_simulation(lambda_home, lambda_away, trials, seed)
    result.mode = "conditional"
    result.lambda_home = lambda_home
    result.lambda_away = lambda_away
    return result


def intervention_simulation(
    base_lambda_home: float,
    base_lambda_away: float,
    causal_adjustments: Optional[Dict[str, float]] = None,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> SimulationResult:
    """
    【干预模拟】给定基准 λ + 因果调整，模拟"假如何因子不同"。

    模拟 "what if" 场景：如果因果引擎检测到某因子（高温、伤病等）
    存在或不存在，预期进球会如何变化。
    输出的相比 normal/conditional 加了调整后的 λ 信息。

    Args:
        base_lambda_home: 基准主队泊松 λ（未经因果调整）
        base_lambda_away: 基准客队泊松 λ
        causal_adjustments: 调整乘数字典
            e.g. {"home_adjustment": 0.85, "away_adjustment": 1.15}
            不提供则退化为 conditional 模拟
        trials: 模拟次数
        seed: 随机种子

    Returns:
        SimulationResult
    """
    adj_home = 1.0
    adj_away = 1.0

    if causal_adjustments:
        adj_home = causal_adjustments.get("home_adjustment", 1.0)
        adj_away = causal_adjustments.get("away_adjustment", 1.0)

    exp_home = base_lambda_home * adj_home
    exp_away = base_lambda_away * adj_away

    result = normal_simulation(exp_home, exp_away, trials, seed)
    result.mode = "intervention"
    result.lambda_home = exp_home
    result.lambda_away = exp_away

    return result


# =============================================================================
# 便捷函数
# =============================================================================

def simulate_from_prediction(
    expected_home_goals: float,
    expected_away_goals: float,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    mode: str = "normal",
    causal_adjustments: Optional[Dict[str, float]] = None,
) -> SimulationResult:
    """
    统一入口：根据 mode 自动选择模拟路径。

    这是 main.py 和外部调用者的推荐入口。

    Args:
        expected_home_goals: 预期主队进球
        expected_away_goals: 预期客队进球
        trials: 模拟次数
        seed: 随机种子
        mode: 模拟模式 ("normal" | "conditional" | "intervention")
        causal_adjustments: 干预模式用的调整乘数

    Returns:
        SimulationResult
    """
    if mode == "intervention":
        return intervention_simulation(
            expected_home_goals, expected_away_goals,
            causal_adjustments=causal_adjustments,
            trials=trials, seed=seed,
        )
    elif mode == "conditional":
        return conditional_simulation(
            expected_home_goals, expected_away_goals,
            trials=trials, seed=seed,
        )
    else:
        return normal_simulation(
            expected_home_goals, expected_away_goals,
            trials=trials, seed=seed,
        )


# =============================================================================
# 确定性验证
# =============================================================================

def verify_determinism() -> bool:
    """
    验证模拟是确定性的（相同参数 → 相同结果）。

    Returns:
        True 如果两次模拟结果一致
    """
    a = normal_simulation(1.8, 1.0, trials=5000, seed=42)
    b = normal_simulation(1.8, 1.0, trials=5000, seed=42)

    return (
        a.home_win.p == b.home_win.p
        and a.draw.p == b.draw.p
        and a.away_win.p == b.away_win.p
        and a.most_likely_score == b.most_likely_score
        and a.expected_home_goals == b.expected_home_goals
    )
