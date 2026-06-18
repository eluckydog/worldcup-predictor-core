"""
worldcup-predictor-core 共享数据类型

定义所有引擎间流通的数据结构，使用 dataclass + 完整类型提示。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# 基础统计结构（移植自对方 Rust 项目 model.rs）
# =============================================================================

@dataclass
class TeamStats:
    """球队历史统计数据"""
    name: str
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    avg_scored: float = 0.0
    avg_conceded: float = 0.0
    clean_sheets: int = 0
    first_year: Optional[int] = None
    last_year: Optional[int] = None


@dataclass
class HeadToHead:
    """两队历史交锋数据"""
    matches: int = 0
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_goals: int = 0
    b_goals: int = 0


@dataclass
class RecentMatch:
    """近期比赛记录"""
    opponent: str = ""
    goals_for: int = 0
    goals_against: int = 0
    year: int = 0
    stage: str = ""
    is_home: bool = False


# =============================================================================
# 泊松引擎输出
# =============================================================================

@dataclass
class TeamForm:
    """加权后的球队攻防形态"""
    attack: float  # 场均进球（加权）
    defense: float  # 场均失球（加权）
    eff_matches: float  # 有效比赛数（权重和）
    is_host: bool = False  # 是否为2026主办国


@dataclass
class PoissonPrediction:
    """泊松引擎的完整预测输出"""
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    expected_total_goals: float = 0.0
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    prob_over_2_5: float = 0.0
    prob_btts: float = 0.0
    predicted_home_score: int = 0
    predicted_away_score: int = 0
    predicted_total_goals: int = 0
    confidence: float = 0.0
    likelihood: float = 0.0  # 对数似然，用于选择门拟合优度


# =============================================================================
# 因果引擎输出
# =============================================================================

@dataclass
class CausalSignal:
    """因果引擎检测到的信号"""
    deviation_score: float = 0.0  # 偏离度(0-100)，越高越异常
    structure_break: bool = False  # 是否存在结构突变
    break_magnitude: float = 0.0  # 突变幅度
    adjustment_factor_home: Optional[float] = None  # 主场调整乘数
    adjustment_factor_away: Optional[float] = None  # 客场调整乘数
    factor_effects: dict = field(default_factory=dict)  # 各因子的 ATE
    dag_coverage: float = 0.0  # DAG 覆盖度(0-1)


@dataclass
class CausalPrediction:
    """因果引擎的完整预测输出"""
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    expected_total_goals: float = 0.0
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    confidence: float = 0.0
    signal: Optional[CausalSignal] = None


# =============================================================================
# 选择门输出
# =============================================================================

@dataclass
class SelectorScore:
    """选择门评分明细"""
    historical_data_volume: float = 0.0  # 历史数据量(0-100)
    dag_coverage: float = 0.0  # DAG覆盖度(0-100)
    poisson_goodness_of_fit: float = 0.0  # 泊松拟合优度(0-100)
    structure_break_score: float = 0.0  # 结构突变分数(0-100)


@dataclass
class SelectorResult:
    """选择门决策结果"""
    poisson_score: float = 0.0  # 泊松引擎总分(0-100)
    causal_score: float = 0.0  # 因果引擎总分(0-100)
    primary_engine: str = "poisson"  # "poisson" | "causal"
    secondary_engine: str = "causal"  # 互补引擎
    primary_weight: float = 0.7  # 主导权重
    secondary_weight: float = 0.3  # 互补权重
    score_gap: float = 0.0  # 分数差距
    detail: Optional[SelectorScore] = None  # 各维度评分明细


# =============================================================================
# 融合层输出
# =============================================================================

@dataclass
class MatchPrediction:
    """最终预测输出"""
    mode: str = "auto"  # "classic" | "causal-only" | "auto" | "debug"

    # 引擎选择信息
    primary_engine: str = "poisson"
    primary_weight: float = 0.7
    secondary_engine: str = "causal"
    secondary_weight: float = 0.3

    # 选择门评分
    selector_scores: Optional[SelectorResult] = None

    # 最终预测值
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    expected_total_goals: float = 0.0
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    confidence: float = 0.0

    # 引擎分解（debug 模式下填充）
    poisson_raw: Optional[PoissonPrediction] = None
    causal_raw: Optional[CausalPrediction] = None
    causal_signals: Optional[CausalSignal] = None


# =============================================================================
# 比赛上下文（用于引擎输入）
# =============================================================================

@dataclass
class MatchContext:
    """一场比赛的全量上下文数据"""
    home: TeamStats
    away: TeamStats
    head_to_head: HeadToHead
    home_recent: list = field(default_factory=list)
    away_recent: list = field(default_factory=list)
    league_avg_goals: float = 1.0

    # 扩展特征（因果引擎用）
    home_form_attack: float = 0.0
    home_form_defense: float = 0.0
    away_form_attack: float = 0.0
    away_form_defense: float = 0.0
    home_attack_strength: float = 1.0
    home_defense_strength: float = 1.0
    away_attack_strength: float = 1.0
    away_defense_strength: float = 1.0

    # 特殊因素（可选）
    temperature_celsius: Optional[float] = None
    referee_style: Optional[str] = None  # "strict", "lenient"
    key_injury_home: bool = False
    key_injury_away: bool = False
    manager_change_home: bool = False
    manager_change_away: bool = False
    is_knockout: bool = False
    rest_days_diff: int = 0  # 正数表示主队休息更多天


# =============================================================================
# MC 模拟输出（Phase 2）
# =============================================================================


@dataclass
class ProbEstimate:
    """概率估计及其 Wilson 置信区间"""
    p: float = 0.0  # 点估计
    lo: float = 0.0  # 95% 置信下限
    hi: float = 0.0  # 95% 置信上限


@dataclass
class ScoreProb:
    """具体比分的概率"""
    score: str = ""  # 比分字符串，如 "2:1"
    probability: float = 0.0


@dataclass
class SimulationResult:
    """蒙特卡洛模拟完整输出"""
    # --- 模拟元信息 ---
    trials: int = 0
    lambda_home: float = 0.0  # 使用的泊松参数
    lambda_away: float = 0.0
    mode: str = "normal"  # "normal" | "conditional" | "intervention"

    # --- 预期进球 ---
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    home_goals_sd: float = 0.0
    away_goals_sd: float = 0.0

    # --- 1X2 概率（含置信区间） ---
    home_win: Optional[ProbEstimate] = None
    draw: Optional[ProbEstimate] = None
    away_win: Optional[ProbEstimate] = None

    # --- 衍生市场 ---
    over_1_5: float = 0.0
    over_2_5: float = 0.0
    over_3_5: float = 0.0
    btts: float = 0.0
    home_clean_sheet: float = 0.0
    away_clean_sheet: float = 0.0

    # --- 比分分布 ---
    most_likely_score: str = ""
    top_scores: list = field(default_factory=list)  # list[ScoreProb]
    samples: list = field(default_factory=list)  # list[tuple[int, int]]
    top_outcomes: dict = field(default_factory=dict)  # {"score_str": prob}


# =============================================================================
# Phase 3: BPD 非理性信号 + EPG 路径优化
# =============================================================================


@dataclass
class IrrationalitySignal:
    """BPD 市场非理性检测信号

    用于检测市场赔率与模型预测之间的系统性偏差。
    """
    score: float = 0.0  # 非理性分数 (0-100)，越高越非理性
    direction: str = "neutral"  # "overvalued_home" / "overvalued_away" / "undervalued_home" / "undervalued_away" / "neutral"
    adjusted_prob: Dict[str, float] = field(default_factory=lambda: {"home": 0.0, "draw": 0.0, "away": 0.0})  # 调整后的概率
    market_implied_prob: Dict[str, float] = field(default_factory=lambda: {"home": 0.0, "draw": 0.0, "away": 0.0})  # 市场隐含概率
    entropy: float = 0.0  # 分布熵
    bpd_energy: float = 0.0  # 玻尔兹曼能量


@dataclass
class PathwaySignal:
    """EPG 路径优化信号

    基于进化策略梯度搜索的最优预测调整路径。
    """
    path: List[dict] = field(default_factory=list)  # 优化路径序列
    final_adjustment: Dict[str, float] = field(default_factory=dict)  # 最终调整量
    convergence: float = 0.0  # 收敛度 (0-100)
    path_complexity: int = 1  # 路径复杂度 (1-5)
    recommendations: List[str] = field(default_factory=list)  # 文字建议


# =============================================================================
# Phase 5: 多源赔率融合系统
# =============================================================================


@dataclass
class OddsRecord:
    """单源赔率记录

    包含一个赔率源的原始数据和去水概率。
    """
    source_name: str = ""  # 来源名，如 "500_AVG" / "JC_SP" / "FOREIGN_BET365"
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    prob_home: float = 0.0  # 去水后的隐含概率
    prob_draw: float = 0.0
    prob_away: float = 0.0
    juice: float = 0.0  # 抽水率 (overround - 1.0)
    weight: float = 1.0  # 融合权重
    timestamp: str = ""  # 抓取时间


@dataclass
class MultiSourceOdds:
    """多源赔率集合

    包含比赛双方的名称和所有可用赔率源的原始+去水数据。
    """
    home_team: str = ""
    away_team: str = ""
    sources: list = field(default_factory=list)  # list[OddsRecord]
    fetch_time: str = ""


@dataclass
class OddsFusionResult:
    """赔率融合结果

    多源赔率去水、加权融合后的最终输出。
    """
    # 融合后的概率（已去除各源抽水并按权重融合）
    fused_home_prob: float = 0.0
    fused_draw_prob: float = 0.0
    fused_away_prob: float = 0.0

    # 等效赔率（基于融合概率反算，以最低抽水率源为基准）
    equiv_home_odds: float = 0.0
    equiv_draw_odds: float = 0.0
    equiv_away_odds: float = 0.0

    # 共识分歧度指标
    divergence: float = 0.0  # 各源隐含概率的标准差均值 (0~1)
    max_divergence: float = 0.0  # 最大单源偏离
    consensus_count: int = 0  # 参与融合的源数

    # 各源详情（用于debug展示）
    source_details: list = field(default_factory=list)  # list[OddsRecord]

    # λ偏差因子（与现有compute_odds_bias输出对齐）
    home_bias: float = 1.0
    away_bias: float = 1.0

    # 原始三元组（向后兼容）
    as_tuple: tuple = (0.0, 0.0, 0.0)  # (equiv_home_odds, equiv_draw_odds, equiv_away_odds)
