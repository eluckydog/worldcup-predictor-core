"""
worldcup-predictor-core — 2026 世界杯预测器双引擎核心

架构：
    ┌─────────────────────────────────────────────────────┐
    │                    用户请求                          │
    │                         ▼                           │
    │  ┌───── 特征提取 ─────┐    ┌──── 执行器 ────────┐   │
    │  │ MatchContext        │    │ fuse(poisson,      │   │
    │  │  + TeamStats        │    │       causal,      │   │
    │  │  + HeadToHead       │    │       mode)        │   │
    │  │  + RecentMatch      │    └───────┬────────────┘   │
    │  └────────┬────────────┘            │                │
    │           ▼                         ▼                │
    │  ┌──────────────── Poisson ────┐                    │
    │  │ engine_poisson.predict()    │                     │
    │  ├───────── Causal ───────────┤                    │
    │  │ engine_causal.predict()    │                     │
    │  ├───────── Selector ─────────┤                    │
    │  │ selector.select_engine()   │                     │
    │  └────────────────────────────┘                    │
    │                         ▼                           │
    │              MatchPrediction                        │
    └─────────────────────────────────────────────────────┘

四种运行模式：
    classic     — 纯泊松（backtest 基准）
    causal-only — 纯因果
    auto        — 选择门决定 + 加权融合（默认推荐）
    debug       — 全输出
"""

from .data_types import (
    TeamStats,
    TeamForm,
    PoissonPrediction,
    CausalPrediction,
    CausalSignal,
    SelectorResult,
    SelectorScore,
    MatchPrediction,
    MatchContext,
    HeadToHead,
    RecentMatch,
)
from .engine_poisson import (
    DIXON_COLES_RHO,
    HOST_EDGE,
    DEBUTANT_STRENGTH,
    DEBUTANT_OPPONENT_BOOST,
    predict_match as poisson_predict,
    predict_from_raw as poisson_predict_from_raw,
    debutant_prediction as poisson_debutant_prediction,
    compute_baseline,
    finalize_baseline,
    joint_score_grid,
    poisson_pmf,
    dixon_coles_tau,
    strength,
    wilson_confidence_interval,
    set_seed as poisson_set_seed,
)
from .engine_causal import (
    FootballCausalEngine,
    CausalEngineConfig,
    MatchDataProvider,
    MockDataProvider,
)
from .selector import select_engine
from .fusion import fuse
from .monte_carlo import (
    normal_simulation,
    conditional_simulation,
    intervention_simulation,
    simulate_from_prediction,
    verify_determinism,
)
from .bayesian import (
    beta_binomial_update,
    combined_estimate,
    weak_uniform_prior,
    skeptical_prior,
    debutant_prior,
    host_nation_prior,
    TeamBelief,
    BeliefTracker,
    PosteriorState,
)
from .team_resolver import (
    resolve_team_name,
    is_debutant,
    is_host_nation,
    get_group,
    get_fifa_rank,
    list_teams_in_group,
    get_all_teams,
    describe_team,
)

__all__ = [
    # 数据模型
    "TeamStats",
    "TeamForm",
    "PoissonPrediction",
    "CausalPrediction",
    "CausalSignal",
    "SelectorResult",
    "SelectorScore",
    "MatchPrediction",
    "MatchContext",
    "HeadToHead",
    "RecentMatch",
    "ProbEstimate",
    "ScoreProb",
    "SimulationResult",
    # 泊松引擎
    "poisson_predict",
    "poisson_predict_from_raw",
    "poisson_debutant_prediction",
    "compute_baseline",
    "finalize_baseline",
    "joint_score_grid",
    "poisson_pmf",
    "dixon_coles_tau",
    "strength",
    "wilson_confidence_interval",
    "poisson_set_seed",
    "DIXON_COLES_RHO",
    "HOST_EDGE",
    "DEBUTANT_STRENGTH",
    "DEBUTANT_OPPONENT_BOOST",
    # 因果引擎
    "FootballCausalEngine",
    "CausalEngineConfig",
    "MatchDataProvider",
    "MockDataProvider",
    # 选择门
    "select_engine",
    # 融合层
    "fuse",
    # MC 模拟
    "normal_simulation",
    "conditional_simulation",
    "intervention_simulation",
    "simulate_from_prediction",
    "verify_determinism",
    # 贝叶斯
    "beta_binomial_update",
    "combined_estimate",
    "weak_uniform_prior",
    "skeptical_prior",
    "debutant_prior",
    "host_nation_prior",
    "TeamBelief",
    "BeliefTracker",
    "PosteriorState",
    # 球队映射
    "resolve_team_name",
    "is_debutant",
    "is_host_nation",
    "get_group",
    "get_fifa_rank",
    "list_teams_in_group",
    "get_all_teams",
    "describe_team",
]
