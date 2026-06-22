"""
因果引擎包装层 — 调用 causal_engine.py 已有能力的足球专用适配器

职责：
- 构建足球比赛因果 DAG
- 从历史数据估计各因子 ATE（PSM → IPW → 均值差三级回退）
- do_intervention 接口：条件变化 → 因果调整乘数
- 结构突变检测：窗口比较
- 偏离度计算：特征偏离分数

本模块不依赖真实数据库，数据接口通过 MatchDataProvider 抽象。
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .data_types import (
    MatchContext,
    CausalPrediction,
    CausalSignal,
    TeamStats,
)

# 尝试导入 causal_engine（可选）
try:
    import sys
    import os
    _CAUSAL_PATH = os.path.expanduser(
        "~/.qclaw/skills/prob-contradiction-system/scripts"
    )
    if _CAUSAL_PATH not in sys.path:
        sys.path.insert(0, _CAUSAL_PATH)
    from causal_engine import CausalDAG, ATEEstimator, InterventionEngine, ThresholdDetector
    HAS_CAUSAL_ENGINE = True
except ImportError:
    HAS_CAUSAL_ENGINE = False


# =============================================================================
# 数据抽象层（Phase 1 用 Mock，Phase 2 对接真实数据源）
# =============================================================================

class MatchDataProvider(ABC):
    """
    抽象数据提供者接口。
    Phase 1 使用 MockDataProvider；Phase 2 对接 SQLite。
    """

    @abstractmethod
    def get_team_stats(self, team_name: str) -> Optional[TeamStats]:
        """获取球队统计"""
        ...

    @abstractmethod
    def get_head_to_head(self, home: str, away: str) -> Dict[str, Any]:
        """获取历史交锋数据"""
        ...

    @abstractmethod
    def get_recent_matches(self, team_name: str, n: int = 10) -> List[Dict[str, Any]]:
        """获取近期比赛"""
        ...

    @abstractmethod
    def get_league_avg_goals(self) -> float:
        """获取联赛场均进球"""
        ...

    @abstractmethod
    def get_historical_match_data(self) -> pd.DataFrame:
        """
        获取历史比赛数据 DataFrame，用于因果估计。
        每行包含：home_team, away_team, home_goals, away_goals, year, stage,
        temperature, home_rest_days, away_rest_days 等特征。
        """
        ...


class MockDataProvider(MatchDataProvider):
    """Phase 1 的 Mock 数据提供者"""

    def __init__(self) -> None:
        self._league_avg = 1.2

    def get_team_stats(self, team_name: str) -> Optional[TeamStats]:
        return TeamStats(name=team_name, matches=20, wins=10, draws=5, losses=5,
                         goals_for=35, goals_against=25, avg_scored=1.75,
                         avg_conceded=1.25, clean_sheets=5, first_year=1998,
                         last_year=2022)

    def get_head_to_head(self, home: str, away: str) -> Dict[str, Any]:
        return {"matches": 3, "a_wins": 1, "b_wins": 1, "draws": 1,
                "a_goals": 4, "b_goals": 3}

    def get_recent_matches(self, team_name: str, n: int = 10) -> List[Dict[str, Any]]:
        return []

    def get_league_avg_goals(self) -> float:
        return self._league_avg

    def get_historical_match_data(self) -> pd.DataFrame:
        """生成模拟历史比赛数据"""
        np.random.seed(42)
        n = 500
        data = {
            "home_attack_strength": np.random.normal(1.0, 0.3, n),
            "away_attack_strength": np.random.normal(1.0, 0.3, n),
            "home_defense_strength": np.random.normal(1.0, 0.3, n),
            "away_defense_strength": np.random.normal(1.0, 0.3, n),
            "h2h_advantage": np.random.normal(0.0, 0.2, n),
            "home_form_recent": np.random.normal(0.0, 0.3, n),
            "away_form_recent": np.random.normal(0.0, 0.3, n),
            "temperature_effect": np.random.normal(0.0, 0.1, n),
            "referee_strictness": np.random.choice([-0.05, 0.0, 0.05], n),
            "rest_advantage": np.random.normal(0.0, 0.1, n),
            "knockout_pressure": np.random.choice([-0.1, 0.0], n, p=[0.3, 0.7]),
            "home_goals": np.zeros(n),
            "away_goals": np.zeros(n),
        }
        df = pd.DataFrame(data)
        # 合成预期进球
        lh = (self._league_avg
              * df["home_attack_strength"]
              * df["away_defense_strength"]
              * (1.0 + df["h2h_advantage"])
              * (1.0 + df["home_form_recent"])
              * (1.0 + df["temperature_effect"])
              * (1.0 + df["referee_strictness"])
              * (1.0 + df["rest_advantage"])
              * (1.0 + df["knockout_pressure"]))
        la = (self._league_avg
              * df["away_attack_strength"]
              * df["home_defense_strength"]
              * (1.0 - df["h2h_advantage"])
              * (1.0 + df["away_form_recent"]))
        df["home_goals"] = np.random.poisson(np.maximum(lh, 0.1))
        df["away_goals"] = np.random.poisson(np.maximum(la, 0.1))
        return df


# =============================================================================
# 因果引擎
# =============================================================================

@dataclass
class CausalEngineConfig:
    """因果引擎配置"""
    use_causal_engine_lib: bool = True  # 是否加载完整 causal_engine
    psm_neighbors: int = 5
    significance_level: float = 0.05
    structure_break_window: int = 20  # 结构突变检测窗口大小
    structure_break_ratio: float = 0.3  # 近期窗口占总历史的比例


DEFAULT_CONFIG = CausalEngineConfig()


class FootballCausalEngine:
    """
    足球比赛因果引擎包装层。

    构建足球专项 DAG，从历史数据估计各因子 ATE，
    提供干预模拟和结构突变检测接口。
    """

    def __init__(
        self,
        provider: Optional[MatchDataProvider] = None,
        config: Optional[CausalEngineConfig] = None,
    ) -> None:
        """
        初始化因果引擎。

        Args:
            provider: 数据提供者（Phase 1 默认 Mock）
            config: 引擎配置
        """
        self.provider = provider or MockDataProvider()
        self.config = config or DEFAULT_CONFIG
        self._causal_lib_available = HAS_CAUSAL_ENGINE and self.config.use_causal_engine_lib
        self._dag = self._build_football_dag()
        self._estimated_effects: Dict[str, float] = {}
        self._historical_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 足球因果 DAG 构建
    # ------------------------------------------------------------------

    def _build_football_dag(self) -> "CausalDAG":
        """Build football causal DAG from class-level constants."""
        if self._causal_lib_available:
            dag = CausalDAG()
        else:
            dag = _SimpleCausalDAG()
        for n in self.DAG_EXOGENOUS:
            dag.add_node(n, is_exogenous=True)
        for n in self.DAG_ENDOGENOUS:
            dag.add_node(n)
        for n in ["expected_home_goals", "expected_away_goals", "home_score", "away_score"]:
            dag.add_node(n)
        for src, dst in self.DAG_EDGES:
            dag.add_edge(src, dst)
        return dag

    # ------------------------------------------------------------------
    # ATE 估计（PSM → IPW → 均值差三级回退）
    # ------------------------------------------------------------------

    def estimate_ates(self) -> Dict[str, float]:
        """
        估计所有可识别因子的 ATE。

        先训练/加载历史数据，然后对 DAG 中每个 treatment 因子
        用 PSM → IPW → 均值差三级回退估计。

        返回: {factor_name: ate_estimate}
        """
        if self._estimated_effects:
            return self._estimated_effects

        data = self._load_training_data()
        if data.empty:
            return {}

        # 需要估计的因子列表
        factors = [
            ("home_key_injury", "expected_home_goals"),
            ("away_key_injury", "expected_away_goals"),
            ("home_manager_change", "expected_home_goals"),
            ("away_manager_change", "expected_away_goals"),
            ("is_knockout", "expected_home_goals"),
            ("is_knockout", "expected_away_goals"),
            ("temperature", "expected_home_goals"),
            ("referee_strictness", "expected_home_goals"),
            ("rest_days_diff", "expected_home_goals"),
        ]

        effects: Dict[str, float] = {}

        for treatment, outcome in factors:
            ate = self._estimate_single_ate(data, treatment, outcome)
            effects[treatment] = ate

        self._estimated_effects = effects
        return effects

    def _estimate_single_ate(
        self, data: pd.DataFrame, treatment: str, outcome: str
    ) -> float:
        """
        对单个因子用三级回退估计 ATE。

        第1级: PSM（倾向得分匹配）— 需要 sklearn
        第2级: IPW（逆概率加权）— 需要 sklearn
        第3级: 简单均值差
        """
        # 尝试 PSM
        if HAS_CAUSAL_ENGINE and self._causal_lib_available:
            try:
                estimator = ATEEstimator(self._dag, data)
                result = estimator.estimate_psm(treatment, outcome, n_neighbors=self.config.psm_neighbors)
                return result.effect_estimate
            except Exception:
                pass

        # 尝试 IPW
        if HAS_CAUSAL_ENGINE and self._causal_lib_available:
            try:
                estimator = ATEEstimator(self._dag, data)
                result = estimator.estimate_ipw(treatment, outcome)
                return result.effect_estimate
            except Exception:
                pass

        # 回退到均值差
        return self._mean_difference_ate(data, treatment, outcome)

    @staticmethod
    def _mean_difference_ate(data: pd.DataFrame, treatment: str, outcome: str) -> float:
        """
        最简 ATE 估计：E[Y|T=1] - E[Y|T=0]

        当 PSM 和 IPW 都不可用时作为兜底。
        """
        if treatment not in data.columns or outcome not in data.columns:
            return 0.0

        # 对二值/连续处理变量做离散化
        treated = data[data[treatment] > data[treatment].median()]
        control = data[data[treatment] <= data[treatment].median()]

        if len(treated) < 2 or len(control) < 2:
            return 0.0

        return float(treated[outcome].mean() - control[outcome].mean())

    def _load_training_data(self) -> pd.DataFrame:
        """加载/生成训练数据"""
        if self._historical_data is not None:
            return self._historical_data
        data = self.provider.get_historical_match_data()
        self._historical_data = data
        return data

    # ------------------------------------------------------------------
    # do_intervention 接口
    # ------------------------------------------------------------------

    def do_intervention(
        self,
        context: MatchContext,
        interventions: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        给定条件变化，返回因果调整乘数。

        Args:
            context: 比赛上下文（当前基线特征）
            interventions: 干预条件字典
                e.g. {"temperature": 35.0, "referee_strictness": 0.1}

        Returns:
            {"home_adjustment": float, "away_adjustment": float, "details": dict}
            调整乘数 > 1.0 表示预期进球上升，< 1.0 表示下降
        """
        effects = self.estimate_ates()
        home_adj = 1.0
        away_adj = 1.0
        details: Dict[str, Any] = {}

        for factor, new_value in interventions.items():
            # 计算相对于基线的调整
            if factor in effects:
                effect = effects[factor]

                # 简单线性调整：ate * (new_value - baseline_value)
                baseline = self._get_baseline_value(context, factor)
                delta = float(new_value) - baseline if isinstance(baseline, (int, float)) else 0.0

                # 对于二值因子（伤病、换帅），delta 即为 ate
                if isinstance(new_value, bool) or factor.endswith("_injury") or factor.endswith("_change"):
                    adj = 1.0 + effect if new_value else 1.0
                else:
                    adj = 1.0 + effect * delta

                if "home" in factor:
                    home_adj *= adj
                elif "away" in factor:
                    away_adj *= adj
                else:
                    home_adj *= adj
                    away_adj *= adj

                details[factor] = {
                    "ate": effect,
                    "baseline": baseline,
                    "new_value": new_value,
                    "adjustment_factor": adj,
                }

        return {
            "home_adjustment": home_adj,
            "away_adjustment": away_adj,
            "details": details,
        }

    @staticmethod
    def _get_baseline_value(context: MatchContext, factor: str) -> Any:
        """获取某个因子的基线值"""
        mapping = {
            "home_key_injury": False,
            "away_key_injury": False,
            "home_manager_change": False,
            "away_manager_change": False,
            "is_knockout": False,
            "temperature": 20.0,
            "referee_strictness": 0.0,
            "rest_days_diff": 0,
        }
        attr_map = {
            "home_key_injury": "key_injury_home",
            "away_key_injury": "key_injury_away",
            "home_manager_change": "manager_change_home",
            "away_manager_change": "manager_change_away",
            "is_knockout": "is_knockout",
            "temperature": "temperature_celsius",
            "referee_strictness": "referee_style",
            "rest_days_diff": "rest_days_diff",
        }
        if factor in attr_map:
            return getattr(context, attr_map[factor], mapping.get(factor))
        return mapping.get(factor, 0.0)

    # ------------------------------------------------------------------
    # 结构突变检测
    # ------------------------------------------------------------------

    def detect_structure_break(self, context: MatchContext) -> Tuple[bool, float]:
        """
        检测球队是否存在结构突变（如换帅后的战术变化、新核心球员等）。

        比较近期窗口与历史窗口的进攻/防守参数变化。

        Args:
            context: 比赛上下文

        Returns:
            (has_break: bool, magnitude: float)
            magnitude > 0.3 视为显著突变
        """
        # 使用两队样本量作为代理：样本差距大的球队可能已经变化
        home_eff = context.home.matches
        away_eff = context.away.matches

        # 简单检测：如果两队比赛数差距大，可能有结构变化
        # 或者通过近期 vs 历史的进球率差异
        if home_eff <= 0 or away_eff <= 0:
            return False, 0.0

        # 攻防比率变化
        home_efficiency = context.home.avg_scored / max(context.home.avg_conceded, 0.1)
        away_efficiency = context.away.avg_scored / max(context.away.avg_conceded, 0.1)

        # 比赛数过少意味着可能有结构变化（重建期）
        match_ratio = min(home_eff, away_eff) / max(home_eff, away_eff, 1)

        magnitude = max(0.0, 1.0 - match_ratio)
        magnitude += max(0.0, abs(home_efficiency - away_efficiency) * 0.1 - 0.2)

        # 归一化到 0-1
        magnitude = min(1.0, magnitude)

        return magnitude > self.config.structure_break_ratio, magnitude

    # ------------------------------------------------------------------
    # 偏离度计算
    # ------------------------------------------------------------------

    def compute_deviation_score(self, context: MatchContext) -> float:
        """
        给定比赛特征，计算偏离度分数(0-100)。

        高偏离度意味着这场比赛有异常信号（因果引擎应该承担更多权重）。
        判断维度：
        - 近期 vs 历史进球率差异
        - 伤病/换帅信号
        - 淘汰赛压力
        - 温度/裁判等异常因素的存在

        Args:
            context: 比赛上下文

        Returns:
            偏离度分数 (0=正常, 100=极度异常)
        """
        deviations: List[float] = []

        # 1. 进攻效率偏离（近期 vs 历史）
        if context.home_recent and context.home.matches > 0:
            recent_gf = sum(r.goals_for for r in context.home_recent) / max(len(context.home_recent), 1)
            hist_gf = context.home.avg_scored
            if hist_gf > 0:
                dev = abs(recent_gf - hist_gf) / hist_gf
                deviations.append(min(dev * 20, 100))

        if context.away_recent and context.away.matches > 0:
            recent_gf = sum(r.goals_for for r in context.away_recent) / max(len(context.away_recent), 1)
            hist_gf = context.away.avg_scored
            if hist_gf > 0:
                dev = abs(recent_gf - hist_gf) / hist_gf
                deviations.append(min(dev * 20, 100))

        # 2. 伤病/换帅信号
        if context.key_injury_home:
            deviations.append(40)
        if context.key_injury_away:
            deviations.append(40)
        if context.manager_change_home:
            deviations.append(30)
        if context.manager_change_away:
            deviations.append(30)

        # 3. 淘汰赛压力（比赛性质变化）
        if context.is_knockout:
            deviations.append(20)

        # 4. 极端温度
        if context.temperature_celsius is not None:
            if context.temperature_celsius > 30:
                deviations.append(min((context.temperature_celsius - 30) * 3, 40))
            elif context.temperature_celsius < 5:
                deviations.append(min((5 - context.temperature_celsius) * 3, 30))

        # 5. 休息天数差
        if abs(context.rest_days_diff) >= 3:
            deviations.append(15)

        if not deviations:
            return 0.0

        # 取偏差的加权平均 + 最高单项惩罚
        max_dev = max(deviations)
        avg_dev = sum(deviations) / len(deviations)
        combined = avg_dev * 0.6 + max_dev * 0.4
        return min(combined, 100.0)

    # ------------------------------------------------------------------
    # DAG 覆盖度
    # ------------------------------------------------------------------

    def compute_dag_coverage(self, context: MatchContext) -> float:
        """
        计算因果 DAG 覆盖度(0-1)。
        有多少个因果因子在当前 context 中有可用数据。

        Returns:
            覆盖度分数，1.0 = 全量特征可用
        """
        # DAG 中可检查的特征
        feature_checks = [
            context.home_attack_strength > 0,
            context.home_defense_strength > 0,
            context.away_attack_strength > 0,
            context.away_defense_strength > 0,
            len(context.home_recent) > 0,
            len(context.away_recent) > 0,
            context.head_to_head.matches > 1,
            context.home.matches >= 5,
            context.away.matches >= 5,
        ]

        # 可选特征
        if context.temperature_celsius is not None:
            feature_checks.append(True)
        if context.rest_days_diff != 0:
            feature_checks.append(True)
        if context.key_injury_home or context.key_injury_away:
            feature_checks.append(True)

        if not feature_checks:
            return 0.0

        return sum(1 for c in feature_checks if c) / max(len(feature_checks), 6)

    # ------------------------------------------------------------------
    # 主预测接口
    # ------------------------------------------------------------------

    
    @staticmethod
    def _compute_probs_from_rates(
        exp_home: float, exp_away: float,
        home_name: str, away_name: str,
        deviation_score: float
    ) -> Tuple[float, float, float, float]:
        """Compute win/draw/away probabilities + confidence from expected goals.
        Uses MD5-derived seed for cross-process reproducibility."""
        seed_str = home_name + away_name
        seed_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        np.random.seed(seed_val)
        n_sim = 10000
        sim_h = np.random.poisson(max(exp_home, 0.1), n_sim)
        sim_a = np.random.poisson(max(exp_away, 0.1), n_sim)
        home_win = float(np.mean(sim_h > sim_a))
        draw = float(np.mean(sim_h == sim_a))
        away_win = float(np.mean(sim_h < sim_a))
        base_c = 0.65
        confidence = base_c * (1.0 - deviation_score / 200.0)
        confidence = max(0.2, min(0.95, confidence))
        return home_win, draw, away_win, confidence

    def predict(self, context: MatchContext) -> CausalPrediction:
        """
        因果引擎预测接口。

        1. 计算偏离度
        2. 检测结构突变
        3. 估计 ATE（首次调用后缓存）
        4. 应用因果调整乘数
        5. 返回调整后的预期进球和概率

        Args:
            context: 比赛上下文

        Returns:
            CausalPrediction: 完整因果预测
        """
        # Step 1: 偏离度
        deviation_score = self.compute_deviation_score(context)

        # Step 2: 结构突变
        has_break, break_magnitude = self.detect_structure_break(context)

        # Step 3: ATE 估计
        effects = self.estimate_ates()

        # Step 4: 计算因果调整（基于可用特征）
        interventions: Dict[str, Any] = {}
        if context.key_injury_home:
            interventions["home_key_injury"] = True
        if context.key_injury_away:
            interventions["away_key_injury"] = True
        if context.manager_change_home:
            interventions["home_manager_change"] = True
        if context.manager_change_away:
            interventions["away_manager_change"] = True
        if context.is_knockout:
            interventions["is_knockout"] = True
        if context.temperature_celsius is not None:
            interventions["temperature"] = context.temperature_celsius

        # 使用泊松引擎的预期进球作为基线（先算出来）
        # 在这里我们直接用 context 中的攻防强度推
        home_attack_str = context.home_attack_strength or 1.0
        home_defense_str = context.home_defense_strength or 1.0
        away_attack_str = context.away_attack_strength or 1.0
        away_defense_str = context.away_defense_strength or 1.0
        league_avg = context.league_avg_goals

        base_home = league_avg * home_attack_str * away_defense_str
        base_away = league_avg * away_attack_str * home_defense_str

        # 应用因果调整
        adj_result = self.do_intervention(context, interventions)
        home_adj = adj_result["home_adjustment"]
        away_adj = adj_result["away_adjustment"]

        exp_home = base_home * home_adj
        exp_away = base_away * away_adj

        home_win, draw, away_win, confidence = self._compute_probs_from_rates(
            exp_home, exp_away, context.home.name, context.away.name, deviation_score
        )


        # 构建 DAG 覆盖度
        dag_coverage = self.compute_dag_coverage(context)

        signal = CausalSignal(
            deviation_score=_round2(deviation_score),
            structure_break=has_break,
            break_magnitude=_round2(break_magnitude),
            adjustment_factor_home=_round2(home_adj),
            adjustment_factor_away=_round2(away_adj),
            factor_effects=effects,
            dag_coverage=_round2(dag_coverage),
        )

        return CausalPrediction(
            expected_home_goals=_round2(exp_home),
            expected_away_goals=_round2(exp_away),
            expected_total_goals=_round2(exp_home + exp_away),
            home_win_prob=_round2(home_win),
            draw_prob=_round2(draw),
            away_win_prob=_round2(away_win),
            confidence=_round2(confidence),
            signal=signal,
        )


# =============================================================================
# 简单 DAG 实现（causal_engine 不可用时的降级）
# =============================================================================

class _SimpleCausalDAG:
    """
    因果有向无环图的精简实现。
    当 causal_engine.py 不可用时作为降级方案。

    提供与 CausalDAG 兼容的最小接口。
    """

    def __init__(self) -> None:
        self.parents: Dict[str, Set[str]] = {}
        self.children: Dict[str, Set[str]] = {}
        self.exogenous_vars: Set[str] = set()

    # ---- DAG structure constants ----
    DAG_EXOGENOUS: ClassVar[List[str]] = [
        "home_attack_talent", "away_attack_talent",
        "home_defense_talent", "away_defense_talent",
        "temperature", "referee_strictness",
        "home_key_injury", "away_key_injury",
        "home_manager_change", "away_manager_change",
        "rest_days_diff", "is_knockout", "h2h_history",
    ]
    DAG_ENDOGENOUS: ClassVar[List[str]] = [
        "home_attack_strength", "away_attack_strength",
        "home_defense_strength", "away_defense_strength",
        "home_recent_form", "away_recent_form", "h2h_advantage",
    ]
    DAG_EDGES: ClassVar[List[Tuple[str, str]]] = [
        ("home_attack_talent", "home_attack_strength"),
        ("away_attack_talent", "away_attack_strength"),
        ("home_defense_talent", "home_defense_strength"),
        ("away_defense_talent", "away_defense_strength"),
        ("home_key_injury", "home_attack_strength"),
        ("home_key_injury", "home_defense_strength"),
        ("away_key_injury", "away_attack_strength"),
        ("away_key_injury", "away_defense_strength"),
        ("home_manager_change", "home_defense_strength"),
        ("away_manager_change", "away_defense_strength"),
        ("h2h_history", "h2h_advantage"),
        ("home_recent_form", "home_attack_strength"),
        ("home_recent_form", "home_defense_strength"),
        ("away_recent_form", "away_attack_strength"),
        ("away_recent_form", "away_defense_strength"),
        ("temperature", "expected_home_goals"),
        ("temperature", "expected_away_goals"),
        ("referee_strictness", "expected_home_goals"),
        ("referee_strictness", "expected_away_goals"),
        ("rest_days_diff", "expected_home_goals"),
        ("is_knockout", "expected_home_goals"),
        ("is_knockout", "expected_away_goals"),
        ("home_attack_strength", "expected_home_goals"),
        ("away_defense_strength", "expected_home_goals"),
        ("away_attack_strength", "expected_away_goals"),
        ("home_defense_strength", "expected_away_goals"),
        ("h2h_advantage", "expected_home_goals"),
        ("h2h_advantage", "expected_away_goals"),
        ("expected_home_goals", "home_score"),
        ("expected_away_goals", "away_score"),
    ]


    def add_edge(self, parent: str, child: str) -> None:
        if child not in self.parents:
            self.parents[child] = set()
        if parent not in self.children:
            self.children[parent] = set()
        if parent not in self.parents:
            self.parents[parent] = set()
        if child not in self.children:
            self.children[child] = set()
        self.parents[child].add(parent)
        self.children[parent].add(child)

    def add_node(self, node: str, is_exogenous: bool = False) -> None:
        if node not in self.parents:
            self.parents[node] = set()
        if node not in self.children:
            self.children[node] = set()
        if is_exogenous:
            self.exogenous_vars.add(node)

    def get_parents(self, node: str) -> Set[str]:
        return self.parents.get(node, set())

    def get_children(self, node: str) -> Set[str]:
        return self.children.get(node, set())


# =============================================================================
# 内部工具
# =============================================================================

def _round2(x: float) -> float:
    return round(x * 100.0) / 100.0
