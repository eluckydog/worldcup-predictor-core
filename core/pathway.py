"""
EPG路径优化 — 基于进化策略梯度的预测路径优化

使用 Evolutionary Policy Gradient 的核心逻辑来搜索最优的预测调整路径。
将"预测调整"视为一个策略优化问题：
- 状态 = 当前预测的概率分布
- 动作 = 调整量（概率偏移）
- 奖励 = 调整后预测对近期比赛的解释力

与原始 EPG 实现 (epg_implementation.py) 的关系：
- 原始 EPG 在连续状态空间中进化策略网络
- 此模块将其简化为：在"预测调整空间"中搜索最优路径
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .data_types import PathwaySignal, PoissonPrediction


class EvolutionaryPolicyGradient:
    """
    进化策略梯度（精简版 — 仅用于路径优化搜索）

    使用进化策略在预测调整空间中搜索最优路径。
    每一代：生成候选调整方案 -> 评估适应度 -> 精英选择 -> 变异 / 交叉
    """

    def __init__(
        self,
        population_size: int = 30,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.15,
        mutation_strength: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            population_size: 种群大小
            elite_ratio: 精英比例
            mutation_rate: 变异率
            mutation_strength: 变异强度
            seed: 随机种子
        """
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self._rng = np.random.default_rng(seed)

    def _init_population(self) -> np.ndarray:
        """
        初始化种群（生成随机调整向量）。

        每个个体是一个三维向量 [Δhome, Δdraw, Δaway]，
        表示对三个结果的概率调整量（总和为 0）。

        Returns:
            np.ndarray: shape=(population_size, 3)
        """
        pop = self._rng.normal(0, 0.05, size=(self.population_size, 3))
        pop = pop - pop.mean(axis=1, keepdims=True)
        return pop

    def _evaluate_fitness(
        self,
        adjustment: np.ndarray,
        base_probs: np.ndarray,
        recent_results: np.ndarray,
    ) -> float:
        """
        评估调整方案的适应度。

        适应度 = 调整后概率对近期比赛结果的平均对数似然。
        无历史数据时，惩罚大幅调整。

        Args:
            adjustment: 调整向量 [Δhome, Δdraw, Δaway]
            base_probs: 基础概率 [p_home, p_draw, p_away]
            recent_results: 近期比赛结果编码 [0=home_win, 1=draw, 2=away_win]

        Returns:
            float: 适应度分数（越高越好）
        """
        adjusted = np.clip(base_probs + adjustment, 0.001, 0.999)
        adjusted = adjusted / max(np.sum(adjusted), 1e-10)

        if len(recent_results) == 0:
            return float(-np.sum(adjustment ** 2))

        log_likelihoods = np.log(adjusted[recent_results.astype(int)])
        return float(np.mean(log_likelihoods))

    def _select_elites(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """选择适应度最高的精英个体。"""
        n_elite = max(1, int(self.population_size * self.elite_ratio))
        elite_idx = np.argsort(fitness)[-n_elite:]
        return population[elite_idx].copy()

    def _reproduce(self, elites: np.ndarray, n_offspring: int) -> np.ndarray:
        """
        从精英种群繁殖后代。
        - 交叉：两个精英均匀混合
        - 变异：高斯噪声
        """
        offspring = []
        n_elite = len(elites)

        for _ in range(n_offspring):
            p1 = elites[self._rng.integers(n_elite)]
            p2 = elites[self._rng.integers(n_elite)]
            child = (p1 + p2) / 2.0

            mask = self._rng.random(3) < self.mutation_rate
            child += mask * self._rng.normal(0, self.mutation_strength, size=3)
            child = child - np.mean(child)
            offspring.append(child)

        return np.array(offspring)

    def search(
        self,
        base_probs: np.ndarray,
        recent_results: np.ndarray,
        max_generations: int = 50,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        搜索最优调整方案。

        Args:
            base_probs: 基础概率 [p_home, p_draw, p_away]
            recent_results: 近期比赛结果编码 (0=home, 1=draw, 2=away)
            max_generations: 最大代数

        Returns:
            Tuple[np.ndarray, List[float]]: (最优调整向量, 适应度历史)
        """
        population = self._init_population()
        fitness_history: List[float] = []
        best_adjustment = np.zeros(3)
        best_fitness = -np.inf

        for _ in range(max_generations):
            fitness = np.array([
                self._evaluate_fitness(ind, base_probs, recent_results)
                for ind in population
            ])

            gen_best_idx = int(np.argmax(fitness))
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_adjustment = population[gen_best_idx].copy()

            fitness_history.append(best_fitness)

            # 早期终止
            if len(fitness_history) >= 10:
                recent = fitness_history[-10:]
                if max(recent) - min(recent) < 0.001:
                    break

            elites = self._select_elites(population, fitness)
            n_offspring = self.population_size - len(elites)

            if n_offspring > 0:
                offspring = self._reproduce(elites, n_offspring)
                population = np.vstack([elites, offspring])
            else:
                population = elites

        return best_adjustment, fitness_history


class PathwayOptimizer:
    """基于EPG的预测路径优化"""

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: 随机种子
        """
        self.epg = EvolutionaryPolicyGradient(
            population_size=30,
            elite_ratio=0.2,
            mutation_rate=0.15,
            mutation_strength=0.1,
            seed=seed,
        )
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 公有 API
    # ------------------------------------------------------------------

    def optimize_pathway(
        self,
        prediction: PoissonPrediction,
        market_odds: Dict[str, float],
        history: List[dict],
    ) -> PathwaySignal:
        """
        路径优化：基于近期比赛结果搜索最优概率调整。

        流程：
        1. 提取基础预测概率为初始状态
        2. 将历史结果编码为适应度评估数据
        3. EPG 在调整空间搜索最优方案
        4. 输出路径序列和最终调整

        Args:
            prediction: 基础预测（泊松引擎输出）
            market_odds: 市场赔率（用于辅助评估）
            history: 近期比赛结果序列

        Returns:
            PathwaySignal: 路径优化信号
        """
        base_probs = np.array([
            prediction.home_win_prob,
            prediction.draw_prob,
            prediction.away_win_prob,
        ])

        recent_results = self._encode_results(history)
        best_adjustment, fitness_history = self.epg.search(
            base_probs, recent_results, max_generations=50
        )

        adjusted = np.clip(base_probs + best_adjustment, 0.001, 0.999)
        adjusted = adjusted / max(np.sum(adjusted), 1e-10)

        convergence = self._compute_convergence(fitness_history)
        path_complexity = self._compute_path_complexity(best_adjustment)
        recommendations = self._generate_recommendations(best_adjustment, convergence)
        path = self._build_path_sequence(base_probs, best_adjustment)

        return PathwaySignal(
            path=path,
            final_adjustment={
                "delta_home": round(float(best_adjustment[0]), 4),
                "delta_draw": round(float(best_adjustment[1]), 4),
                "delta_away": round(float(best_adjustment[2]), 4),
            },
            convergence=round(convergence, 2),
            path_complexity=path_complexity,
            recommendations=recommendations,
        )

    def temperature_adjustment(
        self,
        base_pred: PoissonPrediction,
        recent_matches: List[dict],
        host_advantage: float,
    ) -> dict:
        """
        调整预测以反映环境因素（天气 / 时差 / 主客场温度等）。

        使用启发式方法而非完全 EPG 搜索：
        根据主场优势和近期表现趋势微调概率。

        Args:
            base_pred: 基础预测
            recent_matches: 近期比赛结果
            host_advantage: 主场优势乘数 (>1.0 = 主队更有利)

        Returns:
            dict: 调整结果
        """
        base_probs = np.array([
            base_pred.home_win_prob,
            base_pred.draw_prob,
            base_pred.away_win_prob,
        ])

        delta = self._heuristic_adjustment(base_probs, recent_matches, host_advantage)

        adjusted = np.clip(base_probs + delta, 0.001, 0.999)
        adjusted = adjusted / max(np.sum(adjusted), 1e-10)

        return {
            "adjusted_home_win": round(float(adjusted[0]), 4),
            "adjusted_draw": round(float(adjusted[1]), 4),
            "adjusted_away_win": round(float(adjusted[2]), 4),
            "delta_home": round(float(delta[0]), 4),
            "delta_draw": round(float(delta[1]), 4),
            "delta_away": round(float(delta[2]), 4),
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _heuristic_adjustment(
        self,
        base_probs: np.ndarray,
        recent_matches: List[dict],
        host_advantage: float,
    ) -> np.ndarray:
        """计算启发式调整向量"""
        delta = np.zeros(3)

        # 主场优势
        home_bonus = (host_advantage - 1.0) * 0.1
        delta[0] += home_bonus
        delta[1] -= home_bonus * 0.5
        delta[2] -= home_bonus * 0.5

        # 近期表现趋势
        if recent_matches:
            results = self._encode_results(recent_matches)
            if len(results) > 0:
                n = max(len(results), 1)  # PTD
                home_wins = float(np.sum(results == 0))
                away_wins = float(np.sum(results == 2))
                delta[0] += (home_wins - n * base_probs[0]) / n * 0.05 * 0.3
                delta[2] += (away_wins - n * base_probs[2]) / n * 0.05 * 0.3

        delta = delta - np.mean(delta)
        return delta

    def _encode_results(self, history: List[dict]) -> np.ndarray:
        """将历史比赛结果编码为适应度评估输入。"""
        if not history:
            return np.array([], dtype=np.int64)

        codes = []
        for match in history:
            if isinstance(match, dict):
                hg = match.get("home_goals", 0)
                ag = match.get("away_goals", 0)
                if hg > ag:
                    codes.append(0)
                elif hg == ag:
                    codes.append(1)
                else:
                    codes.append(2)
        return np.array(codes, dtype=np.int64)

    @staticmethod
    def _compute_convergence(fitness_history: List[float]) -> float:
        """计算收敛度 (0-100)。"""
        if len(fitness_history) < 5:
            return 30.0
        recent = fitness_history[-5:]
        spread = max(recent) - min(recent)

        if spread < 0.001:
            return 95.0
        elif spread < 0.01:
            return 80.0
        elif spread < 0.05:
            return 60.0
        elif spread < 0.1:
            return 40.0
        else:
            return 20.0

    @staticmethod
    def _compute_path_complexity(adjustment: np.ndarray) -> int:
        """计算路径复杂度 (1-5)。"""
        abs_adj = np.abs(adjustment)
        n_nonzero = int(np.sum(abs_adj > 0.01))
        max_adj = float(np.max(abs_adj))

        complexity = n_nonzero
        if max_adj > 0.05:
            complexity += 1
        if max_adj > 0.10:
            complexity += 1
        return min(5, max(1, complexity))

    @staticmethod
    def _build_path_sequence(
        base_probs: np.ndarray,
        best_adjustment: np.ndarray,
    ) -> List[dict]:
        """构建3步优化路径序列。"""
        steps = 3
        path = []
        for i in range(1, steps + 1):
            fraction = i / steps
            step_adj = best_adjustment * fraction
            step_probs = np.clip(base_probs + step_adj, 0.001, 0.999)
            step_probs = step_probs / np.sum(step_probs)
            path.append({
                "step": i,
                "home_win_prob": round(float(step_probs[0]), 4),
                "draw_prob": round(float(step_probs[1]), 4),
                "away_win_prob": round(float(step_probs[2]), 4),
            })
        return path

    @staticmethod
    def _generate_recommendations(
        adjustment: np.ndarray,
        convergence: float,
    ) -> List[str]:
        """根据调整结果生成文字建议。"""
        recs: List[str] = []

        if convergence < 40:
            recs.append("市场信号混乱，建议降低置信度")

        if abs(adjustment[0]) > 0.05:
            d = "上调" if adjustment[0] > 0 else "下调"
            recs.append(f"建议{d}主队概率 {abs(adjustment[0]) * 100:.1f}%")

        if abs(adjustment[2]) > 0.05:
            d = "上调" if adjustment[2] > 0 else "下调"
            recs.append(f"建议{d}客队概率 {abs(adjustment[2]) * 100:.1f}%")

        if not recs:
            recs.append("当前预测已接近最优，无需大调整")

        return recs
