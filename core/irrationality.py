"""
BPD非理性因子 — 基于玻尔兹曼策略分布的市场非理性检测

使用 Boltzmann Policy Distribution 的核心逻辑来检测市场赔率中的
系统性非理性行为（过度反应、羊群效应、主场溢价等）。

与原始 BPD 实现 (bpd_implementation.py) 的关系：
- 原始 BPD 建模人类行为的系统性次优性（温度参数控制随机性）
- 此模块将其应用于足球赔率市场：模型概率 = "理性"基准，市场赔率 = "人类"行为
- 温度越低，对偏差越敏感（因为预期理性程度高）
"""

import json
import logging

import numpy as np
from typing import Any, Dict, List, Optional

from .data_types import IrrationalitySignal, MatchContext, MatchPrediction

logger = logging.getLogger(__name__)


class BoltzmannPolicyDistribution:
    """
    玻尔兹曼策略分布（精简版 — 仅用于非理性检测）

    核心逻辑：
    1. 将"市场赔率隐含概率"视为人类在状态空间中的"动作选择"
    2. 将"模型预测概率"视为理性基准
    3. 使用玻尔兹曼能量函数量化偏差
    """

    def __init__(self, temperature: float = 0.05, seed: int = 42):
        """
        Args:
            temperature: 温度参数 β，越低表示预期越理性（对偏差更敏感）
            seed: 随机种子
        """
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

    def compute_energy_divergence(
        self,
        rational_probs: np.ndarray,
        observed_probs: np.ndarray,
    ) -> float:
        """
        计算理性概率与观测概率之间的玻尔兹曼能量散度。

        能量 E = Σ (rational_p(a) - observed_p(a))² 的玻尔兹曼加权版本。
        温度越低，同等偏差的"能量惩罚"越大。

        Args:
            rational_probs: 理性概率分布 [p_home, p_draw, p_away]
            observed_probs: 观测到的概率分布（市场隐含）[p_home, p_draw, p_away]

        Returns:
            float: 能量散度值
        """
        deviation = rational_probs - observed_probs
        energy = np.sum(deviation ** 2) / max(self.temperature, 1e-10)
        return float(energy)

    def compute_entropy(self, probs: np.ndarray) -> float:
        """计算概率分布的熵，衡量不确定性。"""
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def adjust_probabilities(
        self,
        rational_probs: np.ndarray,
        market_implied_probs: np.ndarray,
    ) -> np.ndarray:
        """
        根据BPD模型调整概率。

        调整规则：
        - 如果市场显示非理性（能量高），向理性概率方向拉回
        - 调整幅度由能量和温度共同决定

        Args:
            rational_probs: 理性概率
            market_implied_probs: 市场隐含概率

        Returns:
            np.ndarray: 调整后的概率（已归一化）
        """
        energy = self.compute_energy_divergence(rational_probs, market_implied_probs)

        # 调整系数: 能量越高 -> 越不信任市场 -> 越向理性偏移
        alpha = 1.0 / (1.0 + energy)
        alpha = float(np.clip(alpha, 0.3, 0.9))

        adjusted = alpha * rational_probs + (1.0 - alpha) * market_implied_probs
        adj_sum = np.sum(adjusted)
        adjusted = adjusted / max(adj_sum, 1e-10)
        return adjusted


class IrrationalityDetector:
    """基于BPD模型的市场非理性检测"""

    def __init__(self, temperature: float = 0.05, seed: int = 42):
        """
        Args:
            temperature: BPD温度参数（越低=预期越理性=对偏差越敏感）
            seed: 随机种子
        """
        self.bpd = BoltzmannPolicyDistribution(temperature=temperature, seed=seed)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 公有 API
    # ------------------------------------------------------------------

    def detect_market_irrationality(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, float],
        knox_insight: Optional[Dict[str, Any]] = None,
    ) -> IrrationalitySignal:
        """
        检测市场非理性。

        核心思路：
        1. 将十进制赔率转换为隐含概率（去除抽水）
        2. 模型概率 vs 市场隐含概率 = 理性 vs 人类行为
        3. 用 BPD 计算能量散度 / 熵 / 偏差方向

        Args:
            model_probs: 模型预测概率 {"home": 0.49, "draw": 0.27, "away": 0.24}
            market_odds: 市场赔率 {"home": 2.1, "draw": 3.4, "away": 3.6}

        Returns:
            IrrationalitySignal: 非理性信号
        """
        labels = ["home", "draw", "away"]
        mp = np.array([model_probs.get(k, 0.0) for k in labels])
        market_impl = self._odds_to_implied_prob(market_odds)
        ip = np.array([market_impl.get(k, 0.0) for k in labels])

        # 熵
        entropy = self.bpd.compute_entropy(mp)

        # 玻尔兹曼能量
        bpd_energy = self.bpd.compute_energy_divergence(mp, ip)

        # 非理性分数 (0-100)：使用 sigmoid 型映射
        raw_score = bpd_energy * 5.0
        score = min(100.0, max(0.0, raw_score))

        # 方向判定
        direction = self._determine_direction(mp, ip)

        # 调整概率
        adjusted = self.bpd.adjust_probabilities(mp, ip)
        adjusted_prob = {
            "home": float(adjusted[0]),
            "draw": float(adjusted[1]),
            "away": float(adjusted[2]),
        }

        # Knox 增强的分数调整
        if knox_insight is not None:
            knox_conf = knox_insight.get("confidence", 0)
            if knox_conf > 50:
                knox_dir = knox_insight.get("direction", "neutral")
                if knox_dir != "neutral" and knox_dir == direction:
                    # LLM 确认 BPD 结果
                    if knox_conf > 80:
                        score = min(100.0, score * 1.15)
                    else:
                        score = min(100.0, score * 1.05)
                elif knox_dir != "neutral":
                    # LLM 发现新信号
                    score = min(100.0, score + 8.0 * (knox_conf / 100.0))
                    if knox_conf > 70:
                        direction = knox_dir

        return IrrationalitySignal(
            score=round(score, 2),
            direction=direction,
            adjusted_prob=adjusted_prob,
            market_implied_prob=market_impl,
            entropy=round(entropy, 4),
            bpd_energy=round(bpd_energy, 4),
        )

    def compute_irrationality_score(
        self,
        match_ctx: MatchContext,
        model_pred: MatchPrediction,
        knox_client: Optional[Any] = None,
    ) -> float:
        """
        综合评估非理性水平（用于调整置信度）。

        根据比赛上下文和模型预测评估市场可能存在的非理性程度。
        返回 0-100 的分数用于调整最终预测的置信度。

        Args:
            match_ctx: 比赛上下文
            model_pred: 模型预测
            knox_client: 可选的 KnoxClient 实例（用于 LLM 增强）

        Returns:
            float: 非理性分数 (0-100)，越高越不可信
        """
        market_odds = self._estimate_market_odds(match_ctx)
        model_probs = {
            "home": model_pred.home_win_prob,
            "draw": model_pred.draw_prob,
            "away": model_pred.away_win_prob,
        }

        # Knox 增强：用 LLM 分析市场非理性
        knox_insight: Optional[Dict[str, Any]] = None
        if knox_client is not None:
            try:
                knox_insight = self._analyze_with_knox(
                    knox_client,
                    match_ctx,
                    model_probs,
                    market_odds,
                )
            except Exception as exc:
                logger.warning("Knox irrationality analysis failed: %s", exc)

        signal = self.detect_market_irrationality(
            model_probs, market_odds, knox_insight=knox_insight
        )
        return signal.score

    def _analyze_with_knox(
        self,
        knox_client: Any,
        match_ctx: MatchContext,
        model_probs: Dict[str, float],
        market_odds: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        使用 Knox.chat LLM 辅助分析市场非理性。

        LLM 分析可能捕捉到 BPD 模型无法识别的上下文非理性因素，
        如社交媒体情绪、新闻事件、球员状态等。

        Args:
            knox_client: KnoxClient 实例
            match_ctx: 比赛上下文
            model_probs: 模型预测概率
            market_odds: 市场赔率

        Returns:
            Dict: {"direction": str, "confidence": float, "analysis": str}
        """
        prompt = (
            f"Analyze market irrationality for a World Cup match:\n"
            f"Home: {match_ctx.home.name} (avg scored: {match_ctx.home.avg_scored:.2f}, "
            f"avg conceded: {match_ctx.home.avg_conceded:.2f})\n"
            f"Away: {match_ctx.away.name} (avg scored: {match_ctx.away.avg_scored:.2f}, "
            f"avg conceded: {match_ctx.away.avg_conceded:.2f})\n"
            f"Model probabilities: H={model_probs.get('home', 0):.2%}, "
            f"D={model_probs.get('draw', 0):.2%}, A={model_probs.get('away', 0):.2%}\n"
            f"Market odds: H={market_odds.get('home', 0):.2f}, "
            f"D={market_odds.get('draw', 0):.2f}, A={market_odds.get('away', 0):.2f}\n"
            f"Is knockout: {match_ctx.is_knockout}\n"
            f"\n"
            f"Return a JSON object with keys: direction, confidence (0-100), analysis.\n"
            f"direction: one of overvalued_home, overvalued_away, undervalued_home, "
            f"undervalued_away, neutral"
        )

        response = knox_client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a football betting market analyst. "
                        "Analyze match data and return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        # 尝试解析 JSON
        try:
            # 查找 JSON 块
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0].strip()

            result: Dict[str, Any] = json.loads(json_match)
            return result
        except (json.JSONDecodeError, IndexError):
            logger.debug("Knox response not valid JSON: %s", response[:200])
            return {
                "direction": "neutral",
                "confidence": 0,
                "analysis": response[:200],
            }

    # ------------------------------------------------------------------
    # Knox 增强的检测方法
    # ------------------------------------------------------------------

    def detect_with_knox(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, float],
        knox_client: Any,
        match_ctx: MatchContext,
    ) -> IrrationalitySignal:
        """
        使用 Knox API 增强的非理性检测。

        先运行 BPD 基线检测，再用 LLM 分析补充上下文信号。

        Args:
            model_probs: 模型预测概率
            market_odds: 市场赔率
            knox_client: KnoxClient 实例
            match_ctx: 比赛上下文

        Returns:
            IrrationalitySignal: 增强后的非理性信号
        """
        # 基线 BPD 检测
        signal = self.detect_market_irrationality(model_probs, market_odds)

        # Knox 增强
        try:
            insight = self._analyze_with_knox(
                knox_client, match_ctx, model_probs, market_odds
            )
            knox_confidence = insight.get("confidence", 0)
            if knox_confidence > 50:
                # 根据 LLM 分析调整分数和方向
                llm_direction = insight.get("direction", "neutral")
                if llm_direction != "neutral" and llm_direction == signal.direction:
                    # LLM 确认 BPD 结果 — 提高置信度
                    signal.score = min(100.0, signal.score * 1.1)
                elif llm_direction != "neutral":
                    # LLM 发现 BPD 未捕捉到的信号
                    signal.score = min(100.0, signal.score + 5.0)
                    signal.direction = llm_direction
        except Exception as exc:
            logger.warning("Knox enhancement failed: %s", exc)

        return signal

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _odds_to_implied_prob(self, odds: Dict[str, float]) -> Dict[str, float]:
        """
        将十进制赔率转换为隐含概率（去除市场抽水）。

        Args:
            odds: {"home": 2.1, "draw": 3.4, "away": 3.6}

        Returns:
            dict: 隐含概率（已归一化）
        """
        inv = {k: 1.0 / max(v, 1.01) for k, v in odds.items()}
        total = sum(inv.values())
        if total <= 0:
            return {"home": 0.0, "draw": 0.0, "away": 0.0}
        return {k: v / total for k, v in inv.items()}

    def _determine_direction(
        self,
        model_probs: np.ndarray,
        market_implied: np.ndarray,
    ) -> str:
        """判定偏差方向"""
        home_bias = market_implied[0] - model_probs[0]
        away_bias = market_implied[2] - model_probs[2]

        if abs(home_bias) > abs(away_bias) and abs(home_bias) > 0.02:
            return "overvalued_home" if home_bias > 0 else "undervalued_home"
        elif abs(away_bias) > 0.02:
            return "overvalued_away" if away_bias > 0 else "undervalued_away"
        else:
            return "neutral"

    def _estimate_market_odds(self, ctx: MatchContext) -> Dict[str, float]:
        """
        根据比赛上下文估算市场赔率。

        模拟市场的"合理赔率范围"：
        - 基于球队实力（进攻/防守强度）推算基准概率
        - 加入主场溢价和强队溢价

        Args:
            ctx: 比赛上下文

        Returns:
            dict: 估算的市场赔率（十进制）
        """
        home_attack = ctx.home_attack_strength if ctx.home_attack_strength > 0 else 1.0
        home_defense = ctx.home_defense_strength if ctx.home_defense_strength > 0 else 1.0
        away_attack = ctx.away_attack_strength if ctx.away_attack_strength > 0 else 1.0
        away_defense = ctx.away_defense_strength if ctx.away_defense_strength > 0 else 1.0

        league_avg = ctx.league_avg_goals or 1.2
        lambda_home = league_avg * home_attack * away_defense * 1.08  # 主场优势
        lambda_away = league_avg * away_attack * home_defense

        total_lambda = lambda_home + lambda_away
        if total_lambda < 0.01:
            return {"home": 2.0, "draw": 3.3, "away": 4.0}

        # 泊松近似
        prob_home = lambda_home / (lambda_home + lambda_away + 0.5) * 0.7
        prob_draw = 0.25 + self._rng.uniform(-0.02, 0.02)
        prob_away = max(0.05, 1.0 - prob_home - prob_draw)

        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total

        # 市场偏差模拟
        if home_attack > 1.2:
            prob_home += 0.03
            prob_away -= 0.02
        prob_home += 0.02
        prob_draw -= 0.01
        prob_away -= 0.01

        total = max(prob_home + prob_draw + prob_away, 0.01)  # PTD
        odds_home = round(1.0 / max(prob_home / total, 0.01), 2)
        odds_draw = round(1.0 / max(prob_draw / total, 0.01), 2)
        odds_away = round(1.0 / max(prob_away / total, 0.01), 2)

        return {"home": odds_home, "draw": odds_draw, "away": odds_away}
