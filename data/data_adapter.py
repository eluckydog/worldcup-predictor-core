"""
历史数据适配器 — 从 SQLite 数据库读取数据，提供引擎所需查询接口。

替代 main.py 中的 mock 数据层 _TEAM_BASE_DATA / _get_team_stats / _build_context。
"""

import os
import sqlite3
from typing import Dict, List, Optional, Tuple

from core.data_types import TeamStats, HeadToHead, RecentMatch

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "worldcup.db",
)


# =============================================================================
# 数据库连接管理
# =============================================================================


class DataAdapter:
    """
    数据适配器 — 从 worldcup.db 读取数据并提供查询接口。

    使用 lazy 连接：首次查询时打开数据库连接。
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: SQLite 数据库路径（默认 data/worldcup.db）
        """
        self._db_path = db_path or DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._team_cache: Dict[str, Optional[int]] = {}
        self._global_avg: Optional[float] = None

    # ── 连接管理 ──

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（lazy 初始化）。"""
        if self._conn is None:
            if not os.path.isfile(self._db_path):
                raise FileNotFoundError(
                    f"worldcup.db 不存在: {self._db_path}\n"
                    "请先运行 `python -m data.importer` 导入数据"
                )
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    # ── 内部查询 ──

    def _team_id(self, name: str) -> Optional[int]:
        """
        获取球队内部 ID。

        Args:
            name: 球队名称

        Returns:
            Optional[int]: 球队 ID，未找到返回 None
        """
        if name in self._team_cache:
            return self._team_cache[name]

        conn = self._get_conn()
        cursor = conn.execute("SELECT id FROM teams WHERE name = ?", (name,))
        row = cursor.fetchone()
        tid = row[0] if row else None
        self._team_cache[name] = tid
        return tid

    # ── 查询接口（供引擎使用）──

    def team_stats(self, name: str) -> TeamStats:
        """
        获取球队历史统计数据（所有世界杯比赛汇总）。

        计算每场比赛的进球/失球，返回场均数据。

        Args:
            name: 球队名称

        Returns:
            TeamStats: 包含场均进球、场均失球等统计
        """
        tid = self._team_id(name)
        if tid is None:
            return TeamStats(
                name=name, matches=0, avg_scored=0.0, avg_conceded=0.0,
            )

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT home_score, away_score, "
            "       aet_home, aet_away, penalties_home, penalties_away "
            "FROM matches WHERE home_team_id = ? OR away_team_id = ?",
            (tid, tid),
        )

        matches_played = 0
        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0
        clean_sheets = 0

        for row in cursor:
            matches_played += 1
            is_home = True  # we'll derive from query later for simplicity
            if row[0] is not None:
                gf = row[0]
                ga = row[1]
            else:
                gf = row[2] if row[2] is not None else 0
                ga = row[3] if row[3] is not None else 0

            goals_for += gf
            goals_against += ga

            if gf > ga:
                wins += 1
            elif gf == ga:
                draws += 1
            else:
                losses += 1

            if ga == 0:
                clean_sheets += 1

        # 同时也查询作为客队的比赛
        cursor2 = conn.execute(
            "SELECT home_score, away_score, "
            "       aet_home, aet_away, penalties_home, penalties_away "
            "FROM matches WHERE away_team_id = ?",
            (tid,),
        )
        for row in cursor2:
            # 这里 row 是作为客队，所以 goals_for = away_score, goals_against = home_score
            gf = row[1]  # away_score
            ga = row[0]  # home_score
            goals_for += gf
            goals_against += ga
            if gf > ga:
                wins += 1
            elif gf == ga:
                draws += 1
            else:
                losses += 1
            if ga == 0:
                clean_sheets += 1

        # 注意：上面用独立查询处理了作为主队和客队的情况，实际上每个比赛会被统计两次
        # 修复：只查一次，区分主客场
        matches_played_home = 0
        matches_played_away = 0
        goals_for_home = 0
        goals_against_home = 0
        goals_for_away = 0
        goals_against_away = 0
        clean_sheets_home = 0
        clean_sheets_away = 0

        cursor3 = conn.execute(
            "SELECT home_score, away_score, "
            "       aet_home, aet_away, penalties_home, penalties_away "
            "FROM matches WHERE home_team_id = ?",
            (tid,),
        )
        for row in cursor3:
            matches_played_home += 1
            gf = row[0] if row[0] is not None else (row[2] if row[2] is not None else 0)
            ga = row[1] if row[1] is not None else (row[3] if row[3] is not None else 0)
            goals_for_home += gf
            goals_against_home += ga
            if ga == 0:
                clean_sheets_home += 1

        cursor4 = conn.execute(
            "SELECT home_score, away_score, "
            "       aet_home, aet_away, penalties_home, penalties_away "
            "FROM matches WHERE away_team_id = ?",
            (tid,),
        )
        for row in cursor4:
            matches_played_away += 1
            gf = row[1] if row[1] is not None else (row[3] if row[3] is not None else 0)
            ga = row[0] if row[0] is not None else (row[2] if row[2] is not None else 0)
            goals_for_away += gf
            goals_against_away += ga
            if ga == 0:
                clean_sheets_away += 1

        total_matches = matches_played_home + matches_played_away
        total_gf = goals_for_home + goals_for_away
        total_ga = goals_against_home + goals_against_away
        total_cs = clean_sheets_home + clean_sheets_away

        avg_scored = round(total_gf / max(total_matches, 1), 4)
        avg_conceded = round(total_ga / max(total_matches, 1), 4)

        # 获取最早和最晚参赛年份
        cursor5 = conn.execute(
            "SELECT t.year FROM matches m "
            "JOIN tournaments t ON m.tournament_id = t.id "
            "WHERE m.home_team_id = ? OR m.away_team_id = ? "
            "ORDER BY t.year ASC LIMIT 1",
            (tid, tid),
        )
        first_year_row = cursor5.fetchone()
        first_year = first_year_row[0] if first_year_row else None

        cursor6 = conn.execute(
            "SELECT t.year FROM matches m "
            "JOIN tournaments t ON m.tournament_id = t.id "
            "WHERE m.home_team_id = ? OR m.away_team_id = ? "
            "ORDER BY t.year DESC LIMIT 1",
            (tid, tid),
        )
        last_year_row = cursor6.fetchone()
        last_year = last_year_row[0] if last_year_row else None

        return TeamStats(
            name=name,
            matches=total_matches,
            wins=wins,
            draws=draws,
            losses=losses,
            goals_for=total_gf,
            goals_against=total_ga,
            avg_scored=avg_scored,
            avg_conceded=avg_conceded,
            clean_sheets=total_cs,
            first_year=first_year,
            last_year=last_year,
        )

    def head_to_head(self, team_a: str, team_b: str) -> HeadToHead:
        """
        获取两队历史交锋统计。

        Args:
            team_a: 球队 A
            team_b: 球队 B

        Returns:
            HeadToHead: 包含交锋场次、A/B进球等
        """
        tid_a = self._team_id(team_a)
        tid_b = self._team_id(team_b)
        if tid_a is None or tid_b is None:
            return HeadToHead(matches=0)

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT home_team_id, home_score, away_score, "
            "       aet_home, aet_away "
            "FROM matches "
            "WHERE (home_team_id = ? AND away_team_id = ?) "
            "   OR (home_team_id = ? AND away_team_id = ?)",
            (tid_a, tid_b, tid_b, tid_a),
        )

        matches = 0
        a_wins = 0
        b_wins = 0
        draws = 0
        a_goals = 0
        b_goals = 0

        for row in cursor:
            matches += 1
            home_id = row[0]
            home_score = row[1] if row[1] is not None else 0
            away_score = row[2] if row[2] is not None else 0

            if home_id == tid_a:
                a_goals += home_score
                b_goals += away_score
                if home_score > away_score:
                    a_wins += 1
                elif home_score < away_score:
                    b_wins += 1
                else:
                    draws += 1
            else:
                a_goals += away_score
                b_goals += home_score
                if away_score > home_score:
                    a_wins += 1
                elif away_score < home_score:
                    b_wins += 1
                else:
                    draws += 1

        return HeadToHead(
            matches=matches,
            a_wins=a_wins,
            b_wins=b_wins,
            draws=draws,
            a_goals=a_goals,
            b_goals=b_goals,
        )

    def recent_matches(
        self,
        name: str,
        limit: int = 10,
    ) -> List[RecentMatch]:
        """
        获取球队最近的比赛记录。

        Args:
            name: 球队名称
            limit: 返回数量上限

        Returns:
            List[RecentMatch]: 近期比赛列表（按年份降序）
        """
        tid = self._team_id(name)
        if tid is None:
            return []

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT m.home_team_id, m.home_score, m.away_score, "
            "       t.year, m.stage, "
            "       m.aet_home, m.aet_away, m.penalties_home, m.penalties_away "
            "FROM matches m "
            "JOIN tournaments t ON m.tournament_id = t.id "
            "WHERE m.home_team_id = ? OR m.away_team_id = ? "
            "ORDER BY t.year DESC, m.id DESC "
            "LIMIT ?",
            (tid, tid, limit),
        )

        results: List[RecentMatch] = []
        for row in cursor:
            is_home = row[0] == tid
            gf = row[1] if is_home else row[2]
            ga = row[2] if is_home else row[1]

            # 使用 aet 数据（如果有）
            aet_home = row[5]
            aet_away = row[6]
            if aet_home is not None or aet_away is not None:
                if is_home:
                    gf = aet_home if aet_home is not None else gf
                    ga = aet_away if aet_away is not None else ga
                else:
                    gf = aet_away if aet_away is not None else gf
                    ga = aet_home if aet_home is not None else ga

            # 获取对手名称
            opponent_conn = conn.execute(
                "SELECT name FROM teams WHERE id = ?",
                (row[3] if is_home else row[0],),
            )
            opp_row = opponent_conn.fetchone()
            opponent = opp_row[0] if opp_row else "Unknown"

            results.append(RecentMatch(
                opponent=opponent or "Unknown",
                goals_for=int(gf) if gf is not None else 0,
                goals_against=int(ga) if ga is not None else 0,
                year=row[3],
                stage=row[4] or "",
                is_home=is_home,
            ))

        return results

    def weighted_form(self, name: str) -> Tuple[float, float]:
        """
        获取球队近期加权形态（越近的比赛权重越高）。

        使用指数衰减：weight = 0.9^(years_ago)

        Args:
            name: 球队名称

        Returns:
            Tuple[float, float]: (weighted_avg_scored, weighted_avg_conceded)
        """
        tid = self._team_id(name)
        if tid is None:
            return (0.0, 0.0)

        conn = self._get_conn()

        # 获取最近 40 场比赛
        cursor = conn.execute(
            "SELECT m.home_team_id, m.home_score, m.away_score, t.year, "
            "       m.aet_home, m.aet_away "
            "FROM matches m "
            "JOIN tournaments t ON m.tournament_id = t.id "
            "WHERE m.home_team_id = ? OR m.away_team_id = ? "
            "ORDER BY t.year DESC, m.id DESC "
            "LIMIT 40",
            (tid, tid),
        )

        total_weight = 0.0
        weighted_gf = 0.0
        weighted_ga = 0.0
        base_year = 2026

        for row in cursor:
            is_home = row[0] == tid
            year = row[3]
            years_ago = base_year - year
            weight = 0.9 ** years_ago

            gf = row[1] if is_home else row[2]
            ga = row[2] if is_home else row[1]

            # 使用 aet 数据
            aet_h = row[4]
            aet_a = row[5]
            if aet_h is not None or aet_a is not None:
                gf = aet_h if is_home and aet_h is not None else (
                    aet_a if not is_home and aet_a is not None else gf
                )
                ga = aet_a if is_home and aet_a is not None else (
                    aet_h if not is_home and aet_h is not None else ga
                )

            weighted_gf += float(gf) * weight
            weighted_ga += float(ga) * weight
            total_weight += weight

        if total_weight < 0.001:
            return (0.0, 0.0)

        avg_scored = round(weighted_gf / total_weight, 4)
        avg_conceded = round(weighted_ga / total_weight, 4)
        return (avg_scored, avg_conceded)

    def global_avg_goals(self) -> float:
        """
        获取全部世界杯比赛的平均进球数。

        Returns:
            float: 场均进球
        """
        if self._global_avg is not None:
            return self._global_avg

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT AVG(CAST(home_score AS REAL) + CAST(away_score AS REAL)) "
            "FROM matches "
            "WHERE home_score IS NOT NULL AND away_score IS NOT NULL"
        )
        row = cursor.fetchone()
        avg = row[0] if row[0] is not None else 2.5
        self._global_avg = float(round(avg, 4))
        return self._global_avg

    def has_data(self) -> bool:
        """
        检查数据库是否存在且有数据。

        Returns:
            bool
        """
        if not os.path.isfile(self._db_path):
            return False
        try:
            conn = self._get_conn()
            count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            return count > 0
        except (sqlite3.Error, FileNotFoundError):
            return False

    def get_match_context(
        self,
        home: str,
        away: str,
        league_avg: Optional[float] = None,
    ) -> "MatchContext":
        """
        构建一场比赛的完整 MatchContext。

        从数据库读取两队统计、H2H、近期状态、加权形态。
        这是 _build_context 的数据库版实现。

        Args:
            home: 主队名
            away: 客队名
            league_avg: 联赛场均进球（自动获取如果为None）

        Returns:
            MatchContext: 比赛上下文
        """
        from core.data_types import MatchContext, TeamStats, HeadToHead
        from core.team_resolver import is_host_nation, resolve_team_name

        resolved_home = resolve_team_name(home)
        resolved_away = resolve_team_name(away)

        home_stats = self.team_stats(resolved_home)
        away_stats = self.team_stats(resolved_away)
        h2h = self.head_to_head(resolved_home, resolved_away)

        if league_avg is None:
            league_avg = self.global_avg_goals()

        home_form_attack, home_form_defense = self.weighted_form(resolved_home)
        away_form_attack, away_form_defense = self.weighted_form(resolved_away)

        if home_form_attack <= 0:
            home_form_attack = home_stats.avg_scored
            home_form_defense = home_stats.avg_conceded
        if away_form_attack <= 0:
            away_form_attack = away_stats.avg_scored
            away_form_defense = away_stats.avg_conceded

        ctx = MatchContext(
            home=home_stats,
            away=away_stats,
            head_to_head=h2h,
            league_avg_goals=league_avg,
            home_form_attack=home_form_attack,
            home_form_defense=home_form_defense,
            away_form_attack=away_form_attack,
            away_form_defense=away_form_defense,
            home_attack_strength=home_stats.avg_scored / max(league_avg, 0.01),
            home_defense_strength=home_stats.avg_conceded / max(league_avg, 0.01),
            away_attack_strength=away_stats.avg_scored / max(league_avg, 0.01),
            away_defense_strength=away_stats.avg_conceded / max(league_avg, 0.01),
        )

        if is_host_nation(resolved_home):
            ctx.home_form_attack *= 1.08
        if is_host_nation(resolved_away):
            ctx.away_form_attack *= 1.08

        return ctx
