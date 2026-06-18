"""
世界杯历史数据导入器 — 从 open-football 格式的 cup.txt 解析并导入 SQLite。

扫描 worldcup-data 目录，解析每届世界杯的 cup.txt / cup_finals.txt 文件，
导入到 worldcup-predictor-core/data/worldcup.db。

open-football 格式参考：
  = World Cup 2022       # in Qatar, November 20 - December 18
  Group A  | Qatar Ecuador Senegal Netherlands
  ▪ Group A
  Mon Nov 20
    19:00   Qatar   0-2 (0-2)   Ecuador    @ Stadium
  ▪ Round of 16
  Sat Dec 3
    18:00   Netherlands 3-1 (2-0) USA
  ▪ Quarter-finals
  加时：0-0 a.e.t., 3-4 pen.
"""

import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

# ── 解析常量 ──

TOURNAMENT_HEADER_RE = re.compile(
    r"^=\s*World\s+Cup\s+(\d{4}).*?(?:#\s+in\s+(.+?)(?:,\s|$)|#\s+Finals)"
)

STAGE_RE = re.compile(r"^▪\s*(.+)$")
GROUP_DEF_RE = re.compile(r"^Group\s+([A-Za-z0-9]+)\s*\|(.+)$")

# 匹配带时间的比赛行，如 "  19:00   Team   3-1 (2-0)   Opponent"
# 或 "Mon Nov 20" 日期行
TIME_LINE_RE = re.compile(r"^\s{2,}(\d{1,2}:\d{2})")
DATE_LINE_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)

# 匹配 "Team   Score   Opponent" 格式
# Score = numbers, possibly with a.e.t., pen.
SCORE_RE = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(\d{1,2}:\d{2}\s+)?"
    r"([A-Za-zÀ-ÿ0-9'’\-\.\s]+?)\s+"
    r"(\d+[–-]\d+(?:\s+a\.e\.t\.?(?:\s*\([^)]*\))?(?:\s*,\s*\d+[–-]\d+\s+pen\.?)?)?"
    r"(?:\s*\([^)]*\))?\s*"
    r"(?:a\.e\.t\.?(?:\s*\([^)]*\))?(?:\s*,\s*\d+[–-]\d+\s+pen\.?)?)?)\s+"
    r"([A-Za-zÀ-ÿ0-9'’\-\.\s]+?)(?:\s+@\s+.+)?$"
)

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)

DB_PATH = os.path.join(DATA_ROOT, "worldcup.db")

# ── 已知主办国映射（用于年份 → 主办国解析，当 header 解析失败时） ──

FALLBACK_HOSTS: Dict[int, str] = {
    1930: "Uruguay",
    1934: "Italy",
    1938: "France",
    1950: "Brazil",
    1954: "Switzerland",
    1958: "Sweden",
    1962: "Chile",
    1966: "England",
    1970: "Mexico",
    1974: "West Germany",
    1978: "Argentina",
    1982: "Spain",
    1986: "Mexico",
    1990: "Italy",
    1994: "United States",
    1998: "France",
    2002: "South Korea & Japan",
    2006: "Germany",
    2010: "South Africa",
    2014: "Brazil",
    2018: "Russia",
    2022: "Qatar",
    2026: "Canada, USA, Mexico",
}


# =============================================================================
# 比分解析
# =============================================================================


def parse_score(score_text: str) -> Tuple[int, int, Optional[int], Optional[int],
                                          Optional[int], Optional[int]]:
    """
    解析比分字符串，提取全场比赛、加时赛和点球比分。

    Args:
        score_text: 如 "3-1 (2-0)" 或 "1-1 a.e.t (1-1, 1-0), 1-3 pen."
                   或 "1-1 a.e.t. (0-0, 0-0), 4-2 pen." 或 "0-0 a.e.t., 3-4 pen."

    Returns:
        Tuple[home_ft, away_ft, home_aet, away_aet, home_pen, away_pen]
        - home_ft/away_ft: 常规时间/全场比分
        - home_aet/away_aet: 加时赛比分（None 表示无加时）
        - home_pen/away_pen: 点球比分（None 表示无点球）
        所有值均为整数。
    """
    # 默认值
    home_ft = 0
    away_ft = 0
    home_aet: Optional[int] = None
    away_aet: Optional[int] = None
    home_pen: Optional[int] = None
    away_pen: Optional[int] = None

    text = score_text.strip()

    # 1. 提取点球比分
    pen_match = re.search(r"(\d+)[–\-](\d+)\s+pen\.?", text)
    if pen_match:
        home_pen = int(pen_match.group(1))
        away_pen = int(pen_match.group(2))
        # 移除点球部分
        text = re.sub(r",\s*\d+[–\-]\d+\s+pen\.?", "", text).strip()
        text = re.sub(r"\s*\d+[–\-]\d+\s+pen\.?", "", text).strip()

    # 2. 检查是否有加时赛
    has_aet = "a.e.t" in text or "aet" in text

    if has_aet:
        # 先移除 a.e.t. 标记
        text_clean = re.sub(r"\s*a\.e\.t\.?\s*", " ", text).strip()
        text_clean = re.sub(r"\s*aet\s*", " ", text_clean).strip()

        # 提取括号内的加时比分，如 "1-1, 1-0"
        paren_match = re.search(r"\(([^)]*)\)", text_clean)
        if paren_match:
            inside = paren_match.group(1)
            # 可能有逗号分隔的两个比分（加时和半场），或一个比分
            parts = [p.strip() for p in inside.split(",")]
            for p in parts:
                ft_score_match = re.match(r"(\d+)[–\-](\d+)", p)
                if ft_score_match:
                    home_aet = int(ft_score_match.group(1))
                    away_aet = int(ft_score_match.group(2))
                    break
            # 提取全场比分（括号前的比分）
            score_before_paren = re.search(
                r"(\d+)[–\-](\d+)\s*\([^)]*\)", text
            )
            if score_before_paren:
                home_ft = int(score_before_paren.group(1))
                away_ft = int(score_before_paren.group(2))
            else:
                # 从第一个数字找起
                all_scores = re.findall(r"(\d+)[–\-](\d+)", text)
                if len(all_scores) >= 2:
                    home_ft, away_ft = int(all_scores[0][0]), int(all_scores[0][1])
                elif len(all_scores) == 1:
                    home_ft, away_ft = int(all_scores[0][0]), int(all_scores[0][1])
        else:
            # a.e.t. 但没有括号里的加时比分，如 "3-2 a.e.t."
            ft_score_match = re.search(r"(\d+)[–\-](\d+)", text_clean)
            if ft_score_match:
                home_ft = int(ft_score_match.group(1))
                away_ft = int(ft_score_match.group(2))
    else:
        # 常规比分：可能有半场比分如 "3-1 (2-0)"
        paren_match = re.search(r"\(([^)]*)\)", text)
        if paren_match:
            # 有半场比分，全场比分在括号前
            ft_score_match = re.search(r"(\d+)[–\-](\d+)\s*\(", text)
            if ft_score_match:
                home_ft = int(ft_score_match.group(1))
                away_ft = int(ft_score_match.group(2))
        else:
            # 简单比分，如 "3-1"
            ft_score_match = re.search(r"(\d+)[–\-](\d+)", text)
            if ft_score_match:
                home_ft = int(ft_score_match.group(1))
                away_ft = int(ft_score_match.group(2))

    return home_ft, away_ft, home_aet, away_aet, home_pen, away_pen


# =============================================================================
# 文件扫描
# =============================================================================


def find_data_files(data_dir: str) -> List[str]:
    """
    扫描 worldcup-data 目录，返回所有 cup.txt / cup_finals.txt 文件路径。

    Args:
        data_dir: worldcup-data 目录路径

    Returns:
        List[str]: 需导入的文件路径列表
    """
    files: List[str] = []
    for entry in sorted(os.listdir(data_dir)):
        entry_path = os.path.join(data_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        # 跳过 non-tournament 目录
        if entry in ("min", "more", "planetworldcup", "rsssf"):
            continue
        # 主数据文件 cup.txt（小组赛）
        cup_path = os.path.join(entry_path, "cup.txt")
        if os.path.isfile(cup_path):
            files.append(cup_path)
        # 淘汰赛数据 cup_finals.txt
        cup_finals_path = os.path.join(entry_path, "cup_finals.txt")
        if os.path.isfile(cup_finals_path):
            files.append(cup_finals_path)
        # 附加资格赛数据
        quali_path = os.path.join(entry_path, "quali_playoffs.txt")
        if os.path.isfile(quali_path):
            files.append(quali_path)
    return sorted(files)


# =============================================================================
# 单文件解析器
# =============================================================================


def parse_tournament_file(file_path: str) -> Optional[Dict]:
    """
    解析一个 cup.txt 文件。

    Args:
        file_path: 文件路径

    Returns:
        Optional[Dict]: {
            "year": int,
            "host": str,
            "dates": str,
            "groups": Dict[str, List[str]],  # group_name -> [team_names]
            "stages": List[Dict],  # [{stage, group, matches: [{home, away, home_score, ...}]}]
        }
        解析失败返回 None。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError) as e:
        print(f"  [WARN] 无法读取 {file_path}: {e}")
        return None

    result: Dict = {
        "year": 0,
        "host": "",
        "dates": "",
        "groups": {},
        "stages": [],
    }

    current_stage: str = ""
    current_group: str = ""
    current_date: str = ""
    current_matches: List[Dict] = []
    header_found = False

    for raw_line in lines:
        line = raw_line.rstrip("\n").rstrip("\r")

        # 跳过空行和注释
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # 1) 解析 tournament 头
        header_match = TOURNAMENT_HEADER_RE.match(stripped)
        if header_match:
            result["year"] = int(header_match.group(1))
            host_str = (header_match.group(2) or "").strip()
            # 提取主办国（日期前的部分）
            date_match = re.search(r"([A-Za-z\s,]+),?\s+(?:January|February|March|April|May|June|"
                                   r"July|August|September|October|November|December)", host_str)
            if date_match:
                result["host"] = date_match.group(1).strip().rstrip(",")
            else:
                result["host"] = host_str
            result["dates"] = host_str
            header_found = True
            continue

        if not header_found:
            continue

        # 2) 分组定义
        group_def_match = GROUP_DEF_RE.match(stripped)
        if group_def_match:
            group_name = group_def_match.group(1)
            teams_raw = group_def_match.group(2)
            # 用多个空格或制表符分割队名
            teams = [t.strip() for t in re.split(r"\s{2,}|\t", teams_raw) if t.strip()]
            result["groups"][group_name] = teams
            continue

        # 3) 阶段标记（▪ ...）
        stage_match = STAGE_RE.match(stripped)
        if stage_match:
            # 保存上一阶段的比赛
            if current_matches:
                result["stages"].append({
                    "stage": current_stage,
                    "group": current_group,
                    "matches": current_matches,
                    "date": current_date,
                })
                current_matches = []

            stage_text = stage_match.group(1).strip()

            # 检查是否包含组名，如 "Group A"、"Matchday 1"
            # 或者是否是淘汰赛轮次 "Round of 16", "Quarter-finals" 等
            # 如果是包含 "Matchday" 的，这实际上是组内的比赛日期标记
            # 对于淘汰赛，stage 可能是纯名称
            current_stage = stage_text

            # 检查是否包含组名信息
            group_in_stage_match = re.match(r"Group\s+([A-Za-z0-9]+)", stage_text)
            if group_in_stage_match:
                current_group = group_in_stage_match.group(1)
            # 对于像 "Round of 16" 这样的纯阶段名，保留组为空
            continue

        # 4) 日期行 / 行内日期：如 "Mon Nov 20"、"July 13"、"2 June"
        # 早期格式无时间列，日期和比分在同一行："10 June    Brazil   2-1   Scotland"
        # 需要区分纯日期行（跳过）和带比分的日期行（提取日期后继续匹配）
        date_found = False
        date_text = ""
        if not re.match(r"^\s{2,}\d{1,2}:\d{2}", line):
            date_candidates = re.findall(
                r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}|"
                r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|"
                r"Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2})",
                stripped,
                re.IGNORECASE,
            )
            if date_candidates:
                date_text = date_candidates[0]
                date_found = True
        if date_found:
            # 检查是否有比分数字（"X-Y" 或 "X-Y (a.e.t.)"）——有比分说明是比赛行
            has_score = bool(re.search(r"\d+[–\-]\d+", stripped))
            if has_score:
                current_date = date_text
                # 从 stripped 中去掉日期前缀（如 "27 June"、"7 July"），
                # 否则后续正则会把日期当成队名匹配
                stripped = re.sub(
                    r"^(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?"
                    r"(?:\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)|"
                    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2})"
                    r"(?:\s+\d{1,2}:\d{2})?\s+", "", stripped
                ).strip()
                # 不 continue，继续到匹配行处理
            else:
                current_date = date_text
                continue
        # 5) 之后部分不变

        # 5) 比赛行
        # 格式1（带时间）："  19:00   Qatar   0-2 (0-2)   Ecuador    @ Stadium"
        # 格式2（无时间，早期比赛）："  France     4-1 (3-0)  Mexico"
        time_match = re.match(r"^\s{2,}(\d{1,2}:\d{2})\s+", line)
        start_pos = line.find(time_match.group(1)) if time_match else 0

        # 移除时间部分（如果有）—— 包括 UTC+/- 偏移
        if time_match:
            raw_after = line[time_match.end():]
            # 去除 UTC+/− 偏移
            raw_after = re.sub(r"^\s*UTC[+-]\d+(?::\d+)?\s*", "", raw_after)
            line_after_time = raw_after
        else:
            line_after_time = stripped

        # 尝试匹配比分模式
        # 格式：TeamA Score TeamB [@ Stadium]
        # 注意队名可能包含连字符、撇号等
        score_pattern = re.compile(
            r"^\s+"
            r"(?:\(\d+\)\s+)?"  # 可选的比赛编号如 (73)
            r"(\d{1,2}:\d{2}(?:[–\-]\d+)?\s+)?"  # 可选的UTC时间
            r"([A-Za-zÀ-ÿ0-9'’\-. ]+?)\s{2,}"  # home team (至少2空格)
            r"(\d+[–\-]\d+(?:\s+a\.e\.t\.?(?:\s*\([^)]*\))?(?:\s*,\s*\d+[–\-]\d+\s+pen\.?)?"
            r"(?:\s*\([^)]*\))?)?)\s{2,}"  # score
            r"([A-Za-zÀ-ÿ0-9'’\-. ]+?)"  # away team
            r"(?:\s+@\s+.+)?$"  # optional @ stadium
        )

        match = score_pattern.match(stripped)
        if not match:
            # 回退 A：按2+空格分割，找独立比分部分
            parts = re.split(r"\s{2,}", stripped)
            score_idx = -1
            for idx, p in enumerate(parts):
                if re.match(r"^\d+[–\-]\d+", p.strip()):
                    score_idx = idx
                    break
            if score_idx > 0 and score_idx < len(parts) - 1:
                home_raw = " ".join(p.strip() for p in parts[:score_idx])
                home_clean = re.sub(
                    r"^(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?"
                    r"(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|"
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2})"
                    r"(?:\s+\d{1,2}:\d{2})?\s+", "", home_raw
                )
                home_clean = re.sub(r"^\d{1,2}:\d{2}(?:\s+UTC[+-]\d+)?\s+", "", home_clean)
                if not home_clean:
                    home_clean = home_raw
                score_part = parts[score_idx].strip()
                # 从比分后面的部分中找到客队和比分注释
                extras = []
                away_team = None
                for ep in parts[score_idx+1:]:
                    e = ep.strip()
                    if not e or "@" in e:
                        continue  # 跳过空行和体育场
                    if "a.e.t" in e or "pen" in e or re.match(r"^\(\d+[–\-]\d+", e):
                        extras.append(e)
                    elif away_team is None:
                        away_team = e.split(" @ ")[0].strip()
                if away_team is None and len(parts) > score_idx + 1:
                    away_team = parts[score_idx + 1].strip().split(" @ ")[0].strip()
                # 如果 away_team 还是无效值（空、@开头、#开头），重置为 None
                if not away_team or away_team.startswith("@") or away_team.startswith("#"):
                    away_team = None
                if away_team is None:
                    # 处理 "v" 格式：home_clean 如 "Brazil v Croatia"，客队在 v 之后
                    if " v " in home_clean:
                        vs = home_clean.split(" v ", 1)
                        home_clean = vs[0].strip()
                        away_team = vs[1].strip()
                if away_team is None:
                    # 比分部分可能包含客队（如 "2-0 (0-0) Nigeria" 或 "2-1 a.e.t. (0-0, 0-0) Algeria"）
                    # 消耗所有比分和注释文本，剩余即客队
                    score_tail = re.match(
                        r"^\d+[–\-]\d+"  # 基础比分
                        r"(?:\s+\(\d+[–\-]\d+(?:,\s*\d+[–\-]\d+)*\))?"  # 括号比分 (0-0) 或 (0-0, 2-2)
                        r"(?:\s+(?:a\.e\.t|pen)\.?"  # a.e.t 或 pen 标记
                        r"(?:\s*\([^)]*\))?"  # 括号注释
                        r"(?:,\s*\d+[–\-]\d+\s+pen\.?)?)*"  # 逗号+pen
                        r"\s*",
                        score_part
                    )
                    if score_tail:
                        rest = score_part[score_tail.end():].strip()
                        if rest and "@" not in rest and not re.match(r"^#", rest):
                            away_team = rest.split(" @ ")[0].strip()
                score_text = score_part
                if extras:
                    score_text = score_part + " " + " ".join(extras)
                home_score, away_score, home_aet, away_aet, home_pen, away_pen = \
                    parse_score(score_text)
                match_data = {"home": home_clean.strip(), "away": away_team.strip(),
                    "home_score": home_score, "away_score": away_score,
                    "home_aet": home_aet, "away_aet": away_aet,
                    "home_pen": home_pen, "away_pen": away_pen, "date": current_date}
                current_matches.append(match_data)
                continue

            # 回退 B：去掉行首日期前缀后按空格分割找比分
            text_stripped = re.sub(
                r"^(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?"
                r"(?:\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)|"
                r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2})"
                r"(?:\s+\d{1,2}:\d{2})?\s+", "", stripped
            )
            # 也去掉 UTC 偏移
            text_stripped = re.sub(r"^\d{1,2}:\d{2}(?:\s+UTC[+-]\d+)?\s+", "", text_stripped)
            # 找比分
            sm = re.search(r"(\d+[–\-]\d+(?:\s+\([^)]+\))?)", text_stripped)
            if sm:
                before = text_stripped[:sm.start()].strip()
                after = text_stripped[sm.end():].strip()
                # 去除 after 开头的比分注释（a.e.t., pen., (0-0, 0-0) 等）
                # 这些是比分的一部分，不是客队名
                while True:
                    old_after = after
                    after = re.sub(r"^\([^)]*\)\s*", "", after).strip()
                    after = re.sub(r"^a\.e\.t\.?\s*", "", after).strip()  
                    after = re.sub(r"^pen\.?\s*", "", after).strip()
                    after = re.sub(r"^,\s*", "", after).strip()
                    if after == old_after:
                        break
                # 从 after 中提取客队：跳过 a.e.t./pen 注释，找最后一个形如队名的部分
                # 按2+空格分割 after，取第一段作为客队
                after_parts = re.split(r"\s{2,}", after)
                if len(after_parts) >= 1:
                    # 去 @ 体育场
                    away = after_parts[0].split(" @ ")[0].strip()
                    # 如果 away 含 a.e.t. 等，说明没有2空格分隔，尝试取最后一个2+空格块
                    if "a.e.t" in after or "pen" in after:
                        # 找不含 a.e.t/pen 的最后一段
                        for ap in reversed(after_parts):
                            ap_clean = ap.split(" @ ")[0].strip()
                            if ap_clean and "a.e.t" not in ap_clean and "pen" not in ap_clean:
                                away = ap_clean
                                break
                else:
                    away = after.split(" @ ")[0].strip()
                
                if ' v ' in before:
                    vs = before.split(' v ', 1)
                    home = vs[0].strip()
                    rest = vs[1] + " " + away
                    sc2 = re.search(r"(\d+[–\-]\d+)", rest)
                    if sc2:
                        away = rest[sc2.end():].strip()
                else:
                    home = before
                
                # 验证客队名：不以数字/@/(/a.e.t 开头
                away_ok = away and away[0] not in "0123456789@(" and not away.startswith("a.e.t")
                if away_ok and home:
                    hs, aws, ha, aa, hp, ap = parse_score(sm.group(1))
                    current_matches.append({"home": home.strip(), "away": away.strip(),
                        "home_score": hs, "away_score": aws,
                        "home_aet": ha, "away_aet": aa,
                        "home_pen": hp, "away_pen": ap, "date": current_date})
                    continue

            # 回退 C：纯 v 格式（无日期前缀）
            if ' v ' in stripped and re.search(r"\d+[–\-]\d+", stripped):
                parts = re.split(r"\s{2,}", stripped)
                for p in parts:
                    if ' v ' in p:
                        vs = p.split(' v ', 1)
                        home = vs[0].strip()
                        rest = vs[1]
                        sm = re.search(r"(\d+[–\-]\d+)", rest)
                        if sm:
                            away = rest[sm.end():].split(" @ ")[0].strip()
                            hs, aws, ha, aa, hp, ap = parse_score(sm.group(1))
                            current_matches.append({"home": home, "away": away,
                                "home_score": hs, "away_score": aws,
                                "home_aet": ha, "away_aet": aa,
                                "home_pen": hp, "away_pen": ap,
                                "date": current_date})
                        break
                continue

            # 全部失败
            continue

        # 通过正则匹配成功
        home_team = match.group(2).strip()
        score_text = match.group(3).strip()
        away_team = match.group(4).strip()

        # 清理队名中的时间和比赛编号
        home_team = re.sub(r"^\d{1,2}:\d{2}\s+", "", home_team)
        home_team = re.sub(r"^\(\d+\)\s+", "", home_team)
        away_team = away_team.split(" @ ")[0].strip()

        home_score, away_score, home_aet, away_aet, home_pen, away_pen = \
            parse_score(score_text)

        match_data = {
            "home": home_team.strip(),
            "away": away_team.strip(),
            "home_score": home_score,
            "away_score": away_score,
            "home_aet": home_aet,
            "away_aet": away_aet,
            "home_pen": home_pen,
            "away_pen": away_pen,
            "date": current_date,
        }
        current_matches.append(match_data)

    # 保存最后一阶段的比赛
    if current_matches:
        result["stages"].append({
            "stage": current_stage,
            "group": current_group,
            "matches": current_matches,
            "date": current_date,
        })

    # 补充主办国
    if not result["host"] and result["year"] in FALLBACK_HOSTS:
        result["host"] = FALLBACK_HOSTS[result["year"]]

    if result["year"] == 0:
        print(f"  [WARN] {file_path}: 未找到 tournament 头信息")
        return None

    # 统计比赛数
    total_matches = sum(len(s["matches"]) for s in result["stages"])
    result["num_matches"] = total_matches
    # 统计球队数（基于分组定义或实际出现的球队）
    all_teams = set()
    for group_teams in result["groups"].values():
        all_teams.update(group_teams)
    result["num_teams"] = len(all_teams) if all_teams else 0

    return result


# =============================================================================
# SQLite 导入
# =============================================================================


def init_db(db_path: str) -> sqlite3.Connection:
    """
    初始化 SQLite 数据库，创建表结构。

    Args:
        db_path: 数据库文件路径

    Returns:
        sqlite3.Connection
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tournaments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL UNIQUE,
            host TEXT NOT NULL,
            dates TEXT DEFAULT '',
            num_teams INTEGER DEFAULT 0,
            num_matches INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tournament_id INTEGER NOT NULL,
            stage TEXT DEFAULT '',
            group_name TEXT DEFAULT '',
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            home_score INTEGER NOT NULL DEFAULT 0,
            away_score INTEGER NOT NULL DEFAULT 0,
            aet_home INTEGER,
            aet_away INTEGER,
            penalties_home INTEGER,
            penalties_away INTEGER,
            date TEXT DEFAULT '',
            FOREIGN KEY (tournament_id) REFERENCES tournaments(id),
            FOREIGN KEY (home_team_id) REFERENCES teams(id),
            FOREIGN KEY (away_team_id) REFERENCES teams(id)
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_matches_tournament
            ON matches(tournament_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_matches_teams
            ON matches(home_team_id, away_team_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_matches_stage
            ON matches(stage)
    """)

    conn.commit()
    return conn


def _get_or_create_team(conn: sqlite3.Connection, team_name: str) -> int:
    """
    获取球队 ID，如不存在则创建。

    Args:
        conn: 数据库连接
        team_name: 球队名称

    Returns:
        int: 球队 ID
    """
    cursor = conn.execute("SELECT id FROM teams WHERE name = ?", (team_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    conn.execute("INSERT INTO teams (name) VALUES (?)", (team_name,))
    return cursor.lastrowid  # type: ignore[return-value]


def import_tournament_data(
    conn: sqlite3.Connection,
    parsed: Dict,
    file_path: str,
    verbose: bool = False,
) -> bool:
    """
    将解析后的 tournament 数据导入数据库。

    Args:
        conn: 数据库连接
        parsed: parse_tournament_file() 返回的结构化数据
        file_path: 源文件路径（仅用于日志）
        verbose: 是否输出详细信息

    Returns:
        bool: 是否成功导入
    """
    year = parsed["year"]
    host = parsed["host"]
    dates = parsed["dates"]
    num_teams = parsed["num_teams"]
    num_matches = parsed["num_matches"]

    # 检查是否已导入（追加模式：淘汰赛文件追加到已有 tournament）
    cursor = conn.execute("SELECT id, num_matches FROM tournaments WHERE year = ?", (year,))
    existing = cursor.fetchone()
    is_append = existing is not None
    if is_append:
        tournament_id = existing[0]
        old_match_count = existing[1]
        if verbose:
            print(f"  [APPEND] {year} {host} — 已有 {old_match_count} 场比赛，追加 {num_matches} 场")
    else:
        cursor = conn.execute(
            "INSERT INTO tournaments (year, host, dates, num_teams, num_matches) "
            "VALUES (?, ?, ?, ?, ?)",
            (year, host, dates, num_teams, num_matches),
        )
        tournament_id = cursor.lastrowid

    # 先确保所有球队存在（基于分组定义）
    all_team_names = set()
    for group_teams in parsed["groups"].values():
        for team_name in group_teams:
            all_team_names.add(team_name)
    # 也看看比赛中的球队
    for stage_data in parsed["stages"]:
        for match in stage_data["matches"]:
            all_team_names.add(match["home"])
            all_team_names.add(match["away"])

    if verbose and is_append:
        # 打印球队名检查
        bad_names = [n for n in all_team_names if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", n, re.I)]
        if bad_names:
            print(f"  [WARN] 包含月份名的球队: {bad_names}")

    team_ids: Dict[str, int] = {}
    for name in all_team_names:
        if name and not re.match(r"^W\d+|L\d+|TBD", name):
            team_ids[name] = _get_or_create_team(conn, name)

    # 插入比赛
    match_count = 0
    for stage_data in parsed["stages"]:
        stage_name = stage_data["stage"] or "Unknown"
        group_name = stage_data["group"] or ""
        params: List[Tuple] = []

        for match in stage_data["matches"]:
            home = match["home"].strip()
            away = match["away"].strip()

            # 跳过 placeholder 比赛
            if re.match(r"^W\d+|L\d+|^\(\d+\)|TBD", home) or \
               re.match(r"^W\d+|L\d+|TBD", away):
                continue

            home_id = team_ids.get(home) or _get_or_create_team(conn, home)
            away_id = team_ids.get(away) or _get_or_create_team(conn, away)
            team_ids[home] = home_id
            team_ids[away] = away_id

            # 确定阶段名：对于淘汰赛，使用原始阶段名
            # 对于小组赛，如果有组名，用 "Group {name}"
            if group_name and not re.match(
                r"^(Round of|Quarter|Semi|Final|Match for third|Third)",
                stage_name,
                re.IGNORECASE,
            ):
                display_group = group_name
            else:
                display_group = ""

            # 阶段名如 "Matchday X" 也应映射到 "Group" 阶段
            # 但如果 stage_name 已经是 Round of X 这类，保持不变
            if display_group and not re.match(
                r"^(Round of|Quarter|Semi|Final|Match for third|Third|Group)",
                stage_name,
                re.IGNORECASE,
            ):
                pass  # keep group

            params.append((
                tournament_id, stage_name, display_group,
                home_id, away_id,
                match["home_score"], match["away_score"],
                match["home_aet"], match["away_aet"],
                match["home_pen"], match["away_pen"],
                match["date"],
            ))
            match_count += 1

        if params:
            try:
                conn.executemany(
                    "INSERT INTO matches "
                    "(tournament_id, stage, group_name, home_team_id, away_team_id, "
                    "home_score, away_score, aet_home, aet_away, "
                    "penalties_home, penalties_away, date) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params,
                )
            except sqlite3.IntegrityError:
                for p in params:
                    t = conn.execute("SELECT id FROM tournaments WHERE id=?", (p[0],)).fetchone()
                    h = conn.execute("SELECT id,name FROM teams WHERE id=?", (p[3],)).fetchone()
                    a = conn.execute("SELECT id,name FROM teams WHERE id=?", (p[4],)).fetchone()
                    print("  FAIL: tid=%d(tourn_exists=%s) home_id=%d(h=%s) away_id=%d(a=%s) %s-%s date=%s" % (
                        p[0], t is not None, p[3], h, p[4], a, p[5], p[6], p[11]))
                if any(p[3] is None or p[4] is None for p in params):
                    print("  ROOT CAUSE: some team_id is None")
                raise

    # 更新赛事总比赛数（追加模式时累计）
    total = match_count
    if is_append:
        existing_total = conn.execute(
            "SELECT num_matches FROM tournaments WHERE id = ?", (tournament_id,)
        ).fetchone()[0]
        total = existing_total + match_count
    conn.execute(
        "UPDATE tournaments SET num_matches = ?, num_teams = MAX(num_teams, ?) WHERE id = ?",
        (total, len(team_ids), tournament_id),
    )
    conn.commit()

    if verbose:
        action = "追加" if is_append else "导入"
        print(f"  [OK] {year} {host}: {action} {match_count} 场比赛")
    return True


# =============================================================================
# 导入流程
# =============================================================================


def import_all(
    data_dir: str,
    db_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """
    扫描目录并导入所有世界杯数据。

    Args:
        data_dir: worldcup-data 根目录
        db_path: SQLite 数据库路径（默认 data/worldcup.db）
        verbose: 是否输出详细信息

    Returns:
        int: 成功导入的比赛总数
    """
    db = db_path or DB_PATH

    # 确保 data/ 目录存在
    os.makedirs(os.path.dirname(db), exist_ok=True)

    files = find_data_files(data_dir)
    if not files:
        print(f"[ERROR] worldcup-data 目录未找到 cup.txt 文件: {data_dir}")
        return 0

    if verbose:
        print(f"[INFO] 找到 {len(files)} 个数据文件")
        print(f"[INFO] 目标数据库: {db}")

    conn = init_db(db)

    total_matches = 0
    success_count = 0

    for file_path in files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        if verbose:
            print(f"\n正在导入: {dir_name}/")
        parsed = parse_tournament_file(file_path)
        if parsed is None:
            print(f"  [FAIL] 解析失败: {file_path}")
            continue

        ok = import_tournament_data(conn, parsed, file_path, verbose=verbose)
        if ok:
            success_count += 1
            total_matches += parsed.get("num_matches", 0)

    conn.close()

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"导入完成: {success_count}/{len(files)} 个赛事, {total_matches} 场比赛")
        print(f"{'=' * 50}")

    return total_matches


def get_db_status(db_path: Optional[str] = None) -> Dict:
    """
    获取数据库状态信息。

    Args:
        db_path: 数据库路径

    Returns:
        Dict: { "exists": bool, "tournaments": int, "teams": int, "matches": int }
    """
    db = db_path or DB_PATH
    result: Dict = {
        "exists": os.path.isfile(db),
        "tournaments": 0,
        "teams": 0,
        "matches": 0,
    }
    if not result["exists"]:
        return result

    try:
        conn = sqlite3.connect(db)
        result["tournaments"] = conn.execute(
            "SELECT COUNT(*) FROM tournaments"
        ).fetchone()[0]
        result["teams"] = conn.execute(
            "SELECT COUNT(*) FROM teams"
        ).fetchone()[0]
        result["matches"] = conn.execute(
            "SELECT COUNT(*) FROM matches"
        ).fetchone()[0]
        conn.close()
    except sqlite3.Error:
        pass

    return result


# =============================================================================
# CLI 入口
# =============================================================================

if __name__ == "__main__":
    import sys

    # 默认导入
    default_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..",
        "worldcup-data",
    )

    data_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    if not data_dir:
        # 尝试从桌面路径
        candidates = [
            r"C:\Users\13918\Desktop\_app_data_所有对话_主对话_worldcup-predictor"
            r"\2026-worldcup-predictor\worldcup-data",
        ]
        for c in candidates:
            if os.path.isdir(c):
                data_dir = c
                break

    if not data_dir or not os.path.isdir(data_dir):
        print(f"用法: python data/importer.py <worldcup-data目录>")
        print(f"或配置默认路径")
        sys.exit(1)

    import_all(data_dir, verbose=True)
