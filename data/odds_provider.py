"""
多源赔率提供器（Phase 5）— 采集 + 去水 + 加权融合

数据流:
    多个赔率源 → 去抽水 → 加权融合 → OddsFusionResult → λ偏差因子

当前支持的源:
    - 500_AVG  : 500彩票网欧赔平均指数（低抽水 ~6%，已实现）
    - JC_SP    : 竞彩官方SP（高抽水 ~13%，高抽水但有独立定价）
    - FOREIGN  : 国际博彩公司个体赔率（通过500.com个彩页面抓取）

设计原则:
    - 每个源是一个独立函数，失败时不影响其他源
    - 融合权重 = 1 / (1 + 抽水率) 的归一化值（低抽水 → 高权重）
    - 向后兼容: get_all_today_odds() 和 get_odds_for_match() 工作方式不变
    - 缓存: 按日期分文件，各源独立缓存

用法:
    from data.odds_provider import (
        get_multi_source_odds,     # 获取比赛的多源赔率 + 融合结果
        get_fused_odds,            # 仅获取融合后的等效赔率三元组
        get_all_today_odds,        # 向后兼容: 500.com平均指数列表
        get_odds_for_match,        # 向后兼容: 获取指定比赛的500.com均值
        OddsFuser,                 # 融合引擎（直接调用）
        display_multi_source,      # 格式化展示多源数据
    )
"""

import json
import logging
import os
import re
import time
from datetime import date, datetime
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple, Union

from core.data_types import OddsRecord, MultiSourceOdds, OddsFusionResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odds_cache")
REQUEST_TIMEOUT = 10

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# 500.com 赔率页面
_500_ODDS_URL = "http://datachart.500.com/gzsyxw/zoushi/gzsyxw_fb.shtml"

# 竞彩官方SP页面（中国体育彩票官方）
_JC_SP_URL = "https://www.sporttery.cn/"

# ──────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────


def _fetch_page(url: str, timeout: int = REQUEST_TIMEOUT,
                encoding: Optional[str] = None) -> Optional[str]:
    """
    使用requests或urllib获取网页内容。

    Args:
        url: 目标URL
        timeout: 超时秒数
        encoding: 指定编码（None则自动检测）

    Returns:
        HTML内容，失败返回None
    """
    try:
        import requests
        resp = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
        if encoding:
            resp.encoding = encoding
        if resp.status_code == 200 and len(resp.text) > 500:
            return resp.text
        logger.warning("fetch %s returned status=%d, len=%d",
                       url, resp.status_code, len(resp.text))
        return None
    except ImportError:
        logger.info("requests not available, falling back to urllib")
    except Exception as e:
        logger.warning("requests fetch failed: %s, falling back to urllib", e)

    try:
        from urllib.request import Request, urlopen
        req = Request(url, headers=_DEFAULT_HEADERS)
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = encoding or resp.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")
            if len(html) > 500:
                return html
            logger.warning("urllib: short page (%d chars)", len(html))
            return None
    except Exception as e:
        logger.warning("urllib fetch failed: %s", e)
        return None


def _cache_path(source: str, date_str: Optional[str] = None) -> str:
    """获取各源独立缓存文件路径"""
    if date_str is None:
        date_str = date.today().isoformat()
    sanitized = source.replace(" ", "_").lower()
    return os.path.join(CACHE_DIR, f"{sanitized}_{date_str}.json")


def _save_cache(source: str, data: list, date_str: Optional[str] = None) -> None:
    """保存单源赔率数据到独立缓存"""
    if not data:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_data = {
        "fetch_time": datetime.now().isoformat(),
        "date": date_str or date.today().isoformat(),
        "source": source,
        "count": len(data),
        "odds": data,
    }
    path = _cache_path(source, date_str)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info("[%s] 缓存已保存: %s (%d 条)", source, path, len(data))
    except IOError as e:
        logger.warning("[%s] 缓存写入失败: %s", source, e)


def _load_cache(source: str, date_str: Optional[str] = None) -> Optional[list]:
    """从缓存读取单源赔率数据"""
    path = _cache_path(source, date_str)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        odds = data.get("odds", [])
        if odds:
            logger.info("[%s] 缓存已加载: %s (%d 条)", source, path, len(odds))
            return odds
        return None
    except (IOError, json.JSONDecodeError) as e:
        logger.warning("[%s] 缓存读取失败: %s", source, e)
        return None


def remove_juice(odds_home: float, odds_draw: float, odds_away: float
                 ) -> Tuple[float, float, float, float]:
    """
    去除庄家抽水(overround)，返回归一化概率和抽水率。

    Returns:
        Tuple[float, float, float, float]:
            (prob_home, prob_draw, prob_away, juice)
        juice = overround - 1.0，如0.06表示6%抽水率
    """
    if odds_home <= 0 or odds_draw <= 0 or odds_away <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    inv_home = 1.0 / odds_home
    inv_draw = 1.0 / odds_draw
    inv_away = 1.0 / odds_away
    total_inv = inv_home + inv_draw + inv_away
    if total_inv <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    overround = total_inv
    return (inv_home / total_inv, inv_draw / total_inv,
            inv_away / total_inv, overround - 1.0)


def probs_to_odds(prob_home: float, prob_draw: float, prob_away: float,
                  target_juice: float = 0.06) -> Tuple[float, float, float]:
    """
    将归一化概率反转为等效赔率。

    Args:
        prob_home, prob_draw, prob_away: 归一化概率（应和为1.0）
        target_juice: 目标抽水率（默认6%，模拟低抽水源）

    Returns:
        Tuple[float, float, float]: (odds_home, odds_draw, odds_away)
    """
    if any(p <= 0 for p in [prob_home, prob_draw, prob_away]):
        return (0.0, 0.0, 0.0)
    total_prob = prob_home + prob_draw + prob_away
    if total_prob <= 0:
        return (0.0, 0.0, 0.0)
    overround = 1.0 + target_juice
    oh = overround / (prob_home / total_prob)
    od = overround / (prob_draw / total_prob)
    oa = overround / (prob_away / total_prob)
    return (round(oh, 2), round(od, 2), round(oa, 2))


def _resolve_team_name(name_cn: str) -> str:
    """
    将中文队名解析为官方英文名称。
    使用 team_resolver 模块的映射表。
    """
    try:
        from core.team_resolver import resolve_team_name as resolver
        return resolver(name_cn)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("team_resolver 调用失败: %s", e)

    _SIMPLE_CN_MAP = {
        "墨西哥": "Mexico", "加拿大": "Canada", "美国": "USA",
        "巴西": "Brazil", "阿根廷": "Argentina", "法国": "France",
        "德国": "Germany", "英格兰": "England", "西班牙": "Spain",
        "葡萄牙": "Portugal", "荷兰": "Netherlands", "比利时": "Belgium",
        "意大利": "Italy", "克罗地亚": "Croatia", "丹麦": "Denmark",
        "日本": "Japan", "韩国": "South Korea", "澳大利亚": "Australia",
        "沙特阿拉伯": "Saudi Arabia", "伊朗": "Iran", "摩洛哥": "Morocco",
        "塞内加尔": "Senegal", "尼日利亚": "Nigeria", "乌拉圭": "Uruguay",
        "哥伦比亚": "Colombia", "厄瓜多尔": "Ecuador", "瑞士": "Switzerland",
        "波兰": "Poland", "加纳": "Ghana", "喀麦隆": "Cameroon",
        "秘鲁": "Peru", "牙买加": "Jamaica", "土耳其": "Türkiye",
    }
    return _SIMPLE_CN_MAP.get(name_cn, name_cn)


# ──────────────────────────────────────────────────────────────────
# 源1: 500.com 欧赔平均指数（欧洲博彩公司平均赔率）
# ──────────────────────────────────────────────────────────────────

_SOURCE_500_AVG = "500_AVG"


def _fetch_500_avg() -> Optional[list]:
    """从500彩票网抓取欧赔平均指数"""
    html = _fetch_page(_500_ODDS_URL, encoding="gb2312")
    if not html:
        return None

    # 匹配表格行 <tr><td>场次</td><td>赛事</td><td>主队vs客队</td><td>时间</td><td>赔率H</td><td>赔率D</td><td>赔率A</td>
    rows = re.findall(
        r'<tr[^>]*>\s*<td[^>]*>(\d+)</td>\s*'
        r'<td[^>]*>([^<]+)</td>\s*'
        r'<td[^>]*>([^<]+)</td>\s*'
        r'<td[^>]*>([^<]+)</td>\s*'
        r'<td[^>]*>([\d.]+)</td>\s*'
        r'<td[^>]*>([\d.]+)</td>\s*'
        r'<td[^>]*>([\d.]+)</td>',
        html, re.IGNORECASE,
    )

    results = []
    for match_idx, league, matchup, match_time, odds_h, odds_d, odds_a in rows:
        parts = re.split(r'\s*vs\s*', matchup, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        home_cn = parts[0].strip()
        away_cn = parts[1].strip()
        try:
            oh, od, oa = float(odds_h), float(odds_d), float(odds_a)
        except (ValueError, TypeError):
            continue

        # 去水计算
        ph, pd, pa, juice = remove_juice(oh, od, oa)

        results.append({
            "source": _SOURCE_500_AVG,
            "match_idx": int(match_idx),
            "league": league.strip(),
            "home_cn": home_cn,
            "away_cn": away_cn,
            "team_home": _resolve_team_name(home_cn),
            "team_away": _resolve_team_name(away_cn),
            "time": match_time.strip(),
            "odds_home": oh,
            "odds_draw": od,
            "odds_away": oa,
            "prob_home": round(ph, 4),
            "prob_draw": round(pd, 4),
            "prob_away": round(pa, 4),
            "juice": round(juice, 4),
        })

    if not results:
        # 宽松匹配模式
        rows2 = re.findall(
            r'<tr[^>]*>.*?<td[^>]*>(\d+)</td>.*?'
            r'<td[^>]*>([^<]+)</td>.*?'
            r'<td[^>]*>([^<]+)</td>.*?'
            r'(?:<td[^>]*>([^<]*)</td>\s*)?'
            r'<td[^>]*>([\d.]+)</td>\s*'
            r'<td[^>]*>([\d.]+)</td>\s*'
            r'<td[^>]*>([\d.]+)</td>',
            html, re.DOTALL | re.IGNORECASE,
        )
        for match_idx, league, matchup, match_time, odds_h, odds_d, odds_a in rows2:
            parts = re.split(r'\s*vs\s*', matchup, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) != 2:
                continue
            home_cn = parts[0].strip()
            away_cn = parts[1].strip()
            try:
                oh, od, oa = float(odds_h), float(odds_d), float(odds_a)
                ph, pd, pa, juice = remove_juice(oh, od, oa)
                results.append({
                    "source": _SOURCE_500_AVG,
                    "match_idx": int(match_idx),
                    "league": league.strip(),
                    "home_cn": home_cn, "away_cn": away_cn,
                    "team_home": _resolve_team_name(home_cn),
                    "team_away": _resolve_team_name(away_cn),
                    "time": match_time.strip(),
                    "odds_home": oh, "odds_draw": od, "odds_away": oa,
                    "prob_home": round(ph, 4), "prob_draw": round(pd, 4),
                    "prob_away": round(pa, 4), "juice": round(juice, 4),
                })
            except (ValueError, TypeError):
                continue

    return results if results else None


def _get_500_avg(date_str: Optional[str] = None) -> list:
    """获取500.com平均指数，优先网络抓取，回退缓存"""
    odds = _fetch_500_avg()
    if odds:
        _save_cache(_SOURCE_500_AVG, odds, date_str)
        return odds
    cached = _load_cache(_SOURCE_500_AVG, date_str)
    return cached or []


# ──────────────────────────────────────────────────────────────────
# 源2: 竞彩官方SP（中国体育彩票官方赔率）
# ──────────────────────────────────────────────────────────────────

_SOURCE_JC_SP = "JC_SP"


def _fetch_jc_sp() -> Optional[list]:
    """
    从中国体育彩票官网抓取竞彩足球赔率。

    注意: sporttery.cn 可能有反爬虫限制，本函数采用多级回退策略。
    如果直连失败，尝试从500.com的竞彩栏目获取。
    """
    # 尝试1: 从500.com竞彩专栏读取（已有页面包含竞彩标识）
    html = _fetch_page("http://odds.500.com/fenxi/ouzhi-1.shtml",
                       encoding="gb2312")
    if not html:
        return None

    # 500.com 竞彩专栏的表格结构: [场次][联赛][主队vs客队][时间]
    # [竞彩胜][竞彩平][竞彩负][平均胜][平均平][平均负]
    rows = re.findall(
        r'<tr[^>]*>\s*<td[^>]*>(\d+)</td>\s*'        # 场次
        r'<td[^>]*>([^<]+)</td>\s*'                   # 联赛
        r'<td[^>]*>([^<]+)</td>\s*'                   # 主队vs客队
        r'<td[^>]*>([^<]*)</td>\s*'                   # 时间
        r'<td[^>]*>([\d.]+)</td>\s*'                  # 竞彩胜
        r'<td[^>]*>([\d.]+)</td>\s*'                  # 竞彩平
        r'<td[^>]*>([\d.]+)</td>',                    # 竞彩负
        html, re.IGNORECASE,
    )

    results = []
    for match_idx, league, matchup, match_time, oh_s, od_s, oa_s in rows:
        parts = re.split(r'\s*vs\s*', matchup, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        home_cn = parts[0].strip()
        away_cn = parts[1].strip()
        try:
            oh, od, oa = float(oh_s), float(od_s), float(oa_s)
        except (ValueError, TypeError):
            continue

        ph, pd, pa, juice = remove_juice(oh, od, oa)

        results.append({
            "source": _SOURCE_JC_SP,
            "match_idx": int(match_idx),
            "league": league.strip(),
            "home_cn": home_cn, "away_cn": away_cn,
            "team_home": _resolve_team_name(home_cn),
            "team_away": _resolve_team_name(away_cn),
            "time": match_time.strip(),
            "odds_home": oh, "odds_draw": od, "odds_away": oa,
            "prob_home": round(ph, 4), "prob_draw": round(pd, 4),
            "prob_away": round(pa, 4), "juice": round(juice, 4),
        })

    return results if results else None


def _get_jc_sp(date_str: Optional[str] = None) -> list:
    """获取竞彩官方SP，优先网络抓取，回退缓存"""
    odds = _fetch_jc_sp()
    if odds:
        _save_cache(_SOURCE_JC_SP, odds, date_str)
        return odds
    return _load_cache(_SOURCE_JC_SP, date_str) or []


# ──────────────────────────────────────────────────────────────────
# 源3: 国际博彩公司个体赔率（从500.com个彩页面抓取）
# ──────────────────────────────────────────────────────────────────

_SOURCE_FOREIGN = "FOREIGN"


def _fetch_foreign_odds() -> Optional[list]:
    """
    从500彩票网获取国际博彩公司个体赔率。

    500.com的"欧赔"页面列出每家公司的个体赔率:
    http://odds.500.com/fenxi1/ouzhi.php?id={match_id}&ctype=1

    我们抓取主页获取比赛ID列表，然后批量抓取个彩详情页。
    但如果抓取量过大可能触发反爬，这里采用适中策略:
    只抓取主页已有的竞彩赔率对比表（已含多家公司数据）。

    Returns:
        数据格式与500_AVG兼容，加上bookmaker_name字段
    """
    # 尝试抓取500.com竞彩对比页面
    # 该页面展示 竞彩SP + 平均 + 多家主要公司赔率
    html = _fetch_page("http://odds.500.com/fenxi/ouzhi-1.shtml",
                       encoding="gb2312")
    if not html:
        return None

    # 使用宽松模式匹配所有包含赔率的行
    results = []

    # 匹配: <tr> 包含多个 <td> 包含赔率数字
    rows = re.findall(
        r'<tr[^>]*>\s*<td[^>]*>\d+</td>\s*'           # 场次
        r'<td[^>]*>[^<]+</td>\s*'                     # 联赛
        r'<td[^>]*>([^<]+)</td>\s*'                   # 主队vs客队
        r'<td[^>]*>[^<]*</td>\s*'                     # 时间
        r'<td[^>]*>[\d.]+</td>\s*'                    # 竞彩胜
        r'<td[^>]*>[\d.]+</td>\s*'                    # 竞彩平
        r'<td[^>]*>[\d.]+</td>\s*'                    # 竞彩负
        r'<td[^>]*>([\d.]+)</td>\s*'                  # 公司1胜
        r'<td[^>]*>([\d.]+)</td>\s*'                  # 公司1平
        r'<td[^>]*>([\d.]+)</td>\s*'                  # 公司1负
        r'<td[^>]*>([\d.]+)</td>\s*'                  # 公司2胜
        r'<td[^>]*>([\d.]+)</td>\s*'                  # 公司2平
        r'<td[^>]*>([\d.]+)</td>',                    # 公司2负
        html, re.DOTALL | re.IGNORECASE,
    )

    if not rows:
        return None

    for matchup, c1h, c1d, c1a, c2h, c2d, c2a in rows:
        parts = re.split(r'\s*vs\s*', matchup, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        home_cn = parts[0].strip()
        away_cn = parts[1].strip()
        try:
            oh1, od1, oa1 = float(c1h), float(c1d), float(c1a)
            oh2, od2, oa2 = float(c2h), float(c2d), float(c2a)
        except (ValueError, TypeError):
            continue

        # 公司1
        ph1, pd1, pa1, ju1 = remove_juice(oh1, od1, oa1)
        results.append({
            "source": f"{_SOURCE_FOREIGN}_B1",
            "bookmaker": "bet365",  # 推测，500.com首家公司通常是bet365
            "home_cn": home_cn, "away_cn": away_cn,
            "team_home": _resolve_team_name(home_cn),
            "team_away": _resolve_team_name(away_cn),
            "odds_home": oh1, "odds_draw": od1, "odds_away": oa1,
            "prob_home": round(ph1, 4), "prob_draw": round(pd1, 4),
            "prob_away": round(pa1, 4), "juice": round(ju1, 4),
        })

        # 公司2
        ph2, pd2, pa2, ju2 = remove_juice(oh2, od2, oa2)
        results.append({
            "source": f"{_SOURCE_FOREIGN}_B2",
            "bookmaker": "威廉希尔",  # 推测，500.com第二家通常是威廉希尔
            "home_cn": home_cn, "away_cn": away_cn,
            "team_home": _resolve_team_name(home_cn),
            "team_away": _resolve_team_name(away_cn),
            "odds_home": oh2, "odds_draw": od2, "odds_away": oa2,
            "prob_home": round(ph2, 4), "prob_draw": round(pd2, 4),
            "prob_away": round(pa2, 4), "juice": round(ju2, 4),
        })

    return results if results else None


def _get_foreign_odds(date_str: Optional[str] = None) -> list:
    """获取国际博彩赔率，优先网络抓取，回退缓存"""
    odds = _fetch_foreign_odds()
    if odds:
        _save_cache(_SOURCE_FOREIGN, odds, date_str)
        return odds
    return _load_cache(_SOURCE_FOREIGN, date_str) or []


# ──────────────────────────────────────────────────────────────────
# 源4: 模拟赔率（当网络不可用时的回退）
# ──────────────────────────────────────────────────────────────────

def _get_simulated_odds(match_key: str) -> list:
    """
    当网络缓存均不可用时，基于比赛强度不同生成模拟赔率。

    这是最后的保底层，会在输出中标注[模拟]。

    Returns:
        list: 包含一条模拟赔率记录的列表
    """
    # 从比赛键解析队名强度
    teams = match_key.split("_vs_") if "_vs_" in match_key else [match_key, "unknown"]

    # 简化强度判断
    _ELITE = {"Brazil", "France", "Argentina", "Germany", "England", "Spain",
              "Portugal", "Netherlands"}
    _STRONG = {"Belgium", "Croatia", "Uruguay", "Italy", "Morocco", "Senegal",
               "Japan", "South Korea", "USA", "Mexico", "Switzerland", "Denmark"}

    home_power = 3.0 if teams[0] in _ELITE else (
        2.0 if teams[0] in _STRONG else 1.0)
    away_power = 3.0 if teams[1] in _ELITE else (
        2.0 if teams[1] in _STRONG else 1.0)

    if home_power >= away_power + 1.5:
        oh, od, oa = 1.50, 4.00, 6.00
    elif home_power >= away_power + 0.5:
        oh, od, oa = 1.80, 3.40, 4.00
    elif abs(home_power - away_power) < 0.5:
        oh, od, oa = 2.50, 3.20, 2.70
    elif away_power > home_power + 0.5:
        oh, od, oa = 3.50, 3.30, 2.00
    else:
        oh, od, oa = 2.50, 3.20, 2.80

    ph, pd, pa, juice = remove_juice(oh, od, oa)
    return [{
        "source": "SIMULATED",
        "home_cn": teams[0], "away_cn": teams[1],
        "team_home": teams[0], "team_away": teams[1],
        "odds_home": oh, "odds_draw": od, "odds_away": oa,
        "prob_home": round(ph, 4), "prob_draw": round(pd, 4),
        "prob_away": round(pa, 4), "juice": round(juice, 4),
        "_simulated": True,
    }]


# ──────────────────────────────────────────────────────────────────
# 融合引擎
# ──────────────────────────────────────────────────────────────────


class OddsFuser:
    """
    多源赔率加权融合引擎。

    融合策略:
        1. 对每个源，先将赔率去水得到概率
        2. 权重 = 1 / (1 + 抽水率) → 低抽水源获得更高权重
        3. 加权平均所有源的隐含概率
        4. 计算共识分歧度: 各源概率的标准差的均值（衡量市场分歧）
        5. 从融合概率反算等效赔率（使用最低抽水率作为基准）
    """

    @staticmethod
    def compute_weight(juice: float) -> float:
        """根据抽水率计算该源的融合权重"""
        return 1.0 / (1.0 + max(juice, 0.0))

    @staticmethod
    def fuse_source_records(
        records: List[dict],
        home_team: str = "",
        away_team: str = "",
    ) -> OddsFusionResult:
        """
        融合多条赔率记录。

        Args:
            records: 各源的赔率数据（含去水后概率）
            home_team: 主队名
            away_team: 客队名

        Returns:
            OddsFusionResult
        """
        if not records:
            return OddsFusionResult()

        # 提取有效的去水概率
        valid_sources = []
        for r in records:
            ph = r.get("prob_home", 0) or 0
            pd = r.get("prob_draw", 0) or 0
            pa = r.get("prob_away", 0) or 0
            if ph > 0 and pd > 0 and pa > 0:
                juice = r.get("juice", 0.06) or 0.06
                wt = OddsFuser.compute_weight(juice)
                valid_sources.append((ph, pd, pa, wt, r))

        if not valid_sources:
            return OddsFusionResult()

        # 加权平均
        total_weight = sum(v[3] for v in valid_sources)
        if total_weight <= 0:
            fused_h = sum(v[0] for v in valid_sources) / len(valid_sources)
            fused_d = sum(v[1] for v in valid_sources) / len(valid_sources)
            fused_a = sum(v[2] for v in valid_sources) / len(valid_sources)
        else:
            fused_h = sum(v[0] * v[3] for v in valid_sources) / total_weight
            fused_d = sum(v[1] * v[3] for v in valid_sources) / total_weight
            fused_a = sum(v[2] * v[3] for v in valid_sources) / total_weight

        # 归一化（确保求和=1.0）
        prob_sum = fused_h + fused_d + fused_a
        if prob_sum > 0:
            fused_h /= prob_sum
            fused_d /= prob_sum
            fused_a /= prob_sum

        # 反算等效赔率（使用各源中最小的抽水率）
        min_juice = min(v[3] for v in valid_sources)  # 权重最大=juice最小
        # 找到权重最大的源对应的juice
        best_juice = 0.06
        for v in valid_sources:
            if v[3] >= max(vv[3] for vv in valid_sources):
                best_juice = v[4].get("juice", 0.06) or 0.06
                break

        e_oh, e_od, e_oa = probs_to_odds(fused_h, fused_d, fused_a, target_juice=best_juice)

        # 共识分歧度: 各源概率的标准差均值
        num_sources = len(valid_sources)
        if num_sources >= 2:
            home_probs = [v[0] for v in valid_sources]
            draw_probs = [v[1] for v in valid_sources]
            away_probs = [v[2] for v in valid_sources]
            div_h = stdev(home_probs) if len(set(home_probs)) > 1 else 0.0
            div_d = stdev(draw_probs) if len(set(draw_probs)) > 1 else 0.0
            div_a = stdev(away_probs) if len(set(away_probs)) > 1 else 0.0
            divergence = (div_h + div_d + div_a) / 3.0

            # 最大偏离: 各源与融合值的最大差值
            max_div = 0.0
            for v in valid_sources:
                max_div = max(max_div, abs(v[0] - fused_h),
                              abs(v[1] - fused_d), abs(v[2] - fused_a))
        else:
            divergence = 0.0
            max_div = 0.0

        # 构建源详情
        source_details = []
        for v in valid_sources:
            raw = v[4]
            source_details.append(OddsRecord(
                source_name=raw.get("source", "?"),
                odds_home=raw.get("odds_home", 0),
                odds_draw=raw.get("odds_draw", 0),
                odds_away=raw.get("odds_away", 0),
                prob_home=v[0], prob_draw=v[1], prob_away=v[2],
                juice=raw.get("juice", 0.06),
                weight=round(v[3], 4),
                timestamp=raw.get("_fetch_time", ""),
            ))

        # 计算λ偏差因子（与compute_odds_bias输出对齐）
        from core.engine_poisson import compute_odds_bias
        hb, ab = compute_odds_bias(e_oh, e_od, e_oa)

        return OddsFusionResult(
            fused_home_prob=round(fused_h, 4),
            fused_draw_prob=round(fused_d, 4),
            fused_away_prob=round(fused_a, 4),
            equiv_home_odds=e_oh,
            equiv_draw_odds=e_od,
            equiv_away_odds=e_oa,
            divergence=round(divergence, 4),
            max_divergence=round(max_div, 4),
            consensus_count=num_sources,
            source_details=source_details,
            home_bias=hb,
            away_bias=ab,
            as_tuple=(e_oh, e_od, e_oa),
        )


# ──────────────────────────────────────────────────────────────────
# 比赛匹配工具
# ──────────────────────────────────────────────────────────────────

def _match_team(source_team: str, query_team: str) -> bool:
    """
    队伍名匹配（大小写不敏感、空格忽略）。
    返回: True表示匹配
    """
    s = source_team.strip().lower()
    q = query_team.strip().lower()
    if s == q:
        return True
    # 别名匹配: team_resolver
    try:
        from core.team_resolver import resolve_team_name
        s_resolved = resolve_team_name(source_team).lower()
        q_resolved = resolve_team_name(query_team).lower()
        return s_resolved == q_resolved
    except Exception:
        pass
    return False


def _find_match_in_source(
    source_odds: list,
    home_team: str,
    away_team: str,
) -> Optional[dict]:
    """在一个源的数据中查找指定比赛的赔率"""
    for entry in source_odds:
        hc = entry.get("home_cn", "")
        ac = entry.get("away_cn", "")
        th = entry.get("team_home", "")
        ta = entry.get("team_away", "")
        if (_match_team(hc, home_team) or _match_team(th, home_team)) and \
           (_match_team(ac, away_team) or _match_team(ta, away_team)):
            return entry
        # 反向匹配（主客队调换）
        if (_match_team(hc, away_team) or _match_team(th, away_team)) and \
           (_match_team(ac, home_team) or _match_team(ta, home_team)):
            entry_reversed = dict(entry)
            entry_reversed["odds_home"], entry_reversed["odds_away"] = \
                entry["odds_away"], entry["odds_home"]
            entry_reversed["prob_home"], entry_reversed["prob_away"] = \
                entry["prob_away"], entry["prob_home"]
            return entry_reversed
    return None


# ──────────────────────────────────────────────────────────────────
# 公开API：多源赔率融合
# ──────────────────────────────────────────────────────────────────


def get_multi_source_odds(
    home_team: str,
    away_team: str,
    date_str: Optional[str] = None,
) -> OddsFusionResult:
    """
    获取指定比赛的多源赔率融合结果。

    流程:
        1. 同时从所有可用源获取赔率数据
        2. 在各源中查找该比赛
        3. 将找到的赔率送入融合引擎
        4. 返回 OddsFusionResult（含融合概率、等效赔率、分歧度、各源详情）

    Args:
        home_team: 主队名（中/英文）
        away_team: 客队名
        date_str: 日期，None表示今日

    Returns:
        OddsFusionResult
    """
    # 收集所有源的赔率记录
    all_records = []
    source_fetchers = [
        ("500_AVG", _get_500_avg),
        ("JC_SP", _get_jc_sp),
        ("FOREIGN", _get_foreign_odds),
    ]

    for source_name, fetcher in source_fetchers:
        try:
            source_data = fetcher(date_str)
            if source_data:
                match = _find_match_in_source(source_data, home_team, away_team)
                if match:
                    all_records.append(match)
                    logger.debug("[%s] 找到 %s vs %s: H=%.2f D=%.2f A=%.2f",
                                 source_name, home_team, away_team,
                                 match.get("odds_home", 0),
                                 match.get("odds_draw", 0),
                                 match.get("odds_away", 0))
        except Exception as e:
            logger.warning("[%s] 获取失败: %s", source_name, e)

    # 如果没有任何源找到数据，使用模拟赔率
    if not all_records:
        match_key = f"{home_team}_vs_{away_team}"
        logger.info("任何源均未找到 %s vs %s，使用[模拟]赔率", home_team, away_team)
        all_records = _get_simulated_odds(match_key)

    # 融合
    return OddsFuser.fuse_source_records(all_records, home_team, away_team)


def get_fused_odds(
    home_team: str,
    away_team: str,
    date_str: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    获取融合后的等效赔率三元组（与compute_odds_bias兼容）。

    这是 get_multi_source_odds() 的简化版本，仅返回赔率三元组。

    Returns:
        Tuple[float, float, float]: (等效主胜赔率, 平赔率, 客胜赔率)
    """
    result = get_multi_source_odds(home_team, away_team, date_str)
    return result.as_tuple


# ──────────────────────────────────────────────────────────────────
# 公开API：向后兼容接口
# ──────────────────────────────────────────────────────────────────

def get_all_today_odds() -> list:
    """
    向后兼容: 获取全部当日赔率（500.com平均指数格式）。

    与 Phase 4 之前的 get_all_today_odds() 返回完全相同的结构。
    """
    odds_500 = _get_500_avg()
    # 转换为旧版格式（向后兼容）
    legacy = []
    for entry in odds_500:
        legacy.append({
            "match_idx": entry.get("match_idx", 0),
            "league": entry.get("league", ""),
            "home_cn": entry["home_cn"],
            "away_cn": entry["away_cn"],
            "team_home": entry["team_home"],
            "team_away": entry["team_away"],
            "time": entry.get("time", ""),
            "odds_home": entry["odds_home"],
            "odds_draw": entry["odds_draw"],
            "odds_away": entry["odds_away"],
        })
    return legacy


def get_odds_for_match(
    home_team: str,
    away_team: str,
    date_str: Optional[str] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    向后兼容: 获取指定比赛的500.com平均赔率。

    与 Phase 4 之前的 get_odds_for_match() 行为完全相同。
    """
    odds_500 = _get_500_avg(date_str)
    if not odds_500:
        return None
    match = _find_match_in_source(odds_500, home_team, away_team)
    if match:
        return (match["odds_home"], match["odds_draw"], match["odds_away"])
    return None


def get_odds_list_display() -> list:
    """
    向后兼容: 获取格式化后的赔率列表（用于 --show-odds）。
    """
    return get_all_today_odds()


# ──────────────────────────────────────────────────────────────────
# 展示工具
# ──────────────────────────────────────────────────────────────────


def display_multi_source(result: OddsFusionResult, indent: str = "") -> str:
    """
    格式化展示多源赔率融合结果。

    Args:
        result: OddsFusionResult
        indent: 缩进前缀

    Returns:
        str: 格式化文本
    """
    if not result or result.equiv_home_odds <= 0:
        return f"{indent}(无赔率数据)"

    lines = []
    lines.append(f"{indent}多源赔率融合结果:")
    lines.append(f"{indent}  等效赔率: {result.equiv_home_odds:.2f} / "
                 f"{result.equiv_draw_odds:.2f} / {result.equiv_away_odds:.2f}")
    lines.append(f"{indent}  融合概率: H {result.fused_home_prob:.1%} / "
                 f"D {result.fused_draw_prob:.1%} / A {result.fused_away_prob:.1%}")

    if result.consensus_count > 0:
        lines.append(f"{indent}  参与源数: {result.consensus_count}")
        lines.append(f"{indent}  共识分歧: {result.divergence:.4f}")
        if result.divergence > 0.03:
            lines.append(f"{indent}  ⚠️ 市场分歧较大 (分歧度{result.divergence:.2%})")

        lines.append(f"{indent}  λ偏差因子: H={result.home_bias:.3f}, A={result.away_bias:.3f}")

        for sd in result.source_details:
            sim_tag = " [模拟]" if "SIMULATED" in sd.source_name else ""
            lines.append(f"{indent}    [{sd.source_name}{sim_tag}] "
                         f"赔率={sd.odds_home:.2f}/{sd.odds_draw:.2f}/{sd.odds_away:.2f} "
                         f"→ H {sd.prob_home:.1%}/D {sd.prob_draw:.1%}/A {sd.prob_away:.1%} "
                         f"(抽水{sd.juice:.1%}, 权重{sd.weight:.2f})")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
    print("多源赔率系统自检")
    print("=" * 60)

    # 测试 500.com 源
    print("\n[500.com 平均指数]")
    odds_500 = _get_500_avg()
    if odds_500:
        print(f"  加载 {len(odds_500)} 条")
        for e in odds_500[:3]:
            print(f"  {e['home_cn']} vs {e['away_cn']}: "
                  f"{e['odds_home']:.2f}/{e['odds_draw']:.2f}/{e['odds_away']:.2f} "
                  f"(抽水{e['juice']:.2%})")
    else:
        print("  (无数据)")

    # 测试融合
    print("\n[融合测试 - Mexico vs Canada]")
    fused = get_multi_source_odds("Mexico", "Canada")
    print(display_multi_source(fused, "  "))
