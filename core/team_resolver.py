"""
2026 感知球队映射 — 自动翻译队名、标记首秀、分组查询

功能：
- resolve_team_name: 中文/简称/Türkiye 等 → 官方英文名称
- is_debutant: 是否为首次参赛球队
- is_host_nation: 是否为主办国
- get_group: 获取球队分组

内置 48 队完整名单（参考 2026 世界杯 12 组 × 4 队格式）。
"""

import json
import os
from typing import Dict, List, Optional, Set


# =============================================================================
# 队名映射（中文/简称 → 官方英文名）
# =============================================================================

TEAM_NAME_MAP: Dict[str, str] = {
    # --- 中文 → 英文 ---
    "巴西": "Brazil",
    "阿根廷": "Argentina",
    "法国": "France",
    "德国": "Germany",
    "英格兰": "England",
    "西班牙": "Spain",
    "葡萄牙": "Portugal",
    "荷兰": "Netherlands",
    "比利时": "Belgium",
    "意大利": "Italy",
    "克罗地亚": "Croatia",
    "瑞士": "Switzerland",
    "丹麦": "Denmark",
    "瑞典": "Sweden",
    "挪威": "Norway",
    "波兰": "Poland",
    "乌克兰": "Ukraine",
    "塞尔维亚": "Serbia",
    "土耳其": "Türkiye",
    "威尔士": "Wales",
    "捷克": "Czech Republic",
    "匈牙利": "Hungary",
    "奥地利": "Austria",
    "罗马尼亚": "Romania",
    "希腊": "Greece",
    "斯洛伐克": "Slovakia",
    "苏格兰": "Scotland",
    "爱尔兰": "Ireland",
    "俄罗斯": "Russia",
    "芬兰": "Finland",
    "斯洛文尼亚": "Slovenia",
    "保加利亚": "Bulgaria",
    "美国": "USA",
    "墨西哥": "Mexico",
    "加拿大": "Canada",
    "牙买加": "Jamaica",
    "哥斯达黎加": "Costa Rica",
    "洪都拉斯": "Honduras",
    "巴拿马": "Panama",
    "特立尼达和多巴哥": "Trinidad and Tobago",
    "乌拉圭": "Uruguay",
    "哥伦比亚": "Colombia",
    "厄瓜多尔": "Ecuador",
    "秘鲁": "Peru",
    "智利": "Chile",
    "巴拉圭": "Paraguay",
    "委内瑞拉": "Venezuela",
    "玻利维亚": "Bolivia",
    "日本": "Japan",
    "韩国": "South Korea",
    "澳大利亚": "Australia",
    "伊朗": "Iran",
    "沙特阿拉伯": "Saudi Arabia",
    "卡塔尔": "Qatar",
    "阿联酋": "United Arab Emirates",
    "伊拉克": "Iraq",
    "阿曼": "Oman",
    "乌兹别克斯坦": "Uzbekistan",
    "中国": "China",
    "尼日利亚": "Nigeria",
    "塞内加尔": "Senegal",
    "摩洛哥": "Morocco",
    "突尼斯": "Tunisia",
    "阿尔及利亚": "Algeria",
    "埃及": "Egypt",
    "加纳": "Ghana",
    "喀麦隆": "Cameroon",
    "科特迪瓦": "Ivory Coast",
    "马里": "Mali",
    "布基纳法索": "Burkina Faso",
    "南非": "South Africa",
    "刚果民主共和国": "DR Congo",
    "赞比亚": "Zambia",
    "新西兰": "New Zealand",
    "斐济": "Fiji",
    "塔希提": "Tahiti",
    "所罗门群岛": "Solomon Islands",
    "新喀里多尼亚": "New Caledonia",
    "巴布亚新几内亚": "Papua New Guinea",

    # --- 常见简称 → 官方名 ---
    "美国": "USA",
    "英格兰": "England",
    "韩国": "South Korea",
    "荷兰": "Netherlands",
    "土耳其": "Türkiye",
    "沙特": "Saudi Arabia",
    "阿联酋": "United Arab Emirates",
    "科特迪瓦": "Ivory Coast",
    "DRC": "DR Congo",
    "刚果(金)": "DR Congo",
    "刚果（金）": "DR Congo",

    # --- 土耳其名特殊（DB 存储为 "Turkey"） ---
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Turkey": "Turkey",

    # --- 别名 ---
    "大韩民国": "South Korea",
    "日本国": "Japan",
    "中华台北": "Chinese Taipei",
    "中国香港": "Hong Kong",
    "英格兰": "England",
    "大不列颠": "England",
    "UK": "England",
    "伊朗伊斯兰共和国": "Iran",
}

# Host nations for 2026
HOST_NATIONS: Set[str] = {"USA", "Mexico", "Canada"}

# Default JSON path
_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "wc2026.json",
)


# =============================================================================
# 数据加载
# =============================================================================

def _load_wc2026_data(data_path: Optional[str] = None) -> Dict:
    """加载 2026 世界杯配置数据"""
    path = data_path or _DEFAULT_DATA_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"无法加载 2026 世界杯数据 ({path}): {e}")


# =============================================================================
# 核心函数
# =============================================================================

def resolve_team_name(name: str) -> str:
    """
    将任意形式的球队名解析为官方英文名称。

    支持：
    - 中文队名（如 "巴西" → "Brazil"）
    - 简称（如 "韩国" → "South Korea"）
    - 特殊拼写（如 "Türkiye" → "Türkiye"）
    - 已经是英文的透传

    Args:
        name: 球队名称（任意格式）

    Returns:
        str: 官方英文名称。无匹配时原样返回。
    """
    trimmed = name.strip()

    # 1. 检查映射表
    if trimmed in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[trimmed]

    # 2. 尝试大小写不敏感匹配
    for key, value in TEAM_NAME_MAP.items():
        if key.lower() == trimmed.lower():
            return value

    # 3. 英文名透传
    return trimmed


def _build_lookup(data: Dict) -> Dict[str, dict]:
    """从 wc2026 数据构建 team_name → team_info 查找表"""
    lookup: Dict[str, dict] = {}
    for group in data.get("groups", []):
        for team in group.get("teams", []):
            name = team.get("name", "")
            if name:
                lookup[name] = team
                # 也建别名查找
                for alias in team.get("aliases", []):
                    lookup[alias] = team
    return lookup


def _get_team_info(
    team_name: str,
    data: Optional[Dict] = None,
) -> Optional[dict]:
    """获取球队信息"""
    resolved = resolve_team_name(team_name)
    wc_data = data or _load_wc2026_data()
    lookup = _build_lookup(wc_data)

    if resolved in lookup:
        return lookup[resolved]

    return None


def is_debutant(team_name: str, data: Optional[Dict] = None) -> bool:
    """
    判断球队是否是首次参加世界杯。

    根据 wc2026.json 中每支球队的 is_debutant 标记。

    Args:
        team_name: 球队名
        data: 可选的 wc2026 数据（避免重复加载）

    Returns:
        bool: 是否为首次参赛
    """
    info = _get_team_info(team_name, data)
    if info is None:
        return False  # 未知球队默认不是首秀
    return info.get("is_debutant", False)


def is_host_nation(team_name: str) -> bool:
    """
    判断球队是否是 2026 世界杯主办国。

    主办国：USA, Mexico, Canada

    Args:
        team_name: 球队名

    Returns:
        bool
    """
    resolved = resolve_team_name(team_name)
    return resolved in HOST_NATIONS


def get_group(team_name: str, data: Optional[Dict] = None) -> Optional[str]:
    """
    获取球队的分组信息。

    Args:
        team_name: 球队名
        data: 可选的 wc2026 数据

    Returns:
        Optional[str]: 分组名（如 "A", "B"），未找到返回 None
    """
    info = _get_team_info(team_name, data)
    if info is None:
        return None
    return info.get("group")


def get_fifa_rank(team_name: str, data: Optional[Dict] = None) -> Optional[int]:
    """
    获取球队的 FIFA 排名。

    Args:
        team_name: 球队名
        data: 可选的 wc2026 数据

    Returns:
        Optional[int]: FIFA 排名，未找到返回 None
    """
    info = _get_team_info(team_name, data)
    if info is None:
        return None
    return info.get("fifa_rank")


def list_teams_in_group(
    group_name: str,
    data: Optional[Dict] = None,
) -> List[Dict]:
    """
    列出某组的所有球队。

    Args:
        group_name: 组名（如 "A", "B"）
        data: 可选的 wc2026 数据

    Returns:
        List[Dict]: 球队信息列表
    """
    wc_data = data or _load_wc2026_data()
    for group in wc_data.get("groups", []):
        if group.get("group") == group_name:
            return group.get("teams", [])
    return []


def get_all_teams(data: Optional[Dict] = None) -> List[Dict]:
    """获取所有 48 支球队信息"""
    wc_data = data or _load_wc2026_data()
    teams: List[Dict] = []
    for group in wc_data.get("groups", []):
        teams.extend(group.get("teams", []))
    return teams


# =============================================================================
# 描述性输出
# =============================================================================

def describe_team(team_name: str, data: Optional[Dict] = None) -> str:
    """
    输出球队的完整信息字符串。

    Args:
        team_name: 球队名
        data: 可选的 wc2026 数据

    Returns:
        str: 描述字符串
    """
    resolved = resolve_team_name(team_name)
    wc_data = data or _load_wc2026_data()

    parts = [resolved]

    if is_host_nation(resolved):
        parts.append("[Host] 主办国")

    if is_debutant(resolved, wc_data):
        parts.append("[Debut] 首秀")

    group = get_group(resolved, wc_data)
    if group:
        parts.append(f"[Group] 分组 {group}")

    rank = get_fifa_rank(resolved, wc_data)
    if rank:
        parts.append(f"[FIFA] FIFA #{rank}")

    return " | ".join(parts)
