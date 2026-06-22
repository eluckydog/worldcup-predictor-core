#!/usr/bin/env python3
"""
stakes.py — Group situation & knockout bracket analysis for draw calibration.

Layer 2 (LLM stakes) + Layer 3 (bracket incentive) of the pipeline.
"""
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from core.bracket import compute_standings, R32_BRACKET

# ── Draw adjustment constants (empirically estimated) ──
ADJ_DRAW_ADVANCES_BOTH = 0.10     # mutual draw incentive
ADJ_DRAW_ADVANCES_BOTH_WIN_INCENTIVE = 0.07  # draw advances but winner gets easier bracket
ADJ_DRAW_ADVANCES_BOTH_HIGH_INCENTIVE = 0.03  # draw advances but massive bracket difference
ADJ_ONE_MUST_WIN = -0.01          # one team needs win, other draw enough
ADJ_BOTH_MUST_WIN = -0.03         # both need win to have any chance
ADJ_SOMEONE_OUT = -0.02           # at least one team already eliminated

# ── Bracket configuration (from core/bracket.py R32_BRACKET) ──

def get_opponent(grp: str, pos: int) -> str:
    """Look up R32 opponent for a group winner (1) or runner-up (2)."""
    for slot, (g1, p1), (g2, p2), _ in R32_BRACKET:
        if g1 == grp and p1 == pos:
            home = True
            opp_g, opp_p = g2, p2
            break
        if g2 == grp and p2 == pos:
            home = False
            opp_g, opp_p = g1, p1
            break
    else:
        return "TBD"
    if isinstance(opp_g, str) and opp_g == "3rd":
        return f"3R{opp_p}"
    return f"Group {opp_g} #{opp_p}"


def bracket_incentive_level(grp: str) -> int:
    """
    How much incentive does the group winner have (0=low, 1=medium, 2=high)?
    
    High (2): Winner plays 3rd place team, runner-up plays group winner.
    Medium (1): Winner plays a runner-up, runner-up plays a different opponent.
    Low (0): Both get equivalent opponents.
    """
    w_opp = get_opponent(grp, 1)
    r_opp = get_opponent(grp, 2)
    if "3R" in w_opp and "3R" not in r_opp:
        return 2  # ⚡ winner gets 3rd place
    if "3R" not in w_opp and "3R" in r_opp:
        return 0  # unusual: runner-up gets easier opponent
    return 1  # moderate


# ── Stakes analysis ──

def analyze_match(
    g: str,
    home: str,
    away: str,
    standings: Dict,
) -> Dict:
    """
    Analyze a single MD3 match's group situation.
    
    Args:
        g: Group letter
        home/away: Team names
        standings: Output from bracket.compute_standings()
    
    Returns:
        Dict with: scenario (str), adj (float), bracket_note (str)
    """
    teams = standings.get(g, [])
    team_map = {tm: st for tm, st in teams}
    
    pts_h = team_map[home]['p'] if home in team_map else 0
    pts_a = team_map[away]['p'] if away in team_map else 0
    
    # Other teams in group
    other_teams = [t for t in team_map if t not in (home, away)]
    others_max_possible = max([team_map[t]['p'] + 3 for t in other_teams]) if other_teams else 0
    
    # Can each team still advance?
    h_alive = pts_h + 3 >= others_max_possible
    a_alive = pts_a + 3 >= others_max_possible
    both_alive = h_alive and a_alive
    at_least_one_dead = not both_alive
    
    # Draw scenarios
    draw_adv_h = pts_h + 1 >= others_max_possible
    draw_adv_a = pts_a + 1 >= others_max_possible
    draw_adv_both = draw_adv_h and draw_adv_a and both_alive
    
    # Scenario classification
    bracket_note = ""
    if at_least_one_dead:
        scenario = "At least one team already eliminated/practically certain"
        adj = ADJ_SOMEONE_OUT
    elif draw_adv_both:
        incentive = bracket_incentive_level(g)
        if incentive >= 2:
            scenario = f"DRAW ADVANCES BOTH (winner→3rd place, runner-up→tough opponent)"
            adj = ADJ_DRAW_ADVANCES_BOTH_HIGH_INCENTIVE
        elif incentive == 1:
            scenario = f"DRAW ADVANCES BOTH"
            adj = ADJ_DRAW_ADVANCES_BOTH
        else:
            scenario = f"DRAW ADVANCES BOTH (no bracket difference)"
            adj = ADJ_DRAW_ADVANCES_BOTH
        bracket_note = f"Winner→{get_opponent(g,1)}, Runner-up→{get_opponent(g,2)}"
    elif not draw_adv_h and not draw_adv_a and both_alive:
        scenario = "BOTH MUST WIN"
        adj = ADJ_BOTH_MUST_WIN
    elif not draw_adv_h:
        scenario = f"{home} must win, {away} draw may be enough"
        adj = ADJ_ONE_MUST_WIN
    elif not draw_adv_a:
        scenario = f"{away} must win, {home} draw may be enough"
        adj = ADJ_ONE_MUST_WIN
    else:
        scenario = "Depends on other match results"
        adj = 0.0
    
    return {
        "scenario": scenario,
        "adj": adj,
        "pts_h": pts_h,
        "pts_a": pts_a,
        "both_alive": both_alive,
        "draw_adv_both": draw_adv_both,
        "bracket_note": bracket_note,
    }


def analyze_standings(standings: Dict) -> str:
    """Render current standings as a concise table."""
    lines = []
    for g in sorted(standings):
        teams = standings[g]
        n_comp = sum(1 for _, st in teams if st['mp'] > 0)
        lines.append(f"\n  Group {g} ({n_comp} played):")
        for i, (tm, st) in enumerate(teams):
            md_str = f"({st['mp']}/2)" if st['mp'] < 2 else "(done)"
            lines.append(f"    {i+1}. {tm:22s} {st['p']}pts GD={st['gd']:+d} {md_str}")
    return "\n".join(lines)


def render_stakes(g: str, home: str, away: str, analysis: Dict) -> str:
    """Render stakes analysis for display."""
    adj = analysis['adj']
    adj_str = f"{adj:+.0%}" if adj != 0 else ""
    pts = f"({analysis['pts_h']}vs{analysis['pts_a']})"
    return (f"  [{g}] {home:22s} vs {away:22s} {pts}\n"
            f"        {analysis['scenario']} {adj_str}\n"
            f"        {analysis['bracket_note']}" if analysis['bracket_note'] else "")


# ── Bracket overview (display all 12 group paths) ──

def bracket_overview() -> str:
    lines = ["R32 BRACKET — Group position opponents:"]
    for g in 'ABCDEFGHIJKL':
        w_opp = get_opponent(g, 1)
        r_opp = get_opponent(g, 2)
        lines.append(f"  [{g}] 1st→{w_opp:30s} | 2nd→{r_opp}")
    return "\n".join(lines)
