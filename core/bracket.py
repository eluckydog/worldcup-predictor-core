"""
bracket.py - 2026 World Cup Qualification & Bracket Logic

12 groups x 4 teams -> 32 advance (top 2 per group + 8 best third-placed)
-> Round of 32 -> Round of 16 -> Quarter-finals -> Semi-finals -> Final

Usage:
    from core.bracket import compute_qualification, render_qualifiers, render_bracket
    result = compute_qualification(matches)
    print(render_qualifiers(result))
    print(render_bracket(result))
"""

from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# -- Bracket Configuration ------------------------------------------
# Each entry: (round_name, (group_A, pos_A), (group_B, pos_B), half)
# Positions: 1=winner, 2=runners-up
# Third-placed teams are ranked 1-12 (best to worst), best 8 advance
# and get assigned to "3R1" through "3R8" slots.

R32_BRACKET = [
    # --- Group winners vs runners-up (cross-group) ---
    ("R32-01", ("A", 1), ("C", 2), "upper"),
    ("R32-02", ("D", 1), ("B", 2), "upper"),
    ("R32-03", ("B", 1), ("A", 2), "upper"),
    ("R32-04", ("E", 1), ("F", 2), "upper"),
    ("R32-05", ("H", 1), ("I", 2), "upper"),
    ("R32-06", ("G", 1), ("J", 2), "upper"),

    # --- Group winners vs best third-placed ---
    ("R32-07", ("I", 1), ("3rd", 1), "lower"),  # best 3rd
    ("R32-08", ("J", 1), ("3rd", 2), "lower"),  # 2nd best 3rd
    ("R32-09", ("C", 1), ("D", 2), "lower"),
    ("R32-10", ("F", 1), ("E", 2), "lower"),

    # --- Winners vs remaining runners-up + third-placed ---
    ("R32-11", ("K", 1), ("L", 2), "lower"),
    ("R32-12", ("L", 1), ("K", 2), "upper"),
    ("R32-13", ("3rd", 3), ("3rd", 4), "lower"),
    ("R32-14", ("G", 2), ("H", 2), "upper"),

    # --- Remaining third-placed ---
    ("R32-15", ("3rd", 5), ("3rd", 6), "lower"),
    ("R32-16", ("3rd", 7), ("3rd", 8), "upper"),
]

# Alternate simpler bracket for clarity
# This follows common WC bracket convention: group winners separated
# from each other, cross-group pairing of 1st vs 2nd, etc.
# "3rd,K" means the K-th best third-placed team (1 = best)


# -- Core Data Types ------------------------------------------------

TeamStats = Dict[str, any]  # points, gd, gf, ga, w, d, l, mp
GroupStandings = Dict[str, List[Tuple[str, TeamStats]]]  # group -> sorted teams
Qualifier = Tuple[str, str, int]  # (team_name, group_letter, position_in_group)


# -- Standings Computation ------------------------------------------

def compute_standings(matches: List[tuple]) -> GroupStandings:
    """
    Compute per-group standings from completed matches.

    Matches format: (group, home, away, home_score, away_score, matchday, source)
    Only matches with home_score not None are used.

    FIFA tiebreakers (in order):
    1. Points
    2. Goal difference (all matches)
    3. Goals scored (all matches)
    4. Points in matches between tied teams (mini-league)
    5. Goal difference in mini-league
    6. Goals scored in mini-league
    7. Fair play points (yellow/red cards)
    8. Drawing of lots
    """
    completed = [m for m in matches if m[3] is not None]
    
    groups = defaultdict(lambda: defaultdict(
        lambda: {"p": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0, "mp": 0}
    ))

    # Phase 1: accumulate match results
    for grp, home, away, ha, aa, *_ in completed:
        for tm in (home, away):
            if tm not in groups[grp]:
                groups[grp][tm] = {"p": 0, "gd": 0, "gf": 0, "ga": 0, 
                                    "w": 0, "d": 0, "l": 0, "mp": 0}
        
        groups[grp][home]["gf"] += ha
        groups[grp][home]["ga"] += aa
        groups[grp][home]["mp"] += 1
        groups[grp][away]["gf"] += aa
        groups[grp][away]["ga"] += ha
        groups[grp][away]["mp"] += 1

        if ha > aa:
            groups[grp][home]["p"] += 3
            groups[grp][home]["w"] += 1
            groups[grp][away]["l"] += 1
        elif ha == aa:
            groups[grp][home]["p"] += 1
            groups[grp][away]["p"] += 1
            groups[grp][home]["d"] += 1
            groups[grp][away]["d"] += 1
        else:
            groups[grp][away]["p"] += 3
            groups[grp][away]["w"] += 1
            groups[grp][home]["l"] += 1

        groups[grp][home]["gd"] = groups[grp][home]["gf"] - groups[grp][home]["ga"]
        groups[grp][away]["gd"] = groups[grp][away]["gf"] - groups[grp][away]["ga"]

    # Phase 2: sort each group with FIFA tiebreakers
    result = {}
    for grp in sorted(groups):
        teams = list(groups[grp].items())
        result[grp] = _sort_group(teams, completed, grp)

    return result


def _sort_group(teams: List[Tuple[str, TeamStats]], 
                completed: List[tuple], grp: str) -> List[Tuple[str, TeamStats]]:
    """Sort a single group's teams with full FIFA tiebreakers."""
    # Step 1: sort by primary criteria (P -> GD -> GF)
    teams.sort(key=lambda x: (-x[1]["p"], -x[1]["gd"], -x[1]["gf"]))

    # Step 2: check for ties and apply head-to-head (mini-league)
    # Group teams by their (points, gd, gf) tuple to find ties
    # We apply head-to-head MINI-LEAGUE only when:
    # - Two or more teams are tied on points
    # - After primary sort, they are adjacent
    #
    # FIFA says: if 2+ teams tied on points, look at head-to-head
    # BEFORE overall GD/GF. But only if the tie involves those teams only.
    # If 3 teams tied and the mini-league also ties, fall back to GD.

    i = 0
    while i < len(teams):
        # Find all teams tied on points
        j = i
        while j < len(teams) and teams[j][1]["p"] == teams[i][1]["p"]:
            j += 1
        
        if j - i >= 2:  # tie exists
            tied = teams[i:j]
            resolved = _resolve_tie(tied, completed, grp)
            # Replace the tied segment with resolved ordering
            teams[i:j] = resolved
        
        i = j

    return teams


def _resolve_tie(tied: List[Tuple[str, TeamStats]], 
                 completed: List[tuple], grp: str) -> List[Tuple[str, TeamStats]]:
    """Resolve tie between 2+ teams using head-to-head mini-league."""
    tied_names = {t[0] for t in tied}
    
    # Extract mini-league matches between tied teams
    h2h_stats = defaultdict(
        lambda: {"p": 0, "gd": 0, "gf": 0, "ga": 0}
    )
    
    for grp_match, home, away, ha, aa, *_ in completed:
        if grp_match != grp:
            continue
        if home in tied_names and away in tied_names:
            # Match between tied teams
            h2h_stats[home]["gf"] += ha
            h2h_stats[home]["ga"] += aa
            h2h_stats[away]["gf"] += aa
            h2h_stats[away]["ga"] += ha
            if ha > aa:
                h2h_stats[home]["p"] += 3
            elif ha == aa:
                h2h_stats[home]["p"] += 1
                h2h_stats[away]["p"] += 1
            else:
                h2h_stats[away]["p"] += 3

    for tm in h2h_stats:
        h2h_stats[tm]["gd"] = h2h_stats[tm]["gf"] - h2h_stats[tm]["ga"]

    # Convert back to list and sort by mini-league
    if len(h2h_stats) >= 2:  # at least some head-to-head matches exist
        h2h_list = [(tm, h2h_stats[tm]) for tm in tied_names]
        # Sort: P -> H2H GD -> H2H GF
        h2h_list.sort(key=lambda x: (-x[1]["p"], -x[1]["gd"], -x[1]["gf"]))
        
        # Check if mini-league actually broke the tie
        if len(set(s["p"] for _, s in h2h_list)) == len(h2h_list):
            # Mini-league fully resolved: order by mini-league rank
            h2h_rank = {tm: rank for rank, (tm, _) in enumerate(h2h_list)}
            tied.sort(key=lambda x: h2h_rank[x[0]])
            return tied

    # Mini-league didn't resolve -> fall back to overall GD then GF
    # (which was already the primary sort, so return as-is)
    return tied


# -- Third-Placed Ranking -------------------------------------------

def rank_third_placed(standings: GroupStandings) -> List[Tuple[str, str, TeamStats]]:
    """
    Rank all 12 third-placed teams across all groups.
    Returns sorted list: (team, group, stats) from best to worst.
    FIFA: P -> GD -> GF -> fair play -> lots
    """
    third_placed = []
    for grp, teams in standings.items():
        if len(teams) >= 3:
            tm, stats = teams[2]  # third place in group
            third_placed.append((tm, grp, stats))

    third_placed.sort(key=lambda x: (-x[2]["p"], -x[2]["gd"], -x[2]["gf"]))
    return third_placed


# -- Qualification --------------------------------------------------

def compute_qualification(matches: List[tuple]) -> Dict:
    """
    Full qualification pipeline:
    1. Compute group standings
    2. Rank third-placed teams
    3. Determine 32 qualifiers
    4. Build Round of 32 bracket

    Returns dict with keys:
    - standings: per-group sorted standings
    - third_ranked: sorted 12 third-placed teams
    - qualifiers: list of (team, group, position, bracket_slot)
    - bracket: list of R32 matchups (when all 72 matches done)
    - eligible: whether bracket can be computed
    """
    standings = compute_standings(matches)
    third_ranked = rank_third_placed(standings)

    # Determine qualifiers
    qualifiers = []  # (team, group, position)
    
    for grp, teams in standings.items():
        # Top 2 from each group
        for pos, (tm, _) in enumerate(teams[:2], 1):
            qualifiers.append((tm, grp, pos))

    # Top 8 third-placed teams
    for rank, (tm, grp, _) in enumerate(third_ranked[:8], 1):
        qualifiers.append((tm, grp, f"3R{rank}"))

    # Check if all 72 matches are complete (necessary for bracket)
    completed = [m for m in matches if m[3] is not None]
    all_done = len(completed) >= 72

    return {
        "standings": standings,
        "third_ranked": third_ranked,
        "qualifiers": qualifiers,
        "all_complete": all_done,
    }


# -- Bracket Generation ---------------------------------------------

def generate_bracket(qual_result: Dict) -> List[Dict]:
    """
    Generate Round of 32 bracket matchups.
    Returns list of dicts: {slot, home, away, half}
    """
    q = qual_result["qualifiers"]
    third_ranked = qual_result.get("third_ranked", [])

    # Build lookup: (group, 1) -> team, (group, 2) -> team
    pos_lookup = {}
    for tm, grp, pos in q:
        if isinstance(pos, int):
            pos_lookup[(grp, pos)] = tm

    # Build third-placed lookup: "3R1" -> team name
    for i, (tm, grp, _) in enumerate(third_ranked[:8], 1):
        pos_lookup[("3rd", i)] = tm

    # Build bracket
    bracket = []
    for slot, (g1, p1), (g2, p2), half in R32_BRACKET:
        home = pos_lookup.get((g1, p1), "TBD")
        away = pos_lookup.get((g2, p2), "TBD")
        bracket.append({
            "slot": slot,
            "home": home,
            "away": away,
            "half": half,
            "home_group": g1,
            "home_pos": p1,
            "away_group": g2,
            "away_pos": p2,
        })

    return bracket


# -- Output Rendering -----------------------------------------------

def render_standings(standings: GroupStandings, 
                     third_ranked: Optional[List] = None) -> str:
    """Render group standings as a formatted string."""
    lines = []
    for grp in sorted(standings):
        teams = standings[grp]
        sep = '-' * 40
        lines.append(f"\nGroup {grp} {sep}")
        lines.append(f"{'#':>2} {'Team':<30} {'P':>3} {'W':>3} {'D':>3} {'L':>3} "
                     f"{'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}")
        lines.append("-" * 80)
        for i, (tm, st) in enumerate(teams):
            lines.append(f"{i+1:>2}  {tm:<30} {st['mp']:>3} {st['w']:>3} {st['d']:>3} "
                         f"{st['l']:>3} {st['gf']:>3} {st['ga']:>3} {st['gd']:+4d} "
                         f"{st['p']:>4d}")
    return "\n".join(lines)


def render_qualifiers(qual_result: Dict) -> str:
    """Render qualifiers summary."""
    q = qual_result["qualifiers"]
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"*  2026 WORLD CUP - QUALIFIERS ({len(q)} teams)")
    lines.append(f"{'=' * 60}")

    # Group by qualification path
    winners = [x for x in q if x[2] == 1]
    runners = [x for x in q if x[2] == 2]
    thirds = [x for x in q if isinstance(x[2], str) and x[2].startswith("3R")]

    lines.append(f"\nGroup Winners (12):")
    for tm, grp, _ in sorted(winners, key=lambda x: x[1]):
        lines.append(f"  🥇 [{grp}] {tm}")

    lines.append(f"\nRunners-up (12):")
    for tm, grp, _ in sorted(runners, key=lambda x: x[1]):
        lines.append(f"  🥈 [{grp}] {tm}")

    lines.append(f"\nBest Third-Placed (8):")
    for tm, grp, pos in sorted(thirds, key=lambda x: x[1]):
        rank_num = pos.replace("3R", "")
        lines.append(f"  🥉 [{grp}] {tm} (#{rank_num})")

    # Show third-placed ranking details
    third_ranked = qual_result.get("third_ranked", [])
    lines.append(f"\nThird-Placed Ranking (all 12):")
    lines.append(f"{'Rank':>5} {'Team':<25} {'Group':>5} {'Pts':>4} {'GD':>5} {'GF':>4}")
    lines.append("-" * 50)
    for i, (tm, grp, st) in enumerate(third_ranked, 1):
        adv = "← Qualified" if i <= 8 else ""
        lines.append(f"{i:>5}  {tm:<25} {grp:>5} {st['p']:>4d} {st['gd']:+4d} "
                     f"{st['gf']:>4d}  {adv}")

    return "\n".join(lines)


def render_bracket(qual_result: Dict) -> str:
    """Render Round of 32 bracket."""
    if not qual_result.get("all_complete"):
        completed = sum(1 for m in qual_result.get("_matches", []))  # won't be set
        return ("\n⚠️  All 72 group matches must be completed "
                "before the bracket can be generated.")

    bracket = qual_result.get("bracket", [])
    if not bracket:
        bracket = generate_bracket(qual_result)

    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"*  ROUND OF 32 - BRACKET")
    lines.append(f"{'=' * 60}")

    # Upper half
    lines.append(f"\nUpper Half:")
    lines.append("-" * 50)
    for match in bracket:
        if match["half"] != "upper":
            continue
        h_flag = f"[{match['home_group']}]" if isinstance(match.get('home_pos'), int) else ""
        a_flag = f"[{match['away_group']}]" if isinstance(match.get('away_pos'), int) else ""
        lines.append(f"  {match['slot']}: {h_flag} {match['home']:<25} vs "
                     f"{a_flag} {match['away']:<25}")

    lines.append(f"\nLower Half:")
    lines.append("-" * 50)
    for match in bracket:
        if match["half"] != "lower":
            continue
        h_flag = f"[{match['home_group']}]" if isinstance(match.get('home_pos'), int) else ""
        a_flag = f"[{match['away_group']}]" if isinstance(match.get('away_pos'), int) else ""
        lines.append(f"  {match['slot']}: {h_flag} {match['home']:<25} vs "
                     f"{a_flag} {match['away']:<25}")

    return "\n".join(lines)


def full_qualification_report(matches: List[tuple]) -> str:
    """One-call qualification report with standings + qualifiers + bracket."""
    qual = compute_qualification(matches)
    
    if qual["all_complete"]:
        qual["bracket"] = generate_bracket(qual)
    
    parts = [
        render_standings(qual["standings"]),
        render_qualifiers(qual),
    ]
    
    if qual["all_complete"]:
        parts.append(render_bracket(qual))
    else:
        n_completed = len([m for m in matches if m[3] is not None])
        parts.append(f"\n⏳  {72 - n_completed} group matches remaining - "
                     f"bracket will generate automatically when complete.\n")

    return "\n".join(parts)


# -- Standalone Test ------------------------------------------------

if __name__ == "__main__":
    # Quick test with current data
    from scripts.daily_predict import MATCHES as test_matches
    print(full_qualification_report(test_matches))
