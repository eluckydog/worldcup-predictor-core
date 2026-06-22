#!/usr/bin/env python3
"""Single source of truth: 72 group stage matches for 2026 World Cup."""

# (group, home, away, home_score, away_score, matchday, source)
# home_score=None → not yet played

MATCHES = [
    # ── Group A ──
    ("A","Mexico","South Africa",2,0,1,"DB"),
    ("A","South Korea","Czech Republic",2,1,1,"DB"),
    ("A","Czech Republic","South Africa",1,1,2,"WEB"),
    ("A","Mexico","South Korea",1,0,2,"WEB"),
    ("A","Czech Republic","Mexico",None,None,3,"PENDING"),
    ("A","South Africa","South Korea",None,None,3,"PENDING"),

    # ── Group B ──
    ("B","Canada","Bosnia and Herzegovina",1,1,1,"WEB"),
    ("B","Switzerland","Qatar",1,1,1,"WEB"),
    ("B","Switzerland","Bosnia and Herzegovina",4,1,2,"WEB"),
    ("B","Canada","Qatar",6,0,2,"WEB"),
    ("B","Switzerland","Canada",None,None,3,"PENDING"),
    ("B","Qatar","Bosnia and Herzegovina",None,None,3,"PENDING"),

    # ── Group C ──
    ("C","Morocco","Brazil",1,1,1,"WEB"),
    ("C","Haiti","Scotland",0,1,1,"WEB"),
    ("C","Scotland","Morocco",0,1,2,"WEB"),
    ("C","Brazil","Haiti",3,0,2,"WEB"),
    ("C","Brazil","Scotland",None,None,3,"PENDING"),
    ("C","Morocco","Haiti",None,None,3,"PENDING"),

    # ── Group D ──
    ("D","Paraguay","USA",1,4,1,"WEB"),
    ("D","Turkey","Australia",0,2,1,"WEB"),
    ("D","USA","Australia",2,0,2,"WEB"),
    ("D","Turkey","Paraguay",0,1,2,"WEB"),
    ("D","USA","Turkey",None,None,3,"PENDING"),
    ("D","Australia","Paraguay",None,None,3,"PENDING"),

    # ── Group E ──
    ("E","Curaçao","Germany",1,7,1,"WEB"),
    ("E","Ecuador","Côte d'Ivoire",0,1,1,"WEB"),
    ("E","Germany","Côte d'Ivoire",2,1,2,"WEB"),
    ("E","Ecuador","Curaçao",0,0,2,"WEB"),
    ("E","Germany","Ecuador",None,None,3,"PENDING"),
    ("E","Côte d'Ivoire","Curaçao",None,None,3,"PENDING"),

    # ── Group F ──
    ("F","Japan","Netherlands",2,2,1,"WEB"),
    ("F","Sweden","Tunisia",5,1,1,"WEB"),
    ("F","Netherlands","Sweden",5,1,2,"WEB"),
    ("F","Tunisia","Japan",0,4,2,"WEB"),
    ("F","Netherlands","Tunisia",None,None,3,"PENDING"),
    ("F","Sweden","Japan",None,None,3,"PENDING"),

    # ── Group G ──
    ("G","Belgium","Egypt",1,1,1,"WEB"),
    ("G","New Zealand","Iran",2,2,1,"WEB"),
    ("G","Belgium","Iran",0,0,2,"WEB"),
    ("G","New Zealand","Egypt",1,3,2,"WEB"),
    ("G","Belgium","New Zealand",None,None,3,"PENDING"),
    ("G","Iran","Egypt",None,None,3,"PENDING"),

    # ── Group H ──
    ("H","Cape Verde","Spain",0,0,1,"WEB"),
    ("H","Uruguay","Saudi Arabia",1,1,1,"WEB"),
    ("H","Spain","Saudi Arabia",4,0,2,"WEB"),
    ("H","Uruguay","Cape Verde",2,2,2,"WEB"),
    ("H","Spain","Uruguay",None,None,3,"PENDING"),
    ("H","Saudi Arabia","Cape Verde",None,None,3,"PENDING"),

    # ── Group I ──
    ("I","France","Senegal",3,1,1,"WEB"),
    ("I","Norway","Iraq",4,1,1,"WEB"),
    ("I","Iraq","France",None,None,2,"PENDING"),
    ("I","Norway","Senegal",None,None,2,"PENDING"),
    ("I","France","Norway",None,None,3,"PENDING"),
    ("I","Iraq","Senegal",None,None,3,"PENDING"),

    # ── Group J ──
    ("J","Algeria","Argentina",0,3,1,"WEB"),
    ("J","Jordan","Austria",1,3,1,"WEB"),
    ("J","Austria","Argentina",None,None,2,"PENDING"),
    ("J","Jordan","Algeria",None,None,2,"PENDING"),
    ("J","Algeria","Austria",None,None,3,"PENDING"),
    ("J","Jordan","Argentina",None,None,3,"PENDING"),

    # ── Group K ──
    ("K","Portugal","DR Congo",1,1,1,"WEB"),
    ("K","Uzbekistan","Colombia",1,3,1,"WEB"),
    ("K","Portugal","Uzbekistan",None,None,2,"PENDING"),
    ("K","Colombia","DR Congo",None,None,2,"PENDING"),
    ("K","Portugal","Colombia",None,None,3,"PENDING"),
    ("K","Uzbekistan","DR Congo",None,None,3,"PENDING"),

    # ── Group L ──
    ("L","Croatia","England",2,4,1,"WEB"),
    ("L","Panama","Ghana",0,1,1,"WEB"),
    ("L","England","Ghana",None,None,2,"PENDING"),
    ("L","Croatia","Panama",None,None,2,"PENDING"),
    ("L","Ghana","Croatia",None,None,3,"PENDING"),
    ("L","England","Panama",None,None,3,"PENDING"),
]

# ── Helpers ──

def completed():
    return [m for m in MATCHES if m[3] is not None]

def upcoming():
    return [m for m in MATCHES if m[3] is None]

def by_group(g):
    return [m for m in MATCHES if m[0] == g]

def by_matchday(md):
    return [m for m in MATCHES if m[5] == md]

def group_list():
    """Return sorted list of group letters."""
    return sorted(set(m[0] for m in MATCHES))
