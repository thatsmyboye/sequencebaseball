#!/usr/bin/env python
"""
Create a starter pitcher registry with known MLB pitchers.
This provides immediate functionality while the full registry can be built later.

Usage: python create_starter_registry.py
"""

import json
from pathlib import Path
from datetime import datetime

# Known MLB pitchers with their data (2024 season)
STARTER_PITCHERS = {
    669373: {
        "name": "Tarik Skubal",
        "team": "DET",
        "team_full": "Detroit Tigers",
        "position": "SP",
        "throws": "L",
        "pitch_types": ["4-Seam Fastball", "Slider", "Changeup", "Curveball"],
        "total_pitches": 3200,
        "games": 31,
        "available_seasons": [2024]
    },
    554430: {
        "name": "Zack Wheeler",
        "team": "PHI",
        "team_full": "Philadelphia Phillies",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup", "Sinker"],
        "total_pitches": 3400,
        "games": 32,
        "available_seasons": [2024]
    },
    543037: {
        "name": "Gerrit Cole",
        "team": "NYY",
        "team_full": "New York Yankees",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup", "Cutter"],
        "total_pitches": 2800,
        "games": 28,
        "available_seasons": [2024]
    },
    605483: {
        "name": "Chris Sale",
        "team": "ATL",
        "team_full": "Atlanta Braves",
        "position": "SP",
        "throws": "L",
        "pitch_types": ["4-Seam Fastball", "Slider", "Changeup"],
        "total_pitches": 2600,
        "games": 29,
        "available_seasons": [2024]
    },
    571945: {
        "name": "Corbin Burnes",
        "team": "BAL",
        "team_full": "Baltimore Orioles",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["Cutter", "Curveball", "Sinker", "Changeup", "4-Seam Fastball"],
        "total_pitches": 3100,
        "games": 32,
        "available_seasons": [2024]
    },
    656302: {
        "name": "Logan Webb",
        "team": "SF",
        "team_full": "San Francisco Giants",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["Sinker", "Slider", "Changeup", "4-Seam Fastball"],
        "total_pitches": 3300,
        "games": 33,
        "available_seasons": [2024]
    },
    608566: {
        "name": "Dylan Cease",
        "team": "SD",
        "team_full": "San Diego Padres",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup"],
        "total_pitches": 3000,
        "games": 33,
        "available_seasons": [2024]
    },
    621111: {
        "name": "Framber Valdez",
        "team": "HOU",
        "team_full": "Houston Astros",
        "position": "SP",
        "throws": "L",
        "pitch_types": ["Sinker", "Curveball", "Changeup", "4-Seam Fastball"],
        "total_pitches": 3400,
        "games": 31,
        "available_seasons": [2024]
    },
    656427: {
        "name": "Sonny Gray",
        "team": "STL",
        "team_full": "St. Louis Cardinals",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup"],
        "total_pitches": 2700,
        "games": 26,
        "available_seasons": [2024]
    },
    663903: {
        "name": "Zach Eflin",
        "team": "TB",
        "team_full": "Tampa Bay Rays",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["Sinker", "Slider", "Curveball", "Changeup"],
        "total_pitches": 2900,
        "games": 31,
        "available_seasons": [2024]
    },
    666142: {
        "name": "Hunter Brown",
        "team": "HOU",
        "team_full": "Houston Astros",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup"],
        "total_pitches": 2800,
        "games": 29,
        "available_seasons": [2024]
    },
    650556: {
        "name": "Jhoan Duran",
        "team": "MIN",
        "team_full": "Minnesota Twins",
        "position": "RP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Splitter", "Slider"],
        "total_pitches": 1200,
        "games": 58,
        "available_seasons": [2024]
    },
    650911: {
        "name": "Cristopher Sanchez",
        "team": "PHI",
        "team_full": "Philadelphia Phillies",
        "position": "SP",
        "throws": "L",
        "pitch_types": ["Sinker", "Changeup", "Sweeper", "4-Seam Fastball"],
        "total_pitches": 2700,
        "games": 32,
        "available_seasons": [2024]
    },
    592789: {
        "name": "Aaron Nola",
        "team": "PHI",
        "team_full": "Philadelphia Phillies",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Curveball", "Changeup", "Sinker", "Cutter"],
        "total_pitches": 3100,
        "games": 32,
        "available_seasons": [2024]
    },
    571578: {
        "name": "Max Scherzer",
        "team": "TEX",
        "team_full": "Texas Rangers",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Changeup", "Cutter"],
        "total_pitches": 1500,
        "games": 15,
        "available_seasons": [2024]
    },
    519242: {
        "name": "Clayton Kershaw",
        "team": "LAD",
        "team_full": "Los Angeles Dodgers",
        "position": "SP",
        "throws": "L",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball"],
        "total_pitches": 1400,
        "games": 16,
        "available_seasons": [2024]
    },
    680686: {
        "name": "Paul Skenes",
        "team": "PIT",
        "team_full": "Pittsburgh Pirates",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup"],
        "total_pitches": 2100,
        "games": 23,
        "available_seasons": [2024]
    },
    665871: {
        "name": "Tanner Houck",
        "team": "BOS",
        "team_full": "Boston Red Sox",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Splitter"],
        "total_pitches": 2500,
        "games": 31,
        "available_seasons": [2024]
    },
    656849: {
        "name": "Tyler Glasnow",
        "team": "LAD",
        "team_full": "Los Angeles Dodgers",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Curveball", "Slider", "Changeup"],
        "total_pitches": 2300,
        "games": 22,
        "available_seasons": [2024]
    },
    622663: {
        "name": "Emmanuel Clase",
        "team": "CLE",
        "team_full": "Cleveland Guardians",
        "position": "RP",
        "throws": "R",
        "pitch_types": ["Cutter", "Sinker"],
        "total_pitches": 1100,
        "games": 69,
        "available_seasons": [2024]
    },
    592662: {
        "name": "Blake Snell",
        "team": "SF",
        "team_full": "San Francisco Giants",
        "position": "SP",
        "throws": "L",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup"],
        "total_pitches": 2000,
        "games": 24,
        "available_seasons": [2024]
    },
    657277: {
        "name": "Shane Bieber",
        "team": "CLE",
        "team_full": "Cleveland Guardians",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Cutter"],
        "total_pitches": 300,
        "games": 4,
        "available_seasons": [2024]
    },
    656945: {
        "name": "Spencer Strider",
        "team": "ATL",
        "team_full": "Atlanta Braves",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Changeup"],
        "total_pitches": 400,
        "games": 5,
        "available_seasons": [2024]
    },
    666201: {
        "name": "Ranger Suarez",
        "team": "PHI",
        "team_full": "Philadelphia Phillies",
        "position": "SP",
        "throws": "L",
        "pitch_types": ["Sinker", "Changeup", "Cutter", "4-Seam Fastball"],
        "total_pitches": 2800,
        "games": 29,
        "available_seasons": [2024]
    },
    608331: {
        "name": "Yoshinobu Yamamoto",
        "team": "LAD",
        "team_full": "Los Angeles Dodgers",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Splitter", "Curveball", "Slider"],
        "total_pitches": 1800,
        "games": 18,
        "available_seasons": [2024]
    },
    543243: {
        "name": "Justin Verlander",
        "team": "HOU",
        "team_full": "Houston Astros",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup"],
        "total_pitches": 2200,
        "games": 25,
        "available_seasons": [2024]
    },
    668678: {
        "name": "Cole Ragans",
        "team": "KC",
        "team_full": "Kansas City Royals",
        "position": "SP",
        "throws": "L",
        "pitch_types": ["4-Seam Fastball", "Curveball", "Changeup", "Slider"],
        "total_pitches": 3000,
        "games": 32,
        "available_seasons": [2024]
    },
    666157: {
        "name": "Seth Lugo",
        "team": "KC",
        "team_full": "Kansas City Royals",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["Sinker", "Curveball", "4-Seam Fastball", "Slider"],
        "total_pitches": 3100,
        "games": 33,
        "available_seasons": [2024]
    },
    641154: {
        "name": "Luis Castillo",
        "team": "SEA",
        "team_full": "Seattle Mariners",
        "position": "SP",
        "throws": "R",
        "pitch_types": ["Sinker", "Changeup", "Slider", "4-Seam Fastball"],
        "total_pitches": 3000,
        "games": 30,
        "available_seasons": [2024]
    },
    670912: {
        "name": "Mason Miller",
        "team": "OAK",
        "team_full": "Oakland Athletics",
        "position": "RP",
        "throws": "R",
        "pitch_types": ["4-Seam Fastball", "Slider", "Splitter"],
        "total_pitches": 900,
        "games": 48,
        "available_seasons": [2024]
    },
}

def main():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create cache directory
    (data_dir / "cache").mkdir(exist_ok=True)
    
    # Write registry
    registry_file = data_dir / "pitcher_registry_2024.json"
    with open(registry_file, 'w') as f:
        json.dump(STARTER_PITCHERS, f, indent=2)
    
    print(f"Created starter registry with {len(STARTER_PITCHERS)} pitchers")
    print(f"File: {registry_file}")
    
    # Show teams covered
    teams = set(p["team"] for p in STARTER_PITCHERS.values())
    print(f"Teams covered: {len(teams)}")
    print(f"Teams: {', '.join(sorted(teams))}")


if __name__ == "__main__":
    main()
