#!/usr/bin/env python
"""
Pre-cache ALL 2024 pitcher data for deployment.
Run this locally before deploying to Railway.

This downloads pitch-by-pitch data for every pitcher in the registry
and saves it as CSV files that will be included in the deployment.

Usage:
    python precache_all_data.py           # Cache top 100 pitchers by pitch count
    python precache_all_data.py --all     # Cache ALL pitchers (takes longer)
    python precache_all_data.py --top 50  # Cache top 50 pitchers
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import time
import sys

print("=" * 70)
print("Sequence Baseball - 2024 Data Pre-Caching")
print("=" * 70)

# Try to import pybaseball
try:
    from pybaseball import statcast, statcast_pitcher, cache
    cache.enable()
    print("✓ pybaseball loaded successfully")
except ImportError:
    print("ERROR: pybaseball not installed!")
    print("Install with: pip install pybaseball")
    sys.exit(1)

import pandas as pd

# Paths
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# MLB Teams mapping
MLB_TEAMS = {
    "LAA": "Los Angeles Angels", "ARI": "Arizona Diamondbacks",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros",
    "KC": "Kansas City Royals", "LAD": "Los Angeles Dodgers",
    "WSH": "Washington Nationals", "NYM": "New York Mets",
    "OAK": "Oakland Athletics", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "MIN": "Minnesota Twins",
    "PHI": "Philadelphia Phillies", "ATL": "Atlanta Braves",
    "CWS": "Chicago White Sox", "MIA": "Miami Marlins",
    "NYY": "New York Yankees", "MIL": "Milwaukee Brewers",
}


def build_registry_from_statcast(min_pitches: int = 100) -> dict:
    """Build pitcher registry from 2024 statcast data."""
    print("\n" + "-" * 50)
    print("Step 1: Fetching 2024 season data from Baseball Savant")
    print("-" * 50)
    print("This will take several minutes...")
    
    # Get full season data
    all_data = statcast(start_dt="2024-03-28", end_dt="2024-09-29")
    print(f"✓ Retrieved {len(all_data):,} total pitches")
    
    # Build registry
    print(f"\nBuilding registry (min {min_pitches} pitches)...")
    registry = {}
    
    # Get unique pitchers and their pitch counts
    pitcher_counts = all_data.groupby('pitcher').size().reset_index(name='count')
    qualified = pitcher_counts[pitcher_counts['count'] >= min_pitches]
    print(f"Found {len(qualified)} pitchers with {min_pitches}+ pitches")
    
    for idx, row in qualified.iterrows():
        pitcher_id = int(row['pitcher'])
        pitcher_data = all_data[all_data['pitcher'] == pitcher_id]
        
        # Get pitcher info
        name = pitcher_data['player_name'].iloc[0] if 'player_name' in pitcher_data.columns else f"Pitcher {pitcher_id}"
        throws = pitcher_data['p_throws'].iloc[0] if 'p_throws' in pitcher_data.columns else "R"
        
        # Get pitch types
        pitch_types = []
        if 'pitch_name' in pitcher_data.columns:
            pitch_types = list(pitcher_data['pitch_name'].dropna().unique())[:8]
        
        # Determine team (most common)
        team_abbr = "UNK"
        if 'home_team' in pitcher_data.columns:
            # For home games where pitcher is on home team
            home_games = pitcher_data[pitcher_data['inning_topbot'] == 'Bot']
            if len(home_games) > 0:
                team_abbr = home_games['home_team'].mode().iloc[0] if len(home_games['home_team'].mode()) > 0 else "UNK"
            else:
                team_abbr = pitcher_data['away_team'].mode().iloc[0] if 'away_team' in pitcher_data.columns else "UNK"
        
        team_full = MLB_TEAMS.get(team_abbr, team_abbr)
        
        # Determine position (SP vs RP)
        games = pitcher_data['game_pk'].nunique()
        pitches_per_game = len(pitcher_data) / games if games > 0 else 0
        position = "SP" if pitches_per_game > 50 else "RP"
        
        registry[pitcher_id] = {
            "name": name,
            "team": team_abbr,
            "team_full": team_full,
            "position": position,
            "throws": throws,
            "pitch_types": pitch_types,
            "total_pitches": int(row['count']),
            "games": int(games),
            "available_seasons": [2024]
        }
    
    # Save registry
    registry_file = DATA_DIR / "pitcher_registry_2024.json"
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✓ Saved registry with {len(registry)} pitchers")
    return registry, all_data


def cache_pitcher_data(pitcher_id: int, all_data: pd.DataFrame) -> bool:
    """Cache individual pitcher data from the full dataset."""
    try:
        pitcher_data = all_data[all_data['pitcher'] == pitcher_id].copy()
        
        if len(pitcher_data) == 0:
            return False
        
        # Save to cache
        cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_2024.csv"
        pitcher_data.to_csv(cache_file, index=False)
        return True
        
    except Exception as e:
        print(f"  Error caching {pitcher_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-cache 2024 MLB pitcher data")
    parser.add_argument("--all", action="store_true", help="Cache all pitchers (not just top N)")
    parser.add_argument("--top", type=int, default=100, help="Number of top pitchers to cache (default: 100)")
    parser.add_argument("--min-pitches", type=int, default=100, help="Minimum pitches to include (default: 100)")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Build registry
    registry, all_data = build_registry_from_statcast(min_pitches=args.min_pitches)
    
    # Determine which pitchers to cache
    if args.all:
        pitchers_to_cache = list(registry.keys())
        print(f"\n✓ Will cache ALL {len(pitchers_to_cache)} pitchers")
    else:
        # Sort by pitch count and take top N
        sorted_pitchers = sorted(registry.items(), key=lambda x: x[1]["total_pitches"], reverse=True)
        pitchers_to_cache = [p[0] for p in sorted_pitchers[:args.top]]
        print(f"\n✓ Will cache top {len(pitchers_to_cache)} pitchers by pitch count")
    
    # Cache data
    print("\n" + "-" * 50)
    print("Step 2: Caching individual pitcher data")
    print("-" * 50)
    
    cached = 0
    failed = 0
    
    for i, pitcher_id in enumerate(pitchers_to_cache, 1):
        info = registry[pitcher_id]
        
        if i % 20 == 0 or i == 1:
            print(f"\n[{i}/{len(pitchers_to_cache)}] {info['name']} ({info['team']})...")
        
        if cache_pitcher_data(pitcher_id, all_data):
            cached += 1
        else:
            failed += 1
            print(f"  ✗ Failed: {info['name']}")
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Registry: {len(registry)} pitchers")
    print(f"Cached: {cached} pitcher data files")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.1f} seconds")
    
    # Show storage size
    cache_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*.csv")) / (1024 * 1024)
    registry_size = (DATA_DIR / "pitcher_registry_2024.json").stat().st_size / 1024
    
    print(f"\nStorage:")
    print(f"  Registry: {registry_size:.1f} KB")
    print(f"  Cache: {cache_size:.1f} MB")
    
    print("\n" + "-" * 50)
    print("Next steps:")
    print("1. Commit the data files: git add data/")
    print("2. Commit changes: git commit -m 'Add 2024 pitcher data'")
    print("3. Deploy: railway up (or git push)")
    print("-" * 50)


if __name__ == "__main__":
    main()
