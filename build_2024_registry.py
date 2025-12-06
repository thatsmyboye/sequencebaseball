#!/usr/bin/env python
"""
Build complete 2024 MLB pitcher registry from Baseball Savant
"""
import json
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("Building 2024 MLB Pitcher Registry")
print("=" * 60)

try:
    from pybaseball import statcast, cache
    cache.enable()
    print("✓ pybaseball loaded")
except ImportError:
    print("ERROR: pybaseball not installed")
    print("Run: pip install pybaseball")
    exit(1)

# Directories
data_dir = Path(__file__).parent / "data"
cache_dir = data_dir / "cache"
data_dir.mkdir(exist_ok=True)
cache_dir.mkdir(exist_ok=True)

# MLB Teams
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

print("\nFetching 2024 season data from Baseball Savant...")
print("This will take several minutes...\n")

# Fetch 2024 season data
try:
    all_data = statcast(start_dt="2024-03-20", end_dt="2024-10-01")
    print(f"✓ Retrieved {len(all_data):,} total pitches")
except Exception as e:
    print(f"ERROR fetching data: {e}")
    exit(1)

# Build registry
print("\nProcessing pitchers...")
registry = {}
min_pitches = 100

unique_pitchers = all_data['pitcher'].unique()
print(f"Found {len(unique_pitchers)} unique pitcher IDs")

for i, pitcher_id in enumerate(unique_pitchers):
    if i % 100 == 0:
        print(f"  Processing {i}/{len(unique_pitchers)}...")
    
    pitcher_data = all_data[all_data['pitcher'] == pitcher_id]
    
    if len(pitcher_data) < min_pitches:
        continue
    
    # Get info
    name = pitcher_data['player_name'].iloc[0] if 'player_name' in pitcher_data.columns else f"Pitcher {pitcher_id}"
    throws = pitcher_data['p_throws'].iloc[0] if 'p_throws' in pitcher_data.columns else "R"
    pitch_types = list(pitcher_data['pitch_name'].dropna().unique())
    
    # Determine team
    if 'home_team' in pitcher_data.columns:
        team_counts = pitcher_data['home_team'].value_counts()
        team_abbr = team_counts.index[0] if len(team_counts) > 0 else "UNK"
    else:
        team_abbr = "UNK"
    
    team_full = MLB_TEAMS.get(team_abbr, "Unknown")
    
    # Determine position
    games = pitcher_data['game_date'].nunique()
    pitches_per_game = len(pitcher_data) / games if games > 0 else 0
    position = "SP" if pitches_per_game > 50 else "RP"
    
    registry[int(pitcher_id)] = {
        "name": name,
        "team": team_abbr,
        "team_full": team_full,
        "position": position,
        "throws": throws,
        "pitch_types": pitch_types[:8],  # Limit to 8 pitch types
        "total_pitches": len(pitcher_data),
        "games": int(games),
        "available_seasons": [2024]
    }

# Save registry
registry_file = data_dir / "pitcher_registry_2024.json"
with open(registry_file, 'w') as f:
    json.dump(registry, f, indent=2)

print(f"\n✓ Created registry with {len(registry)} pitchers")
print(f"✓ Saved to: {registry_file}")

# Show stats
teams = {}
for info in registry.values():
    team = info["team"]
    teams[team] = teams.get(team, 0) + 1

print(f"\nTeams: {len(teams)}")
print("\nTop 10 teams by pitcher count:")
for team, count in sorted(teams.items(), key=lambda x: -x[1])[:10]:
    print(f"  {team}: {count}")

print("\nTop 10 pitchers by pitch count:")
top = sorted(registry.items(), key=lambda x: x[1]["total_pitches"], reverse=True)[:10]
for pid, info in top:
    print(f"  {info['name']} ({info['team']}): {info['total_pitches']:,} pitches")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
