#!/usr/bin/env python
"""
Initialize 2025 MLB Pitcher Data
Run this script to build the pitcher registry from live MLB data.

Usage:
    python init_2025_data.py                  # Build registry with default settings
    python init_2025_data.py --min-pitches 50 # Custom minimum pitch threshold
    python init_2025_data.py --fetch-all      # Also fetch data for all pitchers (slow)

Requirements:
    pip install pybaseball
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))


def build_registry(min_pitches: int = 100, verbose: bool = True):
    """Build the pitcher registry from 2025 MLB data."""
    try:
        from pybaseball import statcast, cache
        cache.enable()
    except ImportError:
        print("ERROR: pybaseball not installed")
        print("Install with: pip install pybaseball")
        return None
    
    # Directories
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    # Determine which season to use
    current_year = datetime.now().year
    current_month = datetime.now().month
    season_year = current_year if current_month >= 4 else current_year - 1
    
    registry_file = data_dir / f"pitcher_registry_{season_year}.json"
    
    # Season dates - Use 2024 if 2025 hasn't started yet
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # If we're before April, use previous season
    if current_month < 4:
        season_year = current_year - 1 if current_year > 2024 else 2024
    else:
        season_year = current_year
    
    season_start = f"{season_year}-03-20"
    season_end = f"{season_year}-11-01" if season_year < current_year else datetime.now().strftime("%Y-%m-%d")
    
    if verbose:
        print(f"Using season: {season_year}")
    
    # MLB Teams
    MLB_TEAMS = {
        108: {"abbr": "LAA", "name": "Los Angeles Angels"},
        109: {"abbr": "ARI", "name": "Arizona Diamondbacks"},
        110: {"abbr": "BAL", "name": "Baltimore Orioles"},
        111: {"abbr": "BOS", "name": "Boston Red Sox"},
        112: {"abbr": "CHC", "name": "Chicago Cubs"},
        113: {"abbr": "CIN", "name": "Cincinnati Reds"},
        114: {"abbr": "CLE", "name": "Cleveland Guardians"},
        115: {"abbr": "COL", "name": "Colorado Rockies"},
        116: {"abbr": "DET", "name": "Detroit Tigers"},
        117: {"abbr": "HOU", "name": "Houston Astros"},
        118: {"abbr": "KC", "name": "Kansas City Royals"},
        119: {"abbr": "LAD", "name": "Los Angeles Dodgers"},
        120: {"abbr": "WSH", "name": "Washington Nationals"},
        121: {"abbr": "NYM", "name": "New York Mets"},
        133: {"abbr": "OAK", "name": "Oakland Athletics"},
        134: {"abbr": "PIT", "name": "Pittsburgh Pirates"},
        135: {"abbr": "SD", "name": "San Diego Padres"},
        136: {"abbr": "SEA", "name": "Seattle Mariners"},
        137: {"abbr": "SF", "name": "San Francisco Giants"},
        138: {"abbr": "STL", "name": "St. Louis Cardinals"},
        139: {"abbr": "TB", "name": "Tampa Bay Rays"},
        140: {"abbr": "TEX", "name": "Texas Rangers"},
        141: {"abbr": "TOR", "name": "Toronto Blue Jays"},
        142: {"abbr": "MIN", "name": "Minnesota Twins"},
        143: {"abbr": "PHI", "name": "Philadelphia Phillies"},
        144: {"abbr": "ATL", "name": "Atlanta Braves"},
        145: {"abbr": "CWS", "name": "Chicago White Sox"},
        146: {"abbr": "MIA", "name": "Miami Marlins"},
        147: {"abbr": "NYY", "name": "New York Yankees"},
        158: {"abbr": "MIL", "name": "Milwaukee Brewers"},
    }
    TEAM_ABBR_TO_ID = {info["abbr"]: tid for tid, info in MLB_TEAMS.items()}
    
    if verbose:
        print("=" * 70)
        print("Building 2025 MLB Pitcher Registry")
        print("=" * 70)
        print(f"\nFetching data from {season_start} to {season_end}...")
        print("This may take several minutes depending on network speed.\n")
    
    try:
        # Fetch all 2025 pitches
        all_data = statcast(start_dt=season_start, end_dt=season_end)
        
        if all_data is None or len(all_data) == 0:
            print("ERROR: No data returned from statcast")
            print("The 2025 season may not have started yet, or there may be a network issue.")
            return None
        
        if verbose:
            print(f"✓ Retrieved {len(all_data):,} total pitches")
            print(f"✓ Processing unique pitchers...")
        
        # Build registry
        registry = {}
        
        for pitcher_id in all_data['pitcher'].unique():
            pitcher_data = all_data[all_data['pitcher'] == pitcher_id]
            
            if len(pitcher_data) < min_pitches:
                continue
            
            # Get info
            name = pitcher_data['player_name'].iloc[0]
            throws = pitcher_data['p_throws'].iloc[0]
            pitch_types = list(pitcher_data['pitch_name'].dropna().unique())
            
            # Determine team
            team_counts = pitcher_data['home_team'].value_counts()
            team_abbr = team_counts.index[0] if len(team_counts) > 0 else "UNK"
            
            team_id = TEAM_ABBR_TO_ID.get(team_abbr)
            team_info = MLB_TEAMS.get(team_id, {"abbr": team_abbr, "name": "Unknown"})
            
            # Determine position (SP vs RP)
            games = pitcher_data['game_date'].nunique()
            pitches_per_game = len(pitcher_data) / games if games > 0 else 0
            position = "SP" if pitches_per_game > 50 else "RP"
            
            registry[int(pitcher_id)] = {
                "name": name,
                "team": team_abbr,
                "team_full": team_info["name"],
                "position": position,
                "throws": throws,
                "pitch_types": pitch_types,
                "total_pitches": len(pitcher_data),
                "games": int(games),
                "available_seasons": [season_year]
            }
        
        # Save registry
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        if verbose:
            print(f"\n✓ Built registry with {len(registry)} pitchers")
            print(f"✓ Saved to: {registry_file}")
            
            # Show stats by team
            team_counts = {}
            for info in registry.values():
                team = info["team"]
                team_counts[team] = team_counts.get(team, 0) + 1
            
            print(f"\n✓ Teams: {len(team_counts)}")
            print("\nTop teams by pitcher count:")
            for team, count in sorted(team_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"   {team}: {count} pitchers")
            
            # Show top pitchers by pitch count
            print("\nTop pitchers by pitch count:")
            top_pitchers = sorted(
                registry.items(),
                key=lambda x: x[1]["total_pitches"],
                reverse=True
            )[:10]
            for pid, info in top_pitchers:
                print(f"   {info['name']} ({info['team']}): {info['total_pitches']:,} pitches")
        
        return registry
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_all_pitcher_data(registry: dict, verbose: bool = True):
    """Fetch and cache data for all pitchers in the registry."""
    try:
        from pybaseball import statcast_pitcher, cache
        cache.enable()
    except ImportError:
        print("ERROR: pybaseball not installed")
        return
    
    cache_dir = Path(__file__).parent / "data" / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    season_start = "2025-03-27"
    season_end = datetime.now().strftime("%Y-%m-%d")
    
    total = len(registry)
    
    if verbose:
        print(f"\nFetching data for {total} pitchers...")
    
    for i, (pid, info) in enumerate(registry.items(), 1):
        cache_file = cache_dir / f"pitcher_{pid}_2025.csv"
        
        if cache_file.exists():
            if verbose:
                print(f"  [{i}/{total}] {info['name']} - already cached")
            continue
        
        if verbose:
            print(f"  [{i}/{total}] Fetching {info['name']}...", end=" ")
        
        try:
            data = statcast_pitcher(season_start, season_end, pid)
            if data is not None and len(data) > 0:
                data.to_csv(cache_file, index=False)
                if verbose:
                    print(f"✓ {len(data)} pitches")
            else:
                if verbose:
                    print("no data")
        except Exception as e:
            if verbose:
                print(f"✗ {e}")


def main():
    parser = argparse.ArgumentParser(description="Initialize 2025 MLB pitcher data")
    parser.add_argument(
        "--min-pitches", type=int, default=100,
        help="Minimum number of pitches to include pitcher (default: 100)"
    )
    parser.add_argument(
        "--fetch-all", action="store_true",
        help="Also fetch and cache data for all pitchers"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    # Build registry
    registry = build_registry(
        min_pitches=args.min_pitches,
        verbose=not args.quiet
    )
    
    if registry is None:
        print("\nFailed to build registry. See errors above.")
        sys.exit(1)
    
    # Optionally fetch all data
    if args.fetch_all:
        fetch_all_pitcher_data(registry, verbose=not args.quiet)
    
    print("\n" + "=" * 70)
    print("Done! Start the API with:")
    print("  python run_api.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
