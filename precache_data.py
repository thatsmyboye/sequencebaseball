#!/usr/bin/env python
"""
Pre-cache pitcher data for Railway deployment

This script fetches and caches data for specified pitchers locally.
The cached data is then included in the Docker image for Railway,
eliminating the need for pybaseball at runtime.

Usage:
    python precache_data.py                    # Cache top 50 pitchers
    python precache_data.py --top 100          # Cache top 100 pitchers
    python precache_data.py --pitcher 669373   # Cache specific pitcher
    python precache_data.py --team NYY         # Cache all Yankees pitchers
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Pre-cache pitcher data for deployment")
    parser.add_argument("--top", type=int, default=50, help="Cache top N pitchers by pitch count")
    parser.add_argument("--pitcher", type=int, help="Cache specific pitcher ID")
    parser.add_argument("--team", type=str, help="Cache all pitchers for a team")
    parser.add_argument("--all", action="store_true", help="Cache ALL pitchers (slow)")
    
    args = parser.parse_args()
    
    # Setup
    data_dir = Path(__file__).parent / "data"
    cache_dir = data_dir / "cache"
    registry_file = data_dir / "pitcher_registry_2025.json"
    
    cache_dir.mkdir(exist_ok=True)
    
    # Load registry
    if not registry_file.exists():
        print("ERROR: No pitcher registry found!")
        print("Run: python init_2025_data.py first")
        sys.exit(1)
    
    with open(registry_file) as f:
        registry = {int(k): v for k, v in json.load(f).items()}
    
    print(f"Loaded registry with {len(registry)} pitchers")
    
    # Determine which pitchers to cache
    if args.pitcher:
        if args.pitcher not in registry:
            print(f"ERROR: Pitcher {args.pitcher} not in registry")
            sys.exit(1)
        pitcher_ids = [args.pitcher]
    elif args.team:
        team = args.team.upper()
        pitcher_ids = [pid for pid, info in registry.items() if info["team"] == team]
        if not pitcher_ids:
            print(f"ERROR: No pitchers found for team {args.team}")
            sys.exit(1)
    elif args.all:
        pitcher_ids = list(registry.keys())
    else:
        # Top N by pitch count
        sorted_pitchers = sorted(
            registry.items(),
            key=lambda x: x[1].get("total_pitches", 0),
            reverse=True
        )
        pitcher_ids = [pid for pid, _ in sorted_pitchers[:args.top]]
    
    print(f"Caching data for {len(pitcher_ids)} pitchers...")
    
    # Import pybaseball
    try:
        from pybaseball import statcast_pitcher, cache
        cache.enable()
    except ImportError:
        print("ERROR: pybaseball not installed")
        print("Run: pip install pybaseball")
        sys.exit(1)
    
    # Cache each pitcher
    season_start = "2025-03-27"
    season_end = datetime.now().strftime("%Y-%m-%d")
    
    success = 0
    failed = 0
    skipped = 0
    
    for i, pid in enumerate(pitcher_ids, 1):
        info = registry[pid]
        cache_file = cache_dir / f"pitcher_{pid}_2025.csv"
        
        if cache_file.exists():
            print(f"[{i}/{len(pitcher_ids)}] {info['name']} - already cached")
            skipped += 1
            continue
        
        print(f"[{i}/{len(pitcher_ids)}] Fetching {info['name']} ({info['team']})...", end=" ")
        
        try:
            data = statcast_pitcher(season_start, season_end, pid)
            
            if data is not None and len(data) > 0:
                data.to_csv(cache_file, index=False)
                print(f"✓ {len(data)} pitches")
                success += 1
            else:
                print("no data")
                failed += 1
        except Exception as e:
            print(f"✗ {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"DONE: {success} cached, {skipped} skipped, {failed} failed")
    print("=" * 50)
    
    # Show cache size
    cache_size = sum(f.stat().st_size for f in cache_dir.glob("*.csv")) / (1024 * 1024)
    print(f"Cache size: {cache_size:.1f} MB")
    print(f"Cache location: {cache_dir}")


if __name__ == "__main__":
    main()
