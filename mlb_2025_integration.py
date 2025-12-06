"""
MLB 2025 Season Data Integration
Fetches all pitcher data from the 2025 MLB season using pybaseball
Supports dynamic loading, caching, and comprehensive pitcher registry
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "cache"
REGISTRY_FILE = DATA_DIR / "pitcher_registry_2025.json"

# 2025 Season dates (adjust as needed)
SEASON_2025_START = "2025-03-27"  # Opening Day 2025
SEASON_2025_END = datetime.now().strftime("%Y-%m-%d")  # Up to today

# MLB Team mappings
MLB_TEAMS = {
    108: {"abbr": "LAA", "name": "Los Angeles Angels", "league": "AL", "division": "West"},
    109: {"abbr": "ARI", "name": "Arizona Diamondbacks", "league": "NL", "division": "West"},
    110: {"abbr": "BAL", "name": "Baltimore Orioles", "league": "AL", "division": "East"},
    111: {"abbr": "BOS", "name": "Boston Red Sox", "league": "AL", "division": "East"},
    112: {"abbr": "CHC", "name": "Chicago Cubs", "league": "NL", "division": "Central"},
    113: {"abbr": "CIN", "name": "Cincinnati Reds", "league": "NL", "division": "Central"},
    114: {"abbr": "CLE", "name": "Cleveland Guardians", "league": "AL", "division": "Central"},
    115: {"abbr": "COL", "name": "Colorado Rockies", "league": "NL", "division": "West"},
    116: {"abbr": "DET", "name": "Detroit Tigers", "league": "AL", "division": "Central"},
    117: {"abbr": "HOU", "name": "Houston Astros", "league": "AL", "division": "West"},
    118: {"abbr": "KC", "name": "Kansas City Royals", "league": "AL", "division": "Central"},
    119: {"abbr": "LAD", "name": "Los Angeles Dodgers", "league": "NL", "division": "West"},
    120: {"abbr": "WSH", "name": "Washington Nationals", "league": "NL", "division": "East"},
    121: {"abbr": "NYM", "name": "New York Mets", "league": "NL", "division": "East"},
    133: {"abbr": "OAK", "name": "Oakland Athletics", "league": "AL", "division": "West"},
    134: {"abbr": "PIT", "name": "Pittsburgh Pirates", "league": "NL", "division": "Central"},
    135: {"abbr": "SD", "name": "San Diego Padres", "league": "NL", "division": "West"},
    136: {"abbr": "SEA", "name": "Seattle Mariners", "league": "AL", "division": "West"},
    137: {"abbr": "SF", "name": "San Francisco Giants", "league": "NL", "division": "West"},
    138: {"abbr": "STL", "name": "St. Louis Cardinals", "league": "NL", "division": "Central"},
    139: {"abbr": "TB", "name": "Tampa Bay Rays", "league": "AL", "division": "East"},
    140: {"abbr": "TEX", "name": "Texas Rangers", "league": "AL", "division": "West"},
    141: {"abbr": "TOR", "name": "Toronto Blue Jays", "league": "AL", "division": "East"},
    142: {"abbr": "MIN", "name": "Minnesota Twins", "league": "AL", "division": "Central"},
    143: {"abbr": "PHI", "name": "Philadelphia Phillies", "league": "NL", "division": "East"},
    144: {"abbr": "ATL", "name": "Atlanta Braves", "league": "NL", "division": "East"},
    145: {"abbr": "CWS", "name": "Chicago White Sox", "league": "AL", "division": "Central"},
    146: {"abbr": "MIA", "name": "Miami Marlins", "league": "NL", "division": "East"},
    147: {"abbr": "NYY", "name": "New York Yankees", "league": "AL", "division": "East"},
    158: {"abbr": "MIL", "name": "Milwaukee Brewers", "league": "NL", "division": "Central"},
}

# Team abbreviation to ID mapping
TEAM_ABBR_TO_ID = {info["abbr"]: team_id for team_id, info in MLB_TEAMS.items()}


def ensure_directories():
    """Create necessary directories if they don't exist"""
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


def get_all_2025_pitchers() -> pd.DataFrame:
    """
    Fetch all pitchers who threw pitches in the 2025 MLB season.
    Uses pybaseball's statcast function to get unique pitchers.
    """
    try:
        from pybaseball import statcast
        
        logger.info(f"Fetching 2025 season data from {SEASON_2025_START} to {SEASON_2025_END}...")
        
        # Fetch all pitches for the season (this may take a while)
        # We'll get a sample first to extract unique pitchers
        all_data = statcast(start_dt=SEASON_2025_START, end_dt=SEASON_2025_END)
        
        if all_data is None or len(all_data) == 0:
            logger.warning("No data returned from statcast")
            return pd.DataFrame()
        
        logger.info(f"Retrieved {len(all_data)} total pitches")
        
        # Get unique pitchers with their info
        pitcher_data = all_data.groupby('pitcher').agg({
            'player_name': 'first',  # Pitcher name
            'p_throws': 'first',     # Throwing hand
            'pitch_name': lambda x: list(x.dropna().unique()),  # Pitch types
            'game_date': ['min', 'max', 'count'],  # Date range and pitch count
            'home_team': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,  # Most common team
        }).reset_index()
        
        # Flatten column names
        pitcher_data.columns = ['pitcher_id', 'name', 'throws', 'pitch_types', 
                                'first_game', 'last_game', 'total_pitches', 'team']
        
        return pitcher_data
        
    except ImportError:
        logger.error("pybaseball not installed. Install with: pip install pybaseball")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching 2025 pitchers: {e}")
        return pd.DataFrame()


def fetch_pitcher_statcast_data(pitcher_id: int, 
                                 start_date: str = SEASON_2025_START,
                                 end_date: str = None) -> pd.DataFrame:
    """
    Fetch Statcast data for a specific pitcher.
    
    Args:
        pitcher_id: MLB player ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
    
    Returns:
        DataFrame with pitcher's Statcast data
    """
    try:
        from pybaseball import statcast_pitcher
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching data for pitcher {pitcher_id} from {start_date} to {end_date}")
        
        data = statcast_pitcher(start_date, end_date, pitcher_id)
        
        if data is not None and len(data) > 0:
            # Add season column
            data['game_date'] = pd.to_datetime(data['game_date'])
            data['season'] = data['game_date'].dt.year
            logger.info(f"Retrieved {len(data)} pitches for pitcher {pitcher_id}")
        
        return data
        
    except ImportError:
        logger.error("pybaseball not installed. Install with: pip install pybaseball")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching pitcher {pitcher_id}: {e}")
        return pd.DataFrame()


def build_pitcher_registry(min_pitches: int = 100) -> Dict[int, Dict]:
    """
    Build a comprehensive registry of all 2025 MLB pitchers.
    
    Args:
        min_pitches: Minimum number of pitches to include pitcher
    
    Returns:
        Dictionary mapping pitcher_id to pitcher info
    """
    ensure_directories()
    
    # Check for cached registry
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
            logger.info(f"Loaded cached registry with {len(registry)} pitchers")
            return {int(k): v for k, v in registry.items()}
    
    logger.info("Building pitcher registry from scratch...")
    
    try:
        from pybaseball import statcast
        
        # Fetch all 2025 season data
        logger.info("Fetching complete 2025 season data (this may take several minutes)...")
        all_data = statcast(start_dt=SEASON_2025_START, end_dt=SEASON_2025_END)
        
        if all_data is None or len(all_data) == 0:
            logger.error("No data available for 2025 season")
            return {}
        
        logger.info(f"Processing {len(all_data)} total pitches...")
        
        # Process unique pitchers
        registry = {}
        
        for pitcher_id in all_data['pitcher'].unique():
            pitcher_data = all_data[all_data['pitcher'] == pitcher_id]
            
            if len(pitcher_data) < min_pitches:
                continue
            
            # Get pitcher info
            name = pitcher_data['player_name'].iloc[0]
            throws = pitcher_data['p_throws'].iloc[0]
            pitch_types = pitcher_data['pitch_name'].dropna().unique().tolist()
            
            # Determine team (most common team they pitched for)
            team_counts = pitcher_data['home_team'].value_counts()
            if len(team_counts) > 0:
                team_abbr = team_counts.index[0]
            else:
                team_abbr = "UNK"
            
            # Get team full name
            team_id = TEAM_ABBR_TO_ID.get(team_abbr)
            team_info = MLB_TEAMS.get(team_id, {"name": "Unknown", "abbr": team_abbr})
            
            # Determine position based on pitch count patterns
            # SP typically throws more pitches per game but fewer games
            # RP throws fewer pitches but more appearances
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
                "games": games,
                "available_seasons": [2025]
            }
        
        # Save registry to cache
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Built registry with {len(registry)} pitchers")
        return registry
        
    except ImportError:
        logger.error("pybaseball not installed. Install with: pip install pybaseball")
        return {}
    except Exception as e:
        logger.error(f"Error building registry: {e}")
        return {}


def get_cached_pitcher_data(pitcher_id: int) -> Optional[pd.DataFrame]:
    """
    Get pitcher data from cache if available.
    """
    cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_2025.csv"
    
    if cache_file.exists():
        # Check if cache is recent (within 1 day)
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=1):
            logger.info(f"Loading cached data for pitcher {pitcher_id}")
            return pd.read_csv(cache_file)
    
    return None


def cache_pitcher_data(pitcher_id: int, data: pd.DataFrame):
    """
    Cache pitcher data to disk.
    """
    ensure_directories()
    cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_2025.csv"
    data.to_csv(cache_file, index=False)
    logger.info(f"Cached data for pitcher {pitcher_id}")


def get_pitcher_data(pitcher_id: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Get pitcher data, using cache if available, otherwise fetch from API.
    """
    if use_cache:
        cached = get_cached_pitcher_data(pitcher_id)
        if cached is not None:
            return cached
    
    # Fetch fresh data
    data = fetch_pitcher_statcast_data(pitcher_id)
    
    if data is not None and len(data) > 0:
        cache_pitcher_data(pitcher_id, data)
    
    return data


def search_pitchers(query: str, registry: Dict[int, Dict]) -> List[Dict]:
    """
    Search pitchers by name or team.
    Returns alphabetized list of matching pitchers.
    
    Args:
        query: Search query (name or team)
        registry: Pitcher registry dictionary
    
    Returns:
        List of matching pitcher dictionaries, alphabetized by name
    """
    query_lower = query.lower().strip()
    
    if not query_lower:
        # Return all pitchers alphabetized
        results = [{"id": pid, **info} for pid, info in registry.items()]
        return sorted(results, key=lambda x: x["name"].lower())
    
    matches = []
    
    for pid, info in registry.items():
        # Match by name
        if query_lower in info["name"].lower():
            matches.append({"id": pid, **info, "match_type": "name"})
            continue
        
        # Match by team abbreviation
        if query_lower == info["team"].lower():
            matches.append({"id": pid, **info, "match_type": "team"})
            continue
        
        # Match by full team name
        if query_lower in info["team_full"].lower():
            matches.append({"id": pid, **info, "match_type": "team"})
            continue
    
    # Sort alphabetically by name
    return sorted(matches, key=lambda x: x["name"].lower())


def get_team_pitchers(team: str, registry: Dict[int, Dict]) -> List[Dict]:
    """
    Get all pitchers for a specific team.
    
    Args:
        team: Team abbreviation or full name
        registry: Pitcher registry dictionary
    
    Returns:
        List of pitcher dictionaries for the team, alphabetized
    """
    team_lower = team.lower().strip()
    
    matches = []
    for pid, info in registry.items():
        if (team_lower == info["team"].lower() or 
            team_lower in info["team_full"].lower()):
            matches.append({"id": pid, **info})
    
    return sorted(matches, key=lambda x: x["name"].lower())


# For standalone testing
if __name__ == "__main__":
    print("=" * 70)
    print("MLB 2025 Data Integration Test")
    print("=" * 70)
    
    # Test building registry
    print("\n1. Building pitcher registry...")
    registry = build_pitcher_registry(min_pitches=100)
    
    if registry:
        print(f"   Found {len(registry)} pitchers with 100+ pitches")
        
        # Show top 10 by pitch count
        sorted_pitchers = sorted(
            registry.items(), 
            key=lambda x: x[1]["total_pitches"], 
            reverse=True
        )[:10]
        
        print("\n   Top 10 pitchers by pitch count:")
        for pid, info in sorted_pitchers:
            print(f"   - {info['name']} ({info['team']}): {info['total_pitches']} pitches")
        
        # Test search
        print("\n2. Testing search...")
        results = search_pitchers("skubal", registry)
        print(f"   Search 'skubal': {len(results)} results")
        for r in results[:5]:
            print(f"   - {r['name']} ({r['team']})")
        
        # Test team search
        print("\n3. Testing team search...")
        results = search_pitchers("Yankees", registry)
        print(f"   Search 'Yankees': {len(results)} results")
        for r in results[:5]:
            print(f"   - {r['name']} ({r['team']})")
    else:
        print("   No registry data available. Check network connection.")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
