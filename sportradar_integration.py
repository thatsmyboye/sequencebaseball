"""
SportRadar MLB API Integration
==============================

Integration with SportRadar's MLB v8 API for player data, schedules, pitch metrics,
and game statistics.

NOTE: This does NOT include Statcast data (separate premium package).
However, it provides valuable supplementary data:
- Daily/League schedules
- Game play-by-play
- Pitch metrics (type, velocity, results)
- Player profiles and stats
- Team information
- League leaders
- Injuries and transactions

API Documentation: https://developer.sportradar.com/baseball/reference/mlb-overview

API Key valid until: January 4, 2026
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import json
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Key - Can be overridden via environment variable
SPORTRADAR_API_KEY = os.environ.get(
    "SPORTRADAR_API_KEY", 
    "MkPCdbssBXUy74XhOtt58Ed7Ov8GMrJnVGHk2YmN"
)

# Base URL for SportRadar MLB API v8
# Use 'trial' for trial keys, 'production' for production keys
SPORTRADAR_BASE_URL = "https://api.sportradar.com/mlb/trial/v8/en"

# Rate limiting (SportRadar has limits per second and per minute)
RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Cache directory for responses
CACHE_DIR = Path(__file__).parent / "data" / "sportradar_cache"


# ============================================================================
# API CLIENT CLASS
# ============================================================================

class SportRadarMLB:
    """
    Client for SportRadar MLB API v8
    
    Features:
    - Automatic rate limiting
    - Response caching
    - Error handling with retries
    - Data conversion to pandas DataFrames
    
    Usage:
    ------
    >>> client = SportRadarMLB()
    >>> schedule = client.get_daily_schedule("2024-08-15")
    >>> game_pbp = client.get_game_play_by_play(game_id)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        use_cache: bool = True,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize SportRadar MLB client
        
        Parameters:
        -----------
        api_key : str, optional
            API key (defaults to module constant or env var)
        use_cache : bool
            Whether to cache API responses locally
        cache_ttl_hours : int
            Cache time-to-live in hours
        """
        self.api_key = api_key or SPORTRADAR_API_KEY
        self.base_url = SPORTRADAR_BASE_URL
        self.use_cache = use_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.last_request_time = 0
        
        # Create cache directory
        if self.use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_path(self, endpoint: str, params: Dict) -> Path:
        """Generate cache file path for a request"""
        cache_key = f"{endpoint}_{hash(frozenset(params.items()))}"
        # Clean up the cache key for filesystem
        cache_key = cache_key.replace("/", "_").replace("?", "_")
        return CACHE_DIR / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load response from cache if valid"""
        if not self.use_cache or not cache_path.exists():
            return None
        
        # Check if cache is expired
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - modified_time > self.cache_ttl:
            return None
        
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save response to cache"""
        if self.use_cache:
            try:
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except IOError as e:
                print(f"Warning: Could not save to cache: {e}")
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        retries: int = 3
    ) -> Dict:
        """
        Make an API request with caching and retry logic
        
        Parameters:
        -----------
        endpoint : str
            API endpoint (e.g., 'games/{game_id}/pbp.json')
        params : dict, optional
            Additional query parameters
        retries : int
            Number of retry attempts
            
        Returns:
        --------
        response : dict
            JSON response from API
        """
        params = params or {}
        params['api_key'] = self.api_key
        
        # Check cache first
        cache_path = self._get_cache_path(endpoint, params)
        cached = self._load_from_cache(cache_path)
        if cached is not None:
            return cached
        
        # Rate limiting
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    self._save_to_cache(cache_path, data)
                    return data
                
                elif response.status_code == 403:
                    raise PermissionError(
                        f"API access denied. Check your API key and permissions. "
                        f"Status: {response.status_code}"
                    )
                
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 404:
                    raise ValueError(f"Resource not found: {endpoint}")
                
                else:
                    print(f"API error {response.status_code}: {response.text}")
                    if attempt < retries - 1:
                        time.sleep(1)
                        continue
                    raise Exception(f"API request failed: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise ConnectionError(f"Network error: {e}")
        
        raise Exception("Max retries exceeded")
    
    # ========================================================================
    # SCHEDULE ENDPOINTS
    # ========================================================================
    
    def get_daily_schedule(self, date: str) -> Dict:
        """
        Get schedule for a specific date
        
        Parameters:
        -----------
        date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        schedule : dict
            Games scheduled for that date
        """
        year, month, day = date.split('-')
        endpoint = f"games/{year}/{month}/{day}/schedule.json"
        return self._make_request(endpoint)
    
    def get_league_schedule(self, season: int, season_type: str = "REG") -> Dict:
        """
        Get full season schedule
        
        Parameters:
        -----------
        season : int
            Season year (e.g., 2024)
        season_type : str
            PRE (Spring Training), REG (Regular Season), PST (Postseason)
            
        Returns:
        --------
        schedule : dict
            Full season schedule
        """
        endpoint = f"games/{season}/{season_type}/schedule.json"
        return self._make_request(endpoint)
    
    # ========================================================================
    # GAME ENDPOINTS
    # ========================================================================
    
    def get_game_boxscore(self, game_id: str) -> Dict:
        """
        Get boxscore for a specific game
        
        Parameters:
        -----------
        game_id : str
            SportRadar game ID (UUID format)
            
        Returns:
        --------
        boxscore : dict
            Detailed boxscore with stats
        """
        endpoint = f"games/{game_id}/boxscore.json"
        return self._make_request(endpoint)
    
    def get_game_summary(self, game_id: str) -> Dict:
        """
        Get game summary with lineups and stats
        
        Parameters:
        -----------
        game_id : str
            SportRadar game ID
            
        Returns:
        --------
        summary : dict
            Game summary with team/player stats
        """
        endpoint = f"games/{game_id}/summary.json"
        return self._make_request(endpoint)
    
    def get_game_play_by_play(self, game_id: str) -> Dict:
        """
        Get detailed play-by-play data for a game
        
        This includes pitch-level data but NOT Statcast metrics.
        Includes: pitch type, velocity, results, count, etc.
        
        Parameters:
        -----------
        game_id : str
            SportRadar game ID
            
        Returns:
        --------
        pbp : dict
            Play-by-play with pitch data
        """
        endpoint = f"games/{game_id}/pbp.json"
        return self._make_request(endpoint)
    
    def get_game_pitch_metrics(self, game_id: str) -> Dict:
        """
        Get detailed pitch metrics for a game
        
        Includes pitch type, velocity, and results for all pitchers.
        
        Parameters:
        -----------
        game_id : str
            SportRadar game ID
            
        Returns:
        --------
        metrics : dict
            Pitch metrics for the game
        """
        endpoint = f"games/{game_id}/pitch_metrics.json"
        return self._make_request(endpoint)
    
    # ========================================================================
    # PLAYER ENDPOINTS
    # ========================================================================
    
    def get_player_profile(self, player_id: str) -> Dict:
        """
        Get player profile with career stats
        
        Parameters:
        -----------
        player_id : str
            SportRadar player ID
            
        Returns:
        --------
        profile : dict
            Player biographical info and stats
        """
        endpoint = f"players/{player_id}/profile.json"
        return self._make_request(endpoint)
    
    def get_seasonal_pitch_metrics(
        self, 
        player_id: str, 
        season: int,
        season_type: str = "REG"
    ) -> Dict:
        """
        Get seasonal pitch metrics for a pitcher
        
        Parameters:
        -----------
        player_id : str
            SportRadar player ID
        season : int
            Season year
        season_type : str
            PRE, REG, or PST
            
        Returns:
        --------
        metrics : dict
            Seasonal pitch metrics
        """
        endpoint = f"seasons/{season}/{season_type}/players/{player_id}/pitch_metrics.json"
        return self._make_request(endpoint)
    
    # ========================================================================
    # TEAM ENDPOINTS
    # ========================================================================
    
    def get_team_profile(self, team_id: str) -> Dict:
        """
        Get team profile with roster
        
        Parameters:
        -----------
        team_id : str
            SportRadar team ID
            
        Returns:
        --------
        profile : dict
            Team info and roster
        """
        endpoint = f"teams/{team_id}/profile.json"
        return self._make_request(endpoint)
    
    def get_team_seasonal_stats(
        self, 
        team_id: str, 
        season: int,
        season_type: str = "REG"
    ) -> Dict:
        """
        Get team's seasonal statistics
        
        Parameters:
        -----------
        team_id : str
            SportRadar team ID
        season : int
            Season year
        season_type : str
            PRE, REG, or PST
            
        Returns:
        --------
        stats : dict
            Team seasonal statistics
        """
        endpoint = f"seasons/{season}/{season_type}/teams/{team_id}/statistics.json"
        return self._make_request(endpoint)
    
    # ========================================================================
    # LEAGUE ENDPOINTS
    # ========================================================================
    
    def get_league_hierarchy(self) -> Dict:
        """
        Get league structure with all teams
        
        Returns division, league, and team info.
        Useful for getting team IDs.
        
        Returns:
        --------
        hierarchy : dict
            League/division/team structure
        """
        endpoint = "league/hierarchy.json"
        return self._make_request(endpoint)
    
    def get_teams(self) -> Dict:
        """
        Get all active MLB teams
        
        Returns:
        --------
        teams : dict
            All active teams
        """
        endpoint = "league/teams.json"
        return self._make_request(endpoint)
    
    def get_league_leaders(
        self, 
        season: int,
        season_type: str = "REG"
    ) -> Dict:
        """
        Get league leaders for various statistics
        
        Parameters:
        -----------
        season : int
            Season year
        season_type : str
            PRE, REG, or PST
            
        Returns:
        --------
        leaders : dict
            League leaders in hitting/pitching stats
        """
        endpoint = f"seasons/{season}/{season_type}/leaders.json"
        return self._make_request(endpoint)
    
    def get_standings(self, season: int, season_type: str = "REG") -> Dict:
        """
        Get current standings
        
        Parameters:
        -----------
        season : int
            Season year
        season_type : str
            PRE, REG, or PST
            
        Returns:
        --------
        standings : dict
            Division standings
        """
        endpoint = f"seasons/{season}/{season_type}/standings.json"
        return self._make_request(endpoint)
    
    def get_injuries(self) -> Dict:
        """
        Get current injury report
        
        Returns:
        --------
        injuries : dict
            All current injuries
        """
        endpoint = "league/injuries.json"
        return self._make_request(endpoint)
    
    def get_daily_transactions(self, date: str) -> Dict:
        """
        Get transactions for a specific date
        
        Parameters:
        -----------
        date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        transactions : dict
            Transactions for that date
        """
        year, month, day = date.split('-')
        endpoint = f"league/{year}/{month}/{day}/transactions.json"
        return self._make_request(endpoint)
    
    # ========================================================================
    # GLOSSARY
    # ========================================================================
    
    def get_glossary(self) -> Dict:
        """
        Get API glossary with pitch type codes, outcome codes, etc.
        
        Returns:
        --------
        glossary : dict
            Code definitions
        """
        endpoint = "league/glossary.json"
        return self._make_request(endpoint)


# ============================================================================
# DATA EXTRACTION HELPERS
# ============================================================================

def extract_pitches_from_pbp(pbp_data: Dict, pitcher_id: Optional[str] = None) -> pd.DataFrame:
    """
    Extract pitch-level data from play-by-play response
    
    Parameters:
    -----------
    pbp_data : dict
        Raw play-by-play response from API
    pitcher_id : str, optional
        Filter to specific pitcher
        
    Returns:
    --------
    pitches_df : pd.DataFrame
        Pitch-level data
    """
    pitches = []
    
    game = pbp_data.get('game', {})
    game_id = game.get('id', '')
    game_date = game.get('scheduled', '').split('T')[0] if game.get('scheduled') else ''
    
    for inning in pbp_data.get('game', {}).get('innings', []):
        inning_num = inning.get('number', 0)
        
        for half in ['top', 'bottom']:
            half_inning = inning.get(half, {})
            
            for at_bat in half_inning.get('at_bats', []):
                # Get pitcher info
                pitcher = at_bat.get('pitcher', {})
                current_pitcher_id = pitcher.get('id', '')
                
                if pitcher_id and current_pitcher_id != pitcher_id:
                    continue
                
                # Get batter info
                batter = at_bat.get('hitter', {})
                batter_id = batter.get('id', '')
                
                # Get at-bat context
                outs = at_bat.get('outs_at_start', 0)
                
                # Extract each pitch/event
                for event in at_bat.get('events', []):
                    if event.get('type') == 'pitch':
                        pitch_data = {
                            # Game context
                            'game_date': game_date,
                            'game_id': game_id,
                            'inning': inning_num,
                            'inning_half': half,
                            'outs_when_up': outs,
                            
                            # Player info
                            'pitcher_id': current_pitcher_id,
                            'pitcher_name': f"{pitcher.get('first_name', '')} {pitcher.get('last_name', '')}",
                            'batter_id': batter_id,
                            'batter_name': f"{batter.get('first_name', '')} {batter.get('last_name', '')}",
                            'stand': batter.get('bat_hand', ''),
                            'p_throws': pitcher.get('throw_hand', ''),
                            
                            # Count
                            'balls': event.get('count', {}).get('balls', 0),
                            'strikes': event.get('count', {}).get('strikes', 0),
                            
                            # Pitch details
                            'pitch_type': event.get('pitch_type', ''),
                            'pitch_speed': event.get('pitcher', {}).get('pitch_speed'),
                            'pitch_zone': event.get('zone'),
                            
                            # Outcome
                            'outcome_id': event.get('outcome_id', ''),
                            'description': event.get('description', ''),
                        }
                        
                        pitches.append(pitch_data)
    
    return pd.DataFrame(pitches)


def extract_pitch_metrics(metrics_data: Dict) -> pd.DataFrame:
    """
    Extract pitch metrics from game pitch metrics response
    
    Parameters:
    -----------
    metrics_data : dict
        Raw pitch metrics response
        
    Returns:
    --------
    metrics_df : pd.DataFrame
        Pitch metrics by pitcher and pitch type
    """
    metrics = []
    
    game = metrics_data.get('game', {})
    game_id = game.get('id', '')
    
    for team_key in ['home', 'away']:
        team = game.get(team_key, {})
        
        for player in team.get('players', []):
            if player.get('position') == 'P' or player.get('starter'):
                player_id = player.get('id', '')
                player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
                
                pitching = player.get('pitching', {})
                
                for pitch_type in pitching.get('pitch_types', []):
                    metric = {
                        'game_id': game_id,
                        'player_id': player_id,
                        'player_name': player_name,
                        'pitch_type': pitch_type.get('type', ''),
                        'pitch_type_desc': pitch_type.get('description', ''),
                        'count': pitch_type.get('count', 0),
                        'avg_speed': pitch_type.get('avg_speed'),
                        'max_speed': pitch_type.get('max_speed'),
                        'strikes': pitch_type.get('strikes', 0),
                        'balls': pitch_type.get('balls', 0),
                        'in_play': pitch_type.get('in_play', 0),
                        'swings': pitch_type.get('swings', 0),
                        'misses': pitch_type.get('misses', 0),
                    }
                    metrics.append(metric)
    
    return pd.DataFrame(metrics)


def find_games_by_pitcher(
    client: SportRadarMLB, 
    pitcher_name: str, 
    start_date: str, 
    end_date: str
) -> List[Dict]:
    """
    Find games where a pitcher appeared
    
    Parameters:
    -----------
    client : SportRadarMLB
        API client instance
    pitcher_name : str
        Pitcher's name to search for
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
        
    Returns:
    --------
    games : list
        List of game info dicts where pitcher appeared
    """
    from datetime import datetime, timedelta
    
    games_found = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    pitcher_name_lower = pitcher_name.lower()
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        
        try:
            schedule = client.get_daily_schedule(date_str)
            
            for game in schedule.get('games', []):
                # Get boxscore to check pitchers
                game_id = game.get('id')
                if game.get('status') == 'closed':
                    try:
                        boxscore = client.get_game_boxscore(game_id)
                        
                        # Check both teams' pitchers
                        for team_key in ['home', 'away']:
                            team = boxscore.get('game', {}).get(team_key, {})
                            for player in team.get('players', []):
                                full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
                                if pitcher_name_lower in full_name.lower():
                                    games_found.append({
                                        'game_id': game_id,
                                        'date': date_str,
                                        'home': game.get('home', {}).get('name'),
                                        'away': game.get('away', {}).get('name'),
                                        'pitcher_name': full_name
                                    })
                                    break
                    except Exception as e:
                        print(f"  Could not fetch boxscore for {game_id}: {e}")
                        
        except Exception as e:
            print(f"  Could not fetch schedule for {date_str}: {e}")
        
        current += timedelta(days=1)
    
    return games_found


def get_pitcher_season_pitches(
    client: SportRadarMLB,
    pitcher_name: str,
    season: int = 2024,
    max_games: Optional[int] = None
) -> pd.DataFrame:
    """
    Get all pitches for a pitcher in a season
    
    Parameters:
    -----------
    client : SportRadarMLB
        API client instance
    pitcher_name : str
        Pitcher's name
    season : int
        Season year
    max_games : int, optional
        Limit number of games to process
        
    Returns:
    --------
    pitches_df : pd.DataFrame
        All pitches for the season
    """
    print(f"Finding games for {pitcher_name} in {season}...")
    
    # Find games (regular season typically April-October)
    start_date = f"{season}-03-20"
    end_date = f"{season}-10-01"
    
    games = find_games_by_pitcher(client, pitcher_name, start_date, end_date)
    print(f"Found {len(games)} games")
    
    if max_games:
        games = games[:max_games]
    
    # Extract pitches from each game
    all_pitches = []
    
    for i, game in enumerate(games, 1):
        print(f"Processing game {i}/{len(games)}: {game['date']}...")
        
        try:
            pbp = client.get_game_play_by_play(game['game_id'])
            pitches = extract_pitches_from_pbp(pbp)
            
            # Filter to our pitcher
            pitcher_pitches = pitches[
                pitches['pitcher_name'].str.lower().str.contains(pitcher_name.lower())
            ]
            
            all_pitches.append(pitcher_pitches)
            print(f"  Extracted {len(pitcher_pitches)} pitches")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if all_pitches:
        combined = pd.concat(all_pitches, ignore_index=True)
        print(f"\n✓ Total pitches: {len(combined)}")
        return combined
    else:
        print("\n❌ No pitches extracted")
        return pd.DataFrame()


def convert_sportradar_to_statcast_format(sr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SportRadar pitch data to Statcast-compatible format
    
    Note: SportRadar doesn't have Statcast metrics (spin rate, movement, etc.)
    This creates placeholder values for missing fields.
    
    Parameters:
    -----------
    sr_df : pd.DataFrame
        SportRadar pitch data
        
    Returns:
    --------
    statcast_df : pd.DataFrame
        Statcast-compatible format
    """
    print("Converting SportRadar format to Statcast-compatible format...")
    
    df = sr_df.copy()
    
    # Map pitch type codes to full names
    pitch_type_map = {
        'FA': '4-Seam Fastball',
        'FF': '4-Seam Fastball',
        'FT': '2-Seam Fastball',
        'SI': 'Sinker',
        'FC': 'Cutter',
        'SL': 'Slider',
        'CU': 'Curveball',
        'CB': 'Curveball',
        'KC': 'Knuckle Curve',
        'CH': 'Changeup',
        'FS': 'Splitter',
        'SF': 'Splitter',
        'KN': 'Knuckleball',
    }
    
    # Rename columns to match Statcast format
    column_map = {
        'pitch_speed': 'release_speed',
        'pitch_zone': 'zone',
        'pitcher_id': 'pitcher',
        'batter_id': 'batter',
    }
    
    df = df.rename(columns=column_map)
    
    # Add pitch_name from pitch_type
    if 'pitch_name' not in df.columns:
        df['pitch_name'] = df['pitch_type'].map(pitch_type_map).fillna(df['pitch_type'])
    
    # Add placeholder Statcast fields (these need real Statcast data)
    placeholder_fields = {
        'release_spin_rate': None,
        'pfx_x': 0,
        'pfx_z': 0,
        'plate_x': 0,
        'plate_z': 0,
        'release_pos_x': df['p_throws'].map({'R': 2.0, 'L': -2.0}).fillna(0),
        'release_pos_y': 54.0,
        'release_pos_z': 6.0,
        'sz_top': 3.5,
        'sz_bot': 1.5,
    }
    
    for field, default in placeholder_fields.items():
        if field not in df.columns:
            df[field] = default
    
    print(f"✓ Converted {len(df)} pitches")
    
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def lookup_team_id(client: SportRadarMLB, team_name: str) -> Optional[str]:
    """
    Look up SportRadar team ID by name/abbreviation
    
    Parameters:
    -----------
    client : SportRadarMLB
        API client
    team_name : str
        Team name, city, or abbreviation
        
    Returns:
    --------
    team_id : str or None
        SportRadar team ID
    """
    hierarchy = client.get_league_hierarchy()
    team_name_lower = team_name.lower()
    
    for league in hierarchy.get('leagues', []):
        for division in league.get('divisions', []):
            for team in division.get('teams', []):
                if (team_name_lower in team.get('name', '').lower() or
                    team_name_lower in team.get('market', '').lower() or
                    team_name_lower == team.get('abbr', '').lower()):
                    return team['id']
    
    return None


def get_team_pitchers(client: SportRadarMLB, team_id: str) -> List[Dict]:
    """
    Get all pitchers on a team's roster
    
    Parameters:
    -----------
    client : SportRadarMLB
        API client
    team_id : str
        SportRadar team ID
        
    Returns:
    --------
    pitchers : list
        List of pitcher info dicts
    """
    profile = client.get_team_profile(team_id)
    pitchers = []
    
    for player in profile.get('players', []):
        if player.get('position') == 'P' or player.get('primary_position') == 'P':
            pitchers.append({
                'id': player.get('id'),
                'name': f"{player.get('first_name', '')} {player.get('last_name', '')}",
                'jersey_number': player.get('jersey_number'),
                'throw_hand': player.get('throw_hand'),
            })
    
    return pitchers


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SPORTRADAR MLB API INTEGRATION")
    print("=" * 80)
    
    # Initialize client
    client = SportRadarMLB()
    
    print("\n📖 Example 1: Test API Connection")
    print("-" * 60)
    
    try:
        glossary = client.get_glossary()
        print("✓ API connection successful!")
        print(f"  Pitch types available: {len(glossary.get('pitch_types', []))}")
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("\nNote: If using a trial key, make sure requests are going to the 'trial' endpoint.")
    
    print("\n📖 Example 2: Get Daily Schedule")
    print("-" * 60)
    print("""
# Get today's schedule
from datetime import date
today = date.today().strftime("%Y-%m-%d")
schedule = client.get_daily_schedule(today)

for game in schedule.get('games', []):
    print(f"{game['away']['name']} @ {game['home']['name']}")
    """)
    
    print("\n📖 Example 3: Get Game Pitch Metrics")
    print("-" * 60)
    print("""
# Get pitch metrics for a game
game_id = "abc123-..."  # SportRadar game ID
metrics = client.get_game_pitch_metrics(game_id)

# Convert to DataFrame
metrics_df = extract_pitch_metrics(metrics)
print(metrics_df)
    """)
    
    print("\n📖 Example 4: Get Pitcher Season Data")
    print("-" * 60)
    print("""
# Get all pitches for a pitcher in a season
pitches = get_pitcher_season_pitches(
    client,
    pitcher_name="Skubal",
    season=2024,
    max_games=10  # Limit for testing
)

# Convert to Statcast format for visualization
statcast_format = convert_sportradar_to_statcast_format(pitches)
    """)
    
    print("\n📖 Example 5: Find Team Pitchers")
    print("-" * 60)
    print("""
# Look up team ID
team_id = lookup_team_id(client, "Tigers")

# Get pitchers on roster
pitchers = get_team_pitchers(client, team_id)
for p in pitchers:
    print(f"  {p['name']} ({p['throw_hand']})")
    """)
    
    print("\n" + "=" * 80)
    print("NOTE: SportRadar does NOT include Statcast data.")
    print("For Statcast metrics (spin rate, movement, exit velocity), use pybaseball.")
    print("SportRadar is useful for: schedules, rosters, basic pitch data, injuries.")
    print("=" * 80)
