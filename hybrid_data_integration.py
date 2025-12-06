"""
Hybrid MLB Data Integration
===========================

Combines SportRadar API (schedules, rosters, standings, injuries)
with pybaseball/Baseball Savant (full Statcast pitch data).

This gives you the best of both worlds:
- SportRadar: Real-time schedules, rosters, standings, injuries, transactions
- pybaseball: Full Statcast data (spin rate, movement, exit velocity, etc.)

2025 Season Data: AVAILABLE ✓
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

# SportRadar API
SPORTRADAR_API_KEY = os.environ.get(
    "SPORTRADAR_API_KEY",
    "MkPCdbssBXUy74XhOtt58Ed7Ov8GMrJnVGHk2YmN"
)
SPORTRADAR_BASE_URL = "https://api.sportradar.com/mlb/trial/v8/en"

# Cache directory
CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SPORTRADAR CLIENT (Schedules, Rosters, Standings)
# ============================================================================

class SportRadarClient:
    """
    SportRadar MLB API client for non-Statcast data
    
    Use for:
    - Schedules (daily, season)
    - Team rosters and profiles
    - Standings and rankings
    - Injuries and transactions
    - Basic game data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or SPORTRADAR_API_KEY
        self.base_url = SPORTRADAR_BASE_URL
        self.last_request = 0
        self.rate_limit_delay = 1.0  # seconds
    
    def _request(self, endpoint: str) -> Dict:
        """Make rate-limited API request"""
        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request = time.time()
        
        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, params={'api_key': self.api_key}, timeout=30)
        
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 403:
            raise PermissionError(f"API access denied: {resp.status_code}")
        elif resp.status_code == 429:
            raise Exception("Rate limit exceeded. Wait before retrying.")
        else:
            raise Exception(f"API error {resp.status_code}: {resp.text[:200]}")
    
    # Schedule endpoints
    def get_daily_schedule(self, date: str) -> Dict:
        """Get schedule for date (YYYY-MM-DD)"""
        year, month, day = date.split('-')
        return self._request(f"games/{year}/{month}/{day}/schedule.json")
    
    def get_season_schedule(self, season: int = 2025, season_type: str = "REG") -> Dict:
        """Get full season schedule"""
        return self._request(f"games/{season}/{season_type}/schedule.json")
    
    # Team endpoints
    def get_teams(self) -> Dict:
        """Get all MLB teams"""
        return self._request("league/teams.json")
    
    def get_team_profile(self, team_id: str) -> Dict:
        """Get team roster and info"""
        return self._request(f"teams/{team_id}/profile.json")
    
    # Standings
    def get_standings(self, season: int = 2025, season_type: str = "REG") -> Dict:
        """Get league standings"""
        return self._request(f"seasons/{season}/{season_type}/standings.json")
    
    def get_leaders(self, season: int = 2025, season_type: str = "REG") -> Dict:
        """Get league leaders"""
        return self._request(f"seasons/{season}/{season_type}/leaders.json")
    
    # Injuries
    def get_injuries(self) -> Dict:
        """Get current injury report"""
        return self._request("league/injuries.json")
    
    # Game data (basic - no Statcast)
    def get_game_boxscore(self, game_id: str) -> Dict:
        """Get game boxscore"""
        return self._request(f"games/{game_id}/boxscore.json")
    
    def get_game_summary(self, game_id: str) -> Dict:
        """Get game summary"""
        return self._request(f"games/{game_id}/summary.json")


# ============================================================================
# STATCAST CLIENT (Full Pitch Data via pybaseball)
# ============================================================================

class StatcastClient:
    """
    Statcast data client via pybaseball/Baseball Savant
    
    Use for ALL detailed pitch data:
    - Spin rate, spin direction
    - Pitch movement (pfx_x, pfx_z)
    - Release point (x0, y0, z0)
    - Velocity components (vx0, vy0, vz0)
    - Acceleration (ax, ay, az)
    - Exit velocity, launch angle
    - Expected stats (xwOBA, xBA)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self._pybaseball = None
    
    def _get_pybaseball(self):
        """Lazy import pybaseball"""
        if self._pybaseball is None:
            try:
                import pybaseball
                pybaseball.cache.enable()
                self._pybaseball = pybaseball
            except ImportError:
                raise ImportError(
                    "pybaseball not installed. Install with: pip install pybaseball"
                )
        return self._pybaseball
    
    def get_pitcher_statcast(
        self,
        player_id: int,
        start_date: str = "2025-03-20",
        end_date: str = "2025-11-01"
    ) -> pd.DataFrame:
        """
        Get Statcast data for a pitcher
        
        Parameters:
        -----------
        player_id : int
            MLB player ID
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
            
        Returns:
        --------
        df : pd.DataFrame
            Full Statcast pitch data with ALL fields including:
            - spin_rate, spin_direction
            - pfx_x, pfx_z (movement)
            - release_pos_x/y/z
            - vx0, vy0, vz0 (velocity)
            - ax, ay, az (acceleration)
            - plate_x, plate_z
            - sz_top, sz_bot
            - And 80+ more fields
        """
        pb = self._get_pybaseball()
        return pb.statcast_pitcher(start_date, end_date, player_id)
    
    def get_batter_statcast(
        self,
        player_id: int,
        start_date: str = "2025-03-20",
        end_date: str = "2025-11-01"
    ) -> pd.DataFrame:
        """Get Statcast data for a batter"""
        pb = self._get_pybaseball()
        return pb.statcast_batter(start_date, end_date, player_id)
    
    def get_statcast_range(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get all Statcast data for a date range"""
        pb = self._get_pybaseball()
        return pb.statcast(start_dt=start_date, end_dt=end_date)
    
    def lookup_player(self, last_name: str, first_name: str = "") -> pd.DataFrame:
        """Look up player ID by name"""
        pb = self._get_pybaseball()
        return pb.playerid_lookup(last_name, first_name)
    
    def get_pitch_arsenal_stats(
        self,
        season: int = 2025,
        min_pitches: int = 100
    ) -> pd.DataFrame:
        """Get pitch arsenal statistics for all pitchers"""
        pb = self._get_pybaseball()
        return pb.statcast_pitcher_arsenal_stats(season, minP=min_pitches)


# ============================================================================
# HYBRID DATA SERVICE
# ============================================================================

class HybridMLBData:
    """
    Unified interface for hybrid MLB data access
    
    Automatically routes requests to the appropriate source:
    - SportRadar: Schedules, rosters, standings, injuries
    - Statcast/pybaseball: Detailed pitch and batting data
    
    Example:
    --------
    >>> mlb = HybridMLBData()
    >>> 
    >>> # Get schedule from SportRadar
    >>> schedule = mlb.get_schedule("2025-08-15")
    >>> 
    >>> # Get Statcast data for a pitcher
    >>> skubal_data = mlb.get_pitcher_statcast(669373, season=2025)
    """
    
    def __init__(self):
        self.sportradar = SportRadarClient()
        self.statcast = StatcastClient()
    
    # ========================================================================
    # SCHEDULE & ROSTER (SportRadar)
    # ========================================================================
    
    def get_schedule(self, date: str) -> List[Dict]:
        """Get games scheduled for a date"""
        data = self.sportradar.get_daily_schedule(date)
        return data.get('games', [])
    
    def get_season_schedule(self, season: int = 2025) -> List[Dict]:
        """Get full season schedule"""
        data = self.sportradar.get_season_schedule(season)
        return data.get('games', [])
    
    def get_teams(self) -> List[Dict]:
        """Get all MLB teams"""
        data = self.sportradar.get_teams()
        return data.get('teams', [])
    
    def get_team_roster(self, team_id: str) -> Dict:
        """Get team roster"""
        return self.sportradar.get_team_profile(team_id)
    
    def get_standings(self, season: int = 2025) -> Dict:
        """Get league standings"""
        return self.sportradar.get_standings(season)
    
    def get_injuries(self) -> Dict:
        """Get injury report"""
        return self.sportradar.get_injuries()
    
    # ========================================================================
    # STATCAST DATA (pybaseball)
    # ========================================================================
    
    def get_pitcher_statcast(
        self,
        player_id: int,
        season: int = 2025,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get full Statcast data for a pitcher
        
        Includes ALL Statcast fields:
        - spin_rate, spin_direction
        - pfx_x, pfx_z (movement in inches)
        - release_pos_x/y/z
        - vx0, vy0, vz0, ax, ay, az
        - plate_x, plate_z
        - And 80+ more fields
        """
        if start_date is None:
            start_date = f"{season}-03-20"
        if end_date is None:
            end_date = f"{season}-11-01"
        
        return self.statcast.get_pitcher_statcast(player_id, start_date, end_date)
    
    def get_batter_statcast(
        self,
        player_id: int,
        season: int = 2025,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get full Statcast data for a batter"""
        if start_date is None:
            start_date = f"{season}-03-20"
        if end_date is None:
            end_date = f"{season}-11-01"
        
        return self.statcast.get_batter_statcast(player_id, start_date, end_date)
    
    def lookup_player(self, last_name: str, first_name: str = "") -> pd.DataFrame:
        """Look up player MLB ID"""
        return self.statcast.lookup_player(last_name, first_name)
    
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    
    def get_pitcher_data_by_name(
        self,
        last_name: str,
        first_name: str = "",
        season: int = 2025
    ) -> pd.DataFrame:
        """
        Get Statcast data for a pitcher by name
        
        Example:
        >>> data = mlb.get_pitcher_data_by_name("skubal", "tarik", 2025)
        """
        # Look up player ID
        player = self.lookup_player(last_name, first_name)
        if len(player) == 0:
            raise ValueError(f"Player '{first_name} {last_name}' not found")
        
        player_id = int(player.iloc[0]['key_mlbam'])
        print(f"Found: {player.iloc[0]['name_first']} {player.iloc[0]['name_last']} (ID: {player_id})")
        
        return self.get_pitcher_statcast(player_id, season)
    
    def get_game_pitchers(self, game_id: str) -> Dict:
        """Get pitchers who appeared in a game (from SportRadar)"""
        boxscore = self.sportradar.get_game_boxscore(game_id)
        
        pitchers = {'home': [], 'away': []}
        game = boxscore.get('game', {})
        
        for team_key in ['home', 'away']:
            team = game.get(team_key, {})
            for player in team.get('players', []):
                if player.get('position') == 'P':
                    pitchers[team_key].append({
                        'id': player.get('id'),
                        'name': f"{player.get('first_name')} {player.get('last_name')}",
                        'jersey': player.get('jersey_number')
                    })
        
        return pitchers


# ============================================================================
# STATCAST FIELD REFERENCE
# ============================================================================

STATCAST_FIELDS = {
    # Pitch identification
    'pitch_type': 'Pitch type code (FF, SL, CU, etc.)',
    'pitch_name': 'Full pitch type name',
    
    # Velocity
    'release_speed': 'Pitch velocity at release (mph)',
    'effective_speed': 'Perceived velocity accounting for extension',
    
    # Spin
    'release_spin_rate': 'Spin rate (rpm)',
    'spin_axis': 'Spin axis angle (degrees)',
    
    # Movement
    'pfx_x': 'Horizontal movement (inches) - positive = arm side',
    'pfx_z': 'Vertical movement (inches) - positive = rise',
    
    # Break (different measurement system)
    'break_angle': 'Break angle (degrees clockwise from batter view)',
    'break_length': 'Break distance from straight line (inches)',
    'break_y': 'Distance from plate where break is greatest (feet)',
    
    # Release point
    'release_pos_x': 'Horizontal release point (feet from center)',
    'release_pos_y': 'Distance from plate at release (feet)',
    'release_pos_z': 'Vertical release point (feet)',
    'release_extension': 'Extension past rubber (feet)',
    
    # Initial velocity components
    'vx0': 'Velocity x-component at y=50ft (ft/s)',
    'vy0': 'Velocity y-component at y=50ft (ft/s)',
    'vz0': 'Velocity z-component at y=50ft (ft/s)',
    
    # Acceleration
    'ax': 'Acceleration x-component (ft/s²)',
    'ay': 'Acceleration y-component (ft/s²)',
    'az': 'Acceleration z-component (ft/s²)',
    
    # Plate location
    'plate_x': 'Horizontal position at plate (feet from center)',
    'plate_z': 'Vertical position at plate (feet)',
    
    # Strike zone
    'sz_top': 'Top of strike zone (feet)',
    'sz_bot': 'Bottom of strike zone (feet)',
    'zone': 'Zone number (1-14, with 11-14 outside)',
    
    # Timing
    'plate_time': 'Time from release to plate (seconds)',
    
    # Batted ball (when applicable)
    'launch_speed': 'Exit velocity (mph)',
    'launch_angle': 'Launch angle (degrees)',
    'hit_distance_sc': 'Projected hit distance (feet)',
    'estimated_ba_using_speedangle': 'Expected batting average (xBA)',
    'estimated_woba_using_speedangle': 'Expected wOBA (xwOBA)',
}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID MLB DATA INTEGRATION")
    print("SportRadar + pybaseball/Statcast")
    print("=" * 70)
    
    print("\n📖 Example 1: Initialize the hybrid client")
    print("-" * 50)
    print("""
from hybrid_data_integration import HybridMLBData

mlb = HybridMLBData()
    """)
    
    print("\n📖 Example 2: Get 2025 schedule from SportRadar")
    print("-" * 50)
    print("""
# Get games for a specific date
games = mlb.get_schedule("2025-08-15")
for game in games[:5]:
    print(f"{game['away']['name']} @ {game['home']['name']}")

# Get full season schedule
season = mlb.get_season_schedule(2025)
print(f"Total 2025 games: {len(season)}")
    """)
    
    print("\n📖 Example 3: Get full Statcast data for Tarik Skubal (2025)")
    print("-" * 50)
    print("""
# By player ID (faster)
skubal_data = mlb.get_pitcher_statcast(669373, season=2025)

# Or by name
skubal_data = mlb.get_pitcher_data_by_name("skubal", "tarik", 2025)

# Data includes ALL Statcast fields:
print(f"Pitches: {len(skubal_data)}")
print(f"Columns: {len(skubal_data.columns)}")
print("Sample columns:", list(skubal_data.columns)[:15])

# Access specific metrics:
print(f"Avg Fastball Spin: {skubal_data[skubal_data['pitch_type']=='FF']['release_spin_rate'].mean():.0f} rpm")
print(f"Avg Fastball velo: {skubal_data[skubal_data['pitch_type']=='FF']['release_speed'].mean():.1f} mph")
    """)
    
    print("\n📖 Example 4: Get standings and injuries from SportRadar")
    print("-" * 50)
    print("""
# Current standings
standings = mlb.get_standings(2025)

# Current injuries
injuries = mlb.get_injuries()
    """)
    
    print("\n📖 Example 5: Available Statcast fields for your analysis")
    print("-" * 50)
    print("""
The data points you listed are ALL available via pybaseball:

✅ break_angle      - pfx movement converted to break angle
✅ break_length     - Total break distance  
✅ break_y          - Distance where break peaks
✅ spin_rate        - release_spin_rate
✅ spin_direction   - spin_axis
✅ pfx_x, pfx_z     - Horizontal/vertical movement
✅ p_x, p_z         - plate_x, plate_z (plate crossing)
✅ x0, y0, z0       - release_pos_x/y/z
✅ vx0, vy0, vz0    - Initial velocity components
✅ ax, ay, az       - Acceleration components
✅ start_speed      - release_speed
✅ end_speed        - Can calculate from plate_time
✅ extension        - release_extension
✅ plate_time       - Time to plate
✅ strike_zone      - sz_top, sz_bot
    """)
    
    print("\n" + "=" * 70)
    print("HYBRID APPROACH SUMMARY")
    print("=" * 70)
    print("""
┌─────────────────┬────────────────────────────────────────┐
│ Data Type       │ Source                                 │
├─────────────────┼────────────────────────────────────────┤
│ Schedules       │ SportRadar API                         │
│ Team Rosters    │ SportRadar API                         │
│ Standings       │ SportRadar API                         │
│ Injuries        │ SportRadar API                         │
│ Transactions    │ SportRadar API                         │
├─────────────────┼────────────────────────────────────────┤
│ Pitch Statcast  │ pybaseball (Baseball Savant)           │
│ Spin/Movement   │ pybaseball                             │
│ Release Point   │ pybaseball                             │
│ Exit Velocity   │ pybaseball                             │
│ Expected Stats  │ pybaseball                             │
└─────────────────┴────────────────────────────────────────┘

2025 DATA: ✅ AVAILABLE FROM BOTH SOURCES
    """)
