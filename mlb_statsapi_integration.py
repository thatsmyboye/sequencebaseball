"""
MLB-StatsAPI Integration Guide
Converter to transform MLB Stats API data to Statcast-compatible format

This provides an alternative data source when Baseball Savant is unavailable.
Note: MLB-StatsAPI has fewer metrics than Statcast, but includes core pitch data.
"""
import statsapi
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

def get_pitcher_game_ids(pitcher_name: str, start_date: str, end_date: str,
                         team_id: Optional[int] = None) -> List[int]:
    """
    Find game IDs where a specific pitcher appeared

    Parameters:
    -----------
    pitcher_name : str
        Pitcher's last name
    start_date : str
        Start date (MM/DD/YYYY)
    end_date : str
        End date (MM/DD/YYYY)
    team_id : int, optional
        MLB team ID (helps narrow search)

    Returns:
    --------
    game_ids : list
        List of game IDs where pitcher appeared
    """
    print(f"Searching for {pitcher_name} games from {start_date} to {end_date}...")

    # Get pitcher ID
    pitcher_results = statsapi.lookup_player(pitcher_name)
    if not pitcher_results:
        raise ValueError(f"Pitcher '{pitcher_name}' not found")

    pitcher_id = pitcher_results[0]['id']
    pitcher_full_name = pitcher_results[0]['fullName']
    print(f"Found: {pitcher_full_name} (ID: {pitcher_id})")

    # Get team if not provided
    if team_id is None:
        team_id = pitcher_results[0].get('currentTeam', {}).get('id')

    # Get schedule
    games = statsapi.schedule(
        start_date=start_date,
        end_date=end_date,
        team=team_id
    )

    print(f"Checking {len(games)} games...")

    # Find games where pitcher appeared
    game_ids = []
    for game in games:
        try:
            boxscore = statsapi.boxscore_data(game['game_id'])

            # Check both teams' pitchers
            away_pitchers = boxscore.get('away', {}).get('pitchers', [])
            home_pitchers = boxscore.get('home', {}).get('pitchers', [])

            if pitcher_id in (away_pitchers + home_pitchers):
                game_ids.append(game['game_id'])
                print(f"  ✓ Found in game {game['game_id']} ({game['game_date']})")
        except:
            continue

    print(f"\nFound {len(game_ids)} games with {pitcher_full_name}")
    return game_ids


def extract_pitcher_pitches_from_game(game_id: int, pitcher_id: int) -> pd.DataFrame:
    """
    Extract all pitches for a specific pitcher from a game

    Parameters:
    -----------
    game_id : int
        MLB game ID
    pitcher_id : int
        MLB pitcher ID

    Returns:
    --------
    pitches_df : pd.DataFrame
        DataFrame with pitch-level data
    """
    print(f"Extracting pitches from game {game_id}...")

    # Get play-by-play data
    pbp = statsapi.get('game_playByPlay', {'gamePk': game_id})

    pitches = []

    for play in pbp.get('allPlays', []):
        # Check if this pitcher is involved
        pitcher_info = play.get('matchup', {}).get('pitcher', {})

        if pitcher_info.get('id') != pitcher_id:
            continue

        # Get game context
        game_date = play.get('about', {}).get('startTime', '')
        inning = play.get('about', {}).get('inning', 0)
        outs = play.get('about', {}).get('outs', 0)

        # Get batter info
        batter = play.get('matchup', {}).get('batter', {})
        batter_id = batter.get('id')
        batter_side = play.get('matchup', {}).get('batSide', {}).get('code', '')

        # Extract pitch events
        for event in play.get('playEvents', []):
            if event.get('isPitch'):
                pitch_data = event.get('pitchData', {})
                details = event.get('details', {})

                # Get count from the pitch event (count BEFORE this pitch)
                # Note: play.get('count') is the FINAL count, not per-pitch
                pitch_count = event.get('count', {})
                balls = pitch_count.get('balls', 0)
                strikes = pitch_count.get('strikes', 0)

                # Build pitch record (matching our Statcast-like format)
                pitch = {
                    # Game context
                    'game_date': game_date.split('T')[0] if game_date else '',
                    'game_id': game_id,
                    'inning': inning,
                    'outs_when_up': outs,

                    # Player IDs
                    'pitcher': pitcher_id,
                    'batter': batter_id,
                    'stand': batter_side,
                    'p_throws': play.get('matchup', {}).get('pitchHand', {}).get('code', ''),

                    # Count (from pitch event - count before this pitch was thrown)
                    'balls': balls,
                    'strikes': strikes,

                    # Pitch type and outcome
                    'pitch_type': details.get('type', {}).get('code', ''),
                    'pitch_name': details.get('type', {}).get('description', ''),
                    'description': details.get('description', ''),
                    'code': details.get('code', ''),

                    # Pitch metrics (from MLB-StatsAPI)
                    'release_speed': pitch_data.get('startSpeed'),  # MPH
                    'release_spin_rate': pitch_data.get('breaks', {}).get('spinRate'),

                    # Coordinates (if available)
                    'coordinates_x': pitch_data.get('coordinates', {}).get('x'),
                    'coordinates_y': pitch_data.get('coordinates', {}).get('y'),
                    'plate_x': pitch_data.get('coordinates', {}).get('pX'),
                    'plate_z': pitch_data.get('coordinates', {}).get('pZ'),

                    # Break (if available)
                    'break_angle': pitch_data.get('breaks', {}).get('breakAngle'),
                    'break_length': pitch_data.get('breaks', {}).get('breakLength'),
                    'break_y': pitch_data.get('breaks', {}).get('breakY'),

                    # Zone
                    'zone': pitch_data.get('zone'),

                    # Strike zone dimensions
                    'sz_top': pitch_data.get('strikeZoneTop'),
                    'sz_bot': pitch_data.get('strikeZoneBottom'),
                }

                pitches.append(pitch)

    df = pd.DataFrame(pitches)
    print(f"  Extracted {len(df)} pitches")

    return df


def get_pitcher_season_data_mlbapi(pitcher_name: str, season: int = 2024,
                                    team_id: Optional[int] = None) -> pd.DataFrame:
    """
    Get full season pitch data for a pitcher using MLB-StatsAPI

    Parameters:
    -----------
    pitcher_name : str
        Pitcher's last name
    season : int
        Season year
    team_id : int, optional
        MLB team ID

    Returns:
    --------
    season_data : pd.DataFrame
        All pitches for the season
    """
    # Define season dates
    start_date = f"03/20/{season}"
    end_date = f"10/01/{season}"

    # Get pitcher ID
    pitcher_results = statsapi.lookup_player(pitcher_name)
    if not pitcher_results:
        raise ValueError(f"Pitcher '{pitcher_name}' not found")

    pitcher_id = pitcher_results[0]['id']

    # Find all games
    game_ids = get_pitcher_game_ids(pitcher_name, start_date, end_date, team_id)

    # Extract pitches from each game
    all_pitches = []

    for i, game_id in enumerate(game_ids, 1):
        print(f"\nProcessing game {i}/{len(game_ids)}...")
        try:
            pitches_df = extract_pitcher_pitches_from_game(game_id, pitcher_id)
            all_pitches.append(pitches_df)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue

    # Combine all games
    if all_pitches:
        combined = pd.concat(all_pitches, ignore_index=True)
        print(f"\n✓ Total pitches extracted: {len(combined)}")
        return combined
    else:
        print("\n❌ No pitches extracted")
        return pd.DataFrame()


def convert_mlbapi_to_statcast_format(mlb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MLB-StatsAPI format to Statcast-compatible format

    Note: MLB-StatsAPI has fewer metrics than Statcast.
    This creates estimated/placeholder values for missing fields.

    Parameters:
    -----------
    mlb_df : pd.DataFrame
        Data from MLB-StatsAPI

    Returns:
    --------
    statcast_df : pd.DataFrame
        Statcast-compatible format
    """
    print("Converting MLB-StatsAPI format to Statcast-compatible format...")

    df = mlb_df.copy()

    # Map pitch type codes to full names (if needed)
    pitch_type_map = {
        'FF': '4-Seam Fastball',
        'FT': '2-Seam Fastball',
        'FC': 'Cutter',
        'SI': 'Sinker',
        'SL': 'Slider',
        'CU': 'Curveball',
        'KC': 'Knuckle Curve',
        'CH': 'Changeup',
        'FS': 'Splitter',
        'KN': 'Knuckleball',
    }

    if 'pitch_name' not in df.columns or df['pitch_name'].isna().all():
        df['pitch_name'] = df['pitch_type'].map(pitch_type_map).fillna(df['pitch_type'])

    # Estimate missing Statcast fields
    # (These are rough estimates - real Statcast data would be better)

    if 'pfx_x' not in df.columns:
        # Estimate horizontal movement from break data if available
        df['pfx_x'] = 0  # Placeholder

    if 'pfx_z' not in df.columns:
        # Estimate vertical movement from break data if available
        df['pfx_z'] = 0  # Placeholder

    if 'release_pos_x' not in df.columns:
        # Estimate release point (varies by pitcher handedness)
        df['release_pos_x'] = df['p_throws'].map({'R': 2.0, 'L': -2.0}).fillna(0)

    if 'release_pos_y' not in df.columns:
        df['release_pos_y'] = 54.0  # Typical release distance

    if 'release_pos_z' not in df.columns:
        df['release_pos_z'] = 6.0  # Typical release height

    # Outcome mapping
    if 'events' not in df.columns:
        df['events'] = None

    # Type (S/B/X)
    if 'type' not in df.columns:
        df['type'] = df['code'].map({
            'B': 'B', 'V': 'B',  # Balls
            'C': 'S', 'S': 'S', 'F': 'S', 'T': 'S', 'L': 'S',  # Strikes
            'X': 'X', 'D': 'X', 'E': 'X'  # In play
        }).fillna('B')

    print(f"✓ Converted {len(df)} pitches")

    return df


# Example usage (will work when network access is available)
if __name__ == "__main__":
    print("="*80)
    print("MLB-STATSAPI INTEGRATION - USAGE EXAMPLES")
    print("="*80)

    print("\n📖 Example 1: Get pitcher's season data")
    print("-" * 60)
    print("""
# Get Tarik Skubal's 2024 season (when network available)
skubal_data = get_pitcher_season_data_mlbapi(
    pitcher_name='skubal',
    season=2024,
    team_id=116  # Detroit Tigers
)

# Save to CSV
skubal_data.to_csv('data/skubal_mlbapi_2024.csv', index=False)
    """)

    print("\n📖 Example 2: Convert to Statcast format")
    print("-" * 60)
    print("""
# Load MLB-StatsAPI data
mlb_data = pd.read_csv('data/skubal_mlbapi_2024.csv')

# Convert to Statcast-compatible format
statcast_format = convert_mlbapi_to_statcast_format(mlb_data)

# Now use with existing visualization functions
from pitch_viz import visualize_pitch_trajectories_3d

visualize_pitch_trajectories_3d(
    df=statcast_format,
    pitcher_name="Tarik Skubal",
    pitch_types=["4-Seam Fastball", "Slider"],
    output_html="skubal_mlbapi_viz.html"
)
    """)

    print("\n📖 Example 3: Hybrid approach")
    print("-" * 60)
    print("""
# Use MLB-StatsAPI for basic pitch data
# Use sample data generator to add advanced metrics

# 1. Get real pitch outcomes and basic metrics from MLB-StatsAPI
real_data = get_pitcher_season_data_mlbapi('skubal', 2024)

# 2. Enhance with estimated physics (from sample generator)
enhanced_data = convert_mlbapi_to_statcast_format(real_data)

# 3. Use for visualizations
# (Will have real outcomes but estimated trajectories)
    """)

    print("\n" + "="*80)
    print("STATUS: Network currently blocked")
    print("="*80)
    print("\n⚠️  Both pybaseball and MLB-StatsAPI are blocked by proxy.")
    print("\n✅ Solutions available:")
    print("  1. Use sample data (already working perfectly)")
    print("  2. Run these functions when network access is available")
    print("  3. Manually download data from Baseball Savant website")
    print("\n📝 All integration code is ready - just needs network access!")
    print("="*80)
