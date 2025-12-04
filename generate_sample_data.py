"""
Generate realistic sample Statcast data for development
Supports multiple seasons (2015-2024) with pitcher-specific career spans
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

# Pitcher career information (MLB debut year and seasons available)
PITCHER_CAREERS = {
    'skubal': {
        'id': 669373,
        'name': 'Tarik Skubal',
        'team': 'DET',
        'throws': 'L',
        'debut_year': 2020,
        # Skubal missed most of 2023 with injury, limited data
        'seasons': {
            2020: {'pitches': 400, 'note': 'Rookie debut (shortened COVID season)'},
            2021: {'pitches': 1800, 'note': 'First full season'},
            2022: {'pitches': 1600, 'note': 'Breakout year'},
            2023: {'pitches': 200, 'note': 'Limited due to injury'},
            2024: {'pitches': 2000, 'note': 'Cy Young season'},
        }
    },
    'duran': {
        'id': 650556,
        'name': 'Jhoan Duran',
        'team': 'MIN',
        'throws': 'R',
        'debut_year': 2022,
        'seasons': {
            2022: {'pitches': 800, 'note': 'Rookie season'},
            2023: {'pitches': 1200, 'note': 'All-Star breakout'},
            2024: {'pitches': 1500, 'note': 'Elite closer'},
        }
    },
    'wheeler': {
        'id': 554430,
        'name': 'Zack Wheeler',
        'team': 'PHI',
        'throws': 'R',
        'debut_year': 2013,  # But Statcast starts 2015
        'seasons': {
            2015: {'pitches': 1800, 'note': 'With Mets'},
            2016: {'pitches': 0, 'note': 'Missed (TJ recovery)'},
            2017: {'pitches': 1400, 'note': 'Return from injury'},
            2018: {'pitches': 1900, 'note': 'Strong comeback'},
            2019: {'pitches': 2100, 'note': 'Career year with Mets'},
            2020: {'pitches': 600, 'note': 'First year with Phillies (COVID)'},
            2021: {'pitches': 2200, 'note': 'Cy Young runner-up'},
            2022: {'pitches': 2000, 'note': 'World Series run'},
            2023: {'pitches': 1800, 'note': 'Consistent ace'},
            2024: {'pitches': 2000, 'note': 'Elite performance'},
        }
    },
    'sanchez': {
        'id': 650911,
        'name': 'Cristopher Sanchez',
        'team': 'PHI',
        'throws': 'L',
        'debut_year': 2021,
        'seasons': {
            2021: {'pitches': 150, 'note': 'Brief MLB debut'},
            2022: {'pitches': 400, 'note': 'Spot starts/long relief'},
            2023: {'pitches': 1400, 'note': 'Rotation spot'},
            2024: {'pitches': 1800, 'note': 'Established starter'},
        }
    }
}


def generate_pitch_data_for_season(
    pitcher_key: str,
    season: int,
    pitch_arsenal: dict,
    num_pitches: int
) -> pd.DataFrame:
    """
    Generate realistic Statcast pitch data for a specific season.
    
    Parameters:
    - pitcher_key: str, key in PITCHER_CAREERS
    - season: int, year (2015-2024)
    - pitch_arsenal: dict with pitch types and characteristics
    - num_pitches: int, total pitches to generate
    """
    pitcher_info = PITCHER_CAREERS[pitcher_key]
    pitcher_name = pitcher_info['name']
    pitcher_id = pitcher_info['id']
    throws = pitcher_info['throws']
    
    if num_pitches == 0:
        return pd.DataFrame()
    
    # Seed based on pitcher and season for reproducibility
    np.random.seed(pitcher_id + season)
    
    data = []
    
    # Season date range (approximation)
    season_start = datetime(season, 4, 1)
    season_end = datetime(season, 10, 1)
    season_days = (season_end - season_start).days
    
    at_bat_num = 1
    pitch_num_in_ab = 1
    current_game_date = season_start
    pitches_in_game = 0
    
    # Determine pitches per game (varies by role)
    if pitcher_info['team'] == 'MIN' and pitcher_key == 'duran':
        # Reliever: ~15-25 pitches per appearance
        pitches_per_game = 20
    else:
        # Starter: ~85-100 pitches per game
        pitches_per_game = 90

    for i in range(num_pitches):
        # Advance game date periodically
        pitches_in_game += 1
        if pitches_in_game > pitches_per_game:
            pitches_in_game = 0
            days_advance = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.25, 0.30, 0.25]))
            current_game_date += timedelta(days=days_advance)
            if current_game_date > season_end:
                current_game_date = season_end
                
        # Randomly select pitch type based on usage
        pitch_type = np.random.choice(
            list(pitch_arsenal.keys()),
            p=[p['usage'] for p in pitch_arsenal.values()]
        )
        pitch_info = pitch_arsenal[pitch_type]

        # Batter handedness
        stand = np.random.choice(['R', 'L'], p=[0.55, 0.45])

        # Release point (with small variation)
        release_pos_x = np.random.normal(pitch_info['release_x'], 0.15)
        release_pos_y = np.random.normal(50.0, 0.3)
        release_pos_z = np.random.normal(pitch_info['release_z'], 0.1)
        release_extension = np.random.normal(6.2, 0.2)

        # Velocity - slight year-over-year variation
        velo_adjustment = (season - 2020) * 0.1  # Slight velo changes over years
        release_speed = np.random.normal(pitch_info['velo'] + velo_adjustment, pitch_info['velo_std'])

        # Spin rate
        release_spin_rate = np.random.normal(pitch_info['spin'], pitch_info['spin_std'])

        # Movement (in inches)
        pfx_x = np.random.normal(pitch_info['horz_break'], 1.5)
        pfx_z = np.random.normal(pitch_info['vert_break'], 1.5)

        # Plate location
        plate_x = np.random.normal(0, 0.8)
        plate_z = np.random.normal(2.5, 0.6)

        # Zone (1-9 in strike zone, 11-14 outside)
        if abs(plate_x) < 0.83 and 1.5 < plate_z < 3.5:
            zone = np.random.choice(range(1, 10))
        else:
            zone = np.random.choice(range(11, 15))

        # Count
        balls = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15])
        strikes = np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])

        # Pitch outcome
        outcomes = [
            'ball', 'called_strike', 'swinging_strike', 'foul',
            'hit_into_play', 'blocked_ball', 'foul_tip'
        ]
        outcome_probs = [0.35, 0.15, 0.12, 0.20, 0.15, 0.02, 0.01]
        description = np.random.choice(outcomes, p=outcome_probs)

        # Event (at-bat result)
        events = None
        if description == 'hit_into_play':
            events = np.random.choice([
                'single', 'double', 'triple', 'home_run',
                'field_out', 'grounded_into_double_play', 'force_out'
            ], p=[0.15, 0.05, 0.01, 0.04, 0.60, 0.10, 0.05])
        elif strikes == 2 and description == 'swinging_strike':
            events = 'strikeout'

        # Physics components
        vx0 = np.random.normal(0, 5)
        vy0 = np.random.normal(-130, 10)
        vz0 = np.random.normal(0, 5)

        ax = pfx_x * 12 / (50.0**2) * -1
        ay = 20.0
        az = pfx_z * 12 / (50.0**2) * -1 + 32.174

        effective_speed = release_speed * 0.95
        spin_axis = np.random.normal(pitch_info.get('spin_axis', 200), 15)

        pitch = {
            'pitch_type': pitch_type,
            'game_date': current_game_date.strftime('%Y-%m-%d'),
            'release_speed': round(release_speed, 1),
            'release_pos_x': round(release_pos_x, 2),
            'release_pos_y': round(release_pos_y, 2),
            'release_pos_z': round(release_pos_z, 2),
            'pitcher': pitcher_id,
            'batter': 500000 + i % 100,
            'events': events,
            'description': description,
            'zone': zone,
            'stand': stand,
            'p_throws': throws,
            'balls': balls,
            'strikes': strikes,
            'pfx_x': round(pfx_x, 2),
            'pfx_z': round(pfx_z, 2),
            'plate_x': round(plate_x, 2),
            'plate_z': round(plate_z, 2),
            'vx0': round(vx0, 2),
            'vy0': round(vy0, 2),
            'vz0': round(vz0, 2),
            'ax': round(ax, 2),
            'ay': round(ay, 2),
            'az': round(az, 2),
            'effective_speed': round(effective_speed, 1),
            'release_spin_rate': int(release_spin_rate),
            'release_extension': round(release_extension, 1),
            'pitch_name': pitch_info['name'],
            'spin_axis': int(spin_axis),
            'at_bat_number': at_bat_num,
            'pitch_number': pitch_num_in_ab,
            'inning': ((at_bat_num - 1) // 6) + 1,
            'outs_when_up': np.random.choice([0, 1, 2]),
            'type': 'S' if description in ['called_strike', 'swinging_strike', 'foul', 'foul_tip'] else 'B' if description == 'ball' else 'X',
            'season': season  # Add season column for filtering
        }

        data.append(pitch)

        # Increment pitch/at-bat counters
        pitch_num_in_ab += 1
        if events is not None or pitch_num_in_ab > 8:
            at_bat_num += 1
            pitch_num_in_ab = 1

    return pd.DataFrame(data)


# Pitch arsenals by pitcher
ARSENALS = {
    'skubal': {
        'FF': {
            'name': '4-Seam Fastball',
            'usage': 0.55,
            'velo': 96.5,
            'velo_std': 1.2,
            'spin': 2400,
            'spin_std': 100,
            'horz_break': 6.5,
            'vert_break': 16.0,
            'release_x': -2.1,
            'release_z': 5.8,
            'spin_axis': 205
        },
        'SL': {
            'name': 'Slider',
            'usage': 0.40,
            'velo': 86.0,
            'velo_std': 1.5,
            'spin': 2650,
            'spin_std': 120,
            'horz_break': 2.0,
            'vert_break': 3.0,
            'release_x': -2.0,
            'release_z': 5.7,
            'spin_axis': 75
        },
        'CH': {
            'name': 'Changeup',
            'usage': 0.05,
            'velo': 88.5,
            'velo_std': 1.0,
            'spin': 1650,
            'spin_std': 80,
            'horz_break': 12.0,
            'vert_break': 6.0,
            'release_x': -2.1,
            'release_z': 5.8,
            'spin_axis': 240
        }
    },
    'duran': {
        'FF': {
            'name': '4-Seam Fastball',
            'usage': 0.45,
            'velo': 100.5,
            'velo_std': 1.5,
            'spin': 2300,
            'spin_std': 100,
            'horz_break': -8.0,
            'vert_break': 14.0,
            'release_x': 2.0,
            'release_z': 6.0,
            'spin_axis': 200
        },
        'FS': {
            'name': 'Splitter',
            'usage': 0.50,
            'velo': 94.0,
            'velo_std': 1.2,
            'spin': 1400,
            'spin_std': 80,
            'horz_break': -13.0,
            'vert_break': -4.0,
            'release_x': 2.0,
            'release_z': 5.9,
            'spin_axis': 220
        },
        'SL': {
            'name': 'Slider',
            'usage': 0.05,
            'velo': 88.0,
            'velo_std': 1.5,
            'spin': 2500,
            'spin_std': 100,
            'horz_break': -1.0,
            'vert_break': 1.0,
            'release_x': 1.9,
            'release_z': 5.8,
            'spin_axis': 90
        }
    },
    'wheeler': {
        'FF': {
            'name': '4-Seam Fastball',
            'usage': 0.35,
            'velo': 96.0,
            'velo_std': 1.3,
            'spin': 2350,
            'spin_std': 100,
            'horz_break': -7.5,
            'vert_break': 15.0,
            'release_x': 2.2,
            'release_z': 5.9,
            'spin_axis': 210
        },
        'SL': {
            'name': 'Slider',
            'usage': 0.30,
            'velo': 84.5,
            'velo_std': 1.5,
            'spin': 2700,
            'spin_std': 120,
            'horz_break': 1.0,
            'vert_break': 2.5,
            'release_x': 2.1,
            'release_z': 5.8,
            'spin_axis': 50
        },
        'CU': {
            'name': 'Curveball',
            'usage': 0.15,
            'velo': 79.0,
            'velo_std': 1.5,
            'spin': 2800,
            'spin_std': 150,
            'horz_break': 5.0,
            'vert_break': -8.0,
            'release_x': 2.0,
            'release_z': 5.9,
            'spin_axis': 30
        },
        'CH': {
            'name': 'Changeup',
            'usage': 0.10,
            'velo': 88.0,
            'velo_std': 1.2,
            'spin': 1700,
            'spin_std': 90,
            'horz_break': -14.0,
            'vert_break': 4.0,
            'release_x': 2.2,
            'release_z': 5.8,
            'spin_axis': 230
        },
        'SI': {
            'name': 'Sinker',
            'usage': 0.10,
            'velo': 95.0,
            'velo_std': 1.2,
            'spin': 2100,
            'spin_std': 100,
            'horz_break': -15.0,
            'vert_break': 8.0,
            'release_x': 2.2,
            'release_z': 5.9,
            'spin_axis': 225
        }
    },
    'sanchez': {
        'SI': {
            'name': 'Sinker',
            'usage': 0.45,
            'velo': 93.5,
            'velo_std': 1.2,
            'spin': 2000,
            'spin_std': 100,
            'horz_break': 15.0,
            'vert_break': 7.0,
            'release_x': -2.3,
            'release_z': 5.5,
            'spin_axis': 215
        },
        'CH': {
            'name': 'Changeup',
            'usage': 0.30,
            'velo': 84.0,
            'velo_std': 1.3,
            'spin': 1550,
            'spin_std': 80,
            'horz_break': 14.0,
            'vert_break': 2.0,
            'release_x': -2.3,
            'release_z': 5.5,
            'spin_axis': 245
        },
        'SW': {
            'name': 'Sweeper',
            'usage': 0.15,
            'velo': 80.0,
            'velo_std': 1.5,
            'spin': 2600,
            'spin_std': 120,
            'horz_break': -5.0,
            'vert_break': 0.0,
            'release_x': -2.2,
            'release_z': 5.6,
            'spin_axis': 100
        },
        'FF': {
            'name': '4-Seam Fastball',
            'usage': 0.10,
            'velo': 94.0,
            'velo_std': 1.2,
            'spin': 2200,
            'spin_std': 100,
            'horz_break': 7.0,
            'vert_break': 14.0,
            'release_x': -2.3,
            'release_z': 5.6,
            'spin_axis': 200
        }
    }
}


def generate_all_pitcher_data():
    """Generate multi-season data for all pitchers and save to CSV files."""
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GENERATING MULTI-SEASON STATCAST DATA")
    print("=" * 80)
    
    for pitcher_key, career in PITCHER_CAREERS.items():
        print(f"\n{'='*60}")
        print(f"PITCHER: {career['name']} ({career['team']})")
        print(f"{'='*60}")
        
        all_seasons_data = []
        
        for season, season_info in career['seasons'].items():
            num_pitches = season_info['pitches']
            note = season_info['note']
            
            if num_pitches == 0:
                print(f"  {season}: Skipped - {note}")
                continue
            
            print(f"  {season}: Generating {num_pitches} pitches ({note})...")
            
            season_df = generate_pitch_data_for_season(
                pitcher_key=pitcher_key,
                season=season,
                pitch_arsenal=ARSENALS[pitcher_key],
                num_pitches=num_pitches
            )
            
            if len(season_df) > 0:
                all_seasons_data.append(season_df)
                print(f"         ✓ Date range: {season_df['game_date'].min()} to {season_df['game_date'].max()}")
        
        if all_seasons_data:
            # Combine all seasons
            combined_df = pd.concat(all_seasons_data, ignore_index=True)
            
            # Save combined file (all seasons)
            output_file = data_dir / f"{pitcher_key}_statcast_all.csv"
            combined_df.to_csv(output_file, index=False)
            
            # Also save most recent season as the default file (backwards compatible)
            latest_season = max(career['seasons'].keys())
            latest_df = combined_df[combined_df['season'] == latest_season]
            default_file = data_dir / f"{pitcher_key}_statcast_2024.csv"
            latest_df.to_csv(default_file, index=False)
            
            print(f"\n  ✓ Saved: {output_file}")
            print(f"    Total pitches: {len(combined_df)}")
            print(f"    Seasons: {sorted(combined_df['season'].unique())}")
            print(f"  ✓ Saved: {default_file} (2024 only, {len(latest_df)} pitches)")
    
    print("\n" + "=" * 80)
    print("MULTI-SEASON DATA GENERATION COMPLETE")
    print("=" * 80)
    
    # Print summary
    print("\nSUMMARY:")
    for pitcher_key, career in PITCHER_CAREERS.items():
        available_seasons = [s for s, info in career['seasons'].items() if info['pitches'] > 0]
        print(f"  {career['name']}: {min(available_seasons)}-{max(available_seasons)} ({len(available_seasons)} seasons)")


def get_pitcher_available_seasons(pitcher_key: str) -> dict:
    """Get available seasons for a pitcher with metadata."""
    if pitcher_key not in PITCHER_CAREERS:
        return {}
    
    career = PITCHER_CAREERS[pitcher_key]
    return {
        'pitcher_id': career['id'],
        'pitcher_name': career['name'],
        'available_seasons': [
            {
                'season': season,
                'pitches': info['pitches'],
                'note': info['note']
            }
            for season, info in sorted(career['seasons'].items())
            if info['pitches'] > 0
        ]
    }


if __name__ == "__main__":
    generate_all_pitcher_data()
