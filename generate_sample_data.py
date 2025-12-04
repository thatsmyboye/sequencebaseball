"""
Generate realistic sample Statcast data for development
Based on actual Statcast schema and pitcher profiles
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_pitch_data(pitcher_name, pitcher_id, pitch_arsenal, throws='R', num_pitches=2000):
    """
    Generate realistic Statcast pitch data

    Parameters:
    - pitcher_name: str
    - pitcher_id: int (MLB ID)
    - pitch_arsenal: dict with pitch types and their characteristics
    - throws: str, 'L' or 'R' for pitcher handedness
    - num_pitches: int, total pitches to generate
    """

    data = []
    game_date = datetime(2024, 4, 1)
    at_bat_num = 1
    pitch_num_in_ab = 1

    # Generate pitches
    for i in range(num_pitches):
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
        release_pos_y = np.random.normal(50.0, 0.3)  # Distance from rubber
        release_pos_z = np.random.normal(pitch_info['release_z'], 0.1)
        release_extension = np.random.normal(6.2, 0.2)

        # Velocity
        release_speed = np.random.normal(pitch_info['velo'], pitch_info['velo_std'])

        # Spin rate
        release_spin_rate = np.random.normal(pitch_info['spin'], pitch_info['spin_std'])

        # Movement (in inches)
        pfx_x = np.random.normal(pitch_info['horz_break'], 1.5)  # Horizontal
        pfx_z = np.random.normal(pitch_info['vert_break'], 1.5)  # Vertical (induced)

        # Plate location
        plate_x = np.random.normal(0, 0.8)  # Horizontal location at plate
        plate_z = np.random.normal(2.5, 0.6)  # Vertical location at plate

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

        # Event (at-bat result) - only on last pitch
        events = None
        if description == 'hit_into_play':
            events = np.random.choice([
                'single', 'double', 'triple', 'home_run',
                'field_out', 'grounded_into_double_play', 'force_out'
            ], p=[0.15, 0.05, 0.01, 0.04, 0.60, 0.10, 0.05])
        elif strikes == 2 and description == 'swinging_strike':
            events = 'strikeout'

        # Physics components (initial velocities and accelerations)
        vx0 = np.random.normal(0, 5)  # Initial velocity X
        vy0 = np.random.normal(-130, 10)  # Initial velocity Y (toward plate)
        vz0 = np.random.normal(0, 5)  # Initial velocity Z

        ax = pfx_x * 12 / (50.0**2) * -1  # Acceleration X (from movement)
        ay = 20.0  # Air resistance
        az = pfx_z * 12 / (50.0**2) * -1 + 32.174  # Acceleration Z (gravity + movement)

        # Effective speed (perceived velocity)
        effective_speed = release_speed * 0.95

        # Spin axis (degrees)
        spin_axis = np.random.normal(pitch_info.get('spin_axis', 200), 15)

        # Build pitch record
        pitch = {
            'pitch_type': pitch_type,
            'game_date': game_date.strftime('%Y-%m-%d'),
            'release_speed': round(release_speed, 1),
            'release_pos_x': round(release_pos_x, 2),
            'release_pos_y': round(release_pos_y, 2),
            'release_pos_z': round(release_pos_z, 2),
            'pitcher': pitcher_id,
            'batter': 500000 + i % 100,  # Dummy batter IDs
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
            'type': 'S' if description in ['called_strike', 'swinging_strike', 'foul', 'foul_tip'] else 'B' if description == 'ball' else 'X'
        }

        data.append(pitch)

        # Increment pitch/at-bat counters
        pitch_num_in_ab += 1

        # End at-bat conditions
        if events is not None or pitch_num_in_ab > 8:
            at_bat_num += 1
            pitch_num_in_ab = 1

            # Advance game date every ~25 at-bats (roughly one game)
            if at_bat_num % 25 == 0:
                game_date += timedelta(days=int(np.random.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.3])))

    return pd.DataFrame(data)


# Tarik Skubal Arsenal (2024 Cy Young Winner)
# Left-handed, elite fastball-slider combo
skubal_arsenal = {
    'FF': {  # 4-Seam Fastball
        'name': '4-Seam Fastball',
        'usage': 0.55,
        'velo': 96.5,
        'velo_std': 1.2,
        'spin': 2400,
        'spin_std': 100,
        'horz_break': 6.5,  # Arm-side run (LHP)
        'vert_break': 16.0,  # Rise
        'release_x': -2.1,  # LHP releases from 3B side
        'release_z': 5.8,
        'spin_axis': 205
    },
    'SL': {  # Slider
        'name': 'Slider',
        'usage': 0.40,
        'velo': 86.0,
        'velo_std': 1.5,
        'spin': 2650,
        'spin_std': 120,
        'horz_break': 2.0,  # Glove-side break
        'vert_break': 3.0,  # Drop
        'release_x': -2.0,
        'release_z': 5.7,
        'spin_axis': 75
    },
    'CH': {  # Changeup
        'name': 'Changeup',
        'usage': 0.05,
        'velo': 88.5,
        'velo_std': 1.0,
        'spin': 1650,
        'spin_std': 80,
        'horz_break': 12.0,  # Arm-side fade
        'vert_break': 6.0,  # Drop
        'release_x': -2.1,
        'release_z': 5.8,
        'spin_axis': 240
    }
}

# Jhoan Duran Arsenal
# Right-handed, known for elite "Splinker" (splitter/sinker hybrid)
duran_arsenal = {
    'FF': {  # 4-Seam Fastball
        'name': '4-Seam Fastball',
        'usage': 0.45,
        'velo': 100.5,  # Elite velocity
        'velo_std': 1.5,
        'spin': 2300,
        'spin_std': 100,
        'horz_break': -8.0,  # Arm-side run (RHP)
        'vert_break': 14.0,  # Rise
        'release_x': 2.0,  # RHP releases from 1B side
        'release_z': 6.0,
        'spin_axis': 200
    },
    'FS': {  # Splinker (Splitter)
        'name': 'Splitter',
        'usage': 0.50,  # His signature pitch
        'velo': 94.0,  # Hardest splitter in baseball
        'velo_std': 1.2,
        'spin': 1400,  # Low spin
        'spin_std': 80,
        'horz_break': -13.0,  # Heavy arm-side fade
        'vert_break': -4.0,  # Significant drop
        'release_x': 2.0,
        'release_z': 5.9,
        'spin_axis': 220
    },
    'SL': {  # Slider (rare)
        'name': 'Slider',
        'usage': 0.05,
        'velo': 88.0,
        'velo_std': 1.5,
        'spin': 2500,
        'spin_std': 100,
        'horz_break': -1.0,  # Glove-side
        'vert_break': 1.0,
        'release_x': 1.9,
        'release_z': 5.8,
        'spin_axis': 90
    }
}

# Zack Wheeler Arsenal (Phillies Ace)
# Right-handed, elite four-pitch mix
wheeler_arsenal = {
    'FF': {  # 4-Seam Fastball
        'name': '4-Seam Fastball',
        'usage': 0.35,
        'velo': 96.0,
        'velo_std': 1.3,
        'spin': 2350,
        'spin_std': 100,
        'horz_break': -7.5,  # Arm-side run (RHP)
        'vert_break': 15.0,  # Rise
        'release_x': 2.2,  # RHP releases from 1B side
        'release_z': 5.9,
        'spin_axis': 210
    },
    'SL': {  # Slider
        'name': 'Slider',
        'usage': 0.30,
        'velo': 84.5,
        'velo_std': 1.5,
        'spin': 2700,
        'spin_std': 120,
        'horz_break': 1.0,  # Glove-side break
        'vert_break': 2.5,  # Drop
        'release_x': 2.1,
        'release_z': 5.8,
        'spin_axis': 50
    },
    'CU': {  # Curveball
        'name': 'Curveball',
        'usage': 0.15,
        'velo': 79.0,
        'velo_std': 1.5,
        'spin': 2800,
        'spin_std': 150,
        'horz_break': 5.0,  # Glove-side
        'vert_break': -8.0,  # Drop
        'release_x': 2.0,
        'release_z': 5.9,
        'spin_axis': 30
    },
    'CH': {  # Changeup
        'name': 'Changeup',
        'usage': 0.10,
        'velo': 88.0,
        'velo_std': 1.2,
        'spin': 1700,
        'spin_std': 90,
        'horz_break': -14.0,  # Arm-side fade
        'vert_break': 4.0,  # Some drop
        'release_x': 2.2,
        'release_z': 5.8,
        'spin_axis': 230
    },
    'SI': {  # Sinker
        'name': 'Sinker',
        'usage': 0.10,
        'velo': 95.0,
        'velo_std': 1.2,
        'spin': 2100,
        'spin_std': 100,
        'horz_break': -15.0,  # Heavy arm-side run
        'vert_break': 8.0,  # Less rise than 4-seam
        'release_x': 2.2,
        'release_z': 5.9,
        'spin_axis': 225
    }
}

# Cristopher Sanchez Arsenal (Phillies)
# Left-handed, sinker-changeup dominant
sanchez_arsenal = {
    'SI': {  # Sinker (primary pitch)
        'name': 'Sinker',
        'usage': 0.45,
        'velo': 93.5,
        'velo_std': 1.2,
        'spin': 2000,
        'spin_std': 100,
        'horz_break': 15.0,  # Arm-side run (LHP)
        'vert_break': 7.0,  # Less rise, more sink
        'release_x': -2.3,  # LHP releases from 3B side
        'release_z': 5.5,
        'spin_axis': 215
    },
    'CH': {  # Changeup (elite)
        'name': 'Changeup',
        'usage': 0.30,
        'velo': 84.0,
        'velo_std': 1.3,
        'spin': 1550,
        'spin_std': 80,
        'horz_break': 14.0,  # Arm-side fade
        'vert_break': 2.0,  # Drop
        'release_x': -2.3,
        'release_z': 5.5,
        'spin_axis': 245
    },
    'SW': {  # Sweeper
        'name': 'Sweeper',
        'usage': 0.15,
        'velo': 80.0,
        'velo_std': 1.5,
        'spin': 2600,
        'spin_std': 120,
        'horz_break': -5.0,  # Glove-side sweep
        'vert_break': 0.0,  # Horizontal break
        'release_x': -2.2,
        'release_z': 5.6,
        'spin_axis': 100
    },
    'FF': {  # 4-Seam Fastball
        'name': '4-Seam Fastball',
        'usage': 0.10,
        'velo': 94.0,
        'velo_std': 1.2,
        'spin': 2200,
        'spin_std': 100,
        'horz_break': 7.0,  # Arm-side (LHP)
        'vert_break': 14.0,  # Rise
        'release_x': -2.3,
        'release_z': 5.6,
        'spin_axis': 200
    }
}

print("="*80)
print("GENERATING SAMPLE STATCAST DATA")
print("="*80)

print("\nGenerating Tarik Skubal data (2000 pitches)...")
skubal_df = generate_pitch_data("Tarik Skubal", 669373, skubal_arsenal, throws='L', num_pitches=2000)
skubal_df.to_csv('data/skubal_statcast_2024.csv', index=False)
print(f"✓ Saved: data/skubal_statcast_2024.csv")
print(f"  Shape: {skubal_df.shape}")
print(f"  Date range: {skubal_df['game_date'].min()} to {skubal_df['game_date'].max()}")
print(f"  Pitch types: {skubal_df['pitch_name'].value_counts().to_dict()}")

print("\nGenerating Jhoan Duran data (1500 pitches)...")
duran_df = generate_pitch_data("Jhoan Duran", 650556, duran_arsenal, throws='R', num_pitches=1500)
duran_df.to_csv('data/duran_statcast_2024.csv', index=False)
print(f"✓ Saved: data/duran_statcast_2024.csv")
print(f"  Shape: {duran_df.shape}")
print(f"  Date range: {duran_df['game_date'].min()} to {duran_df['game_date'].max()}")
print(f"  Pitch types: {duran_df['pitch_name'].value_counts().to_dict()}")

print("\nGenerating Zack Wheeler data (2000 pitches)...")
wheeler_df = generate_pitch_data("Zack Wheeler", 554430, wheeler_arsenal, throws='R', num_pitches=2000)
wheeler_df.to_csv('data/wheeler_statcast_2024.csv', index=False)
print(f"✓ Saved: data/wheeler_statcast_2024.csv")
print(f"  Shape: {wheeler_df.shape}")
print(f"  Date range: {wheeler_df['game_date'].min()} to {wheeler_df['game_date'].max()}")
print(f"  Pitch types: {wheeler_df['pitch_name'].value_counts().to_dict()}")

print("\nGenerating Cristopher Sanchez data (1800 pitches)...")
sanchez_df = generate_pitch_data("Cristopher Sanchez", 650911, sanchez_arsenal, throws='L', num_pitches=1800)
sanchez_df.to_csv('data/sanchez_statcast_2024.csv', index=False)
print(f"✓ Saved: data/sanchez_statcast_2024.csv")
print(f"  Shape: {sanchez_df.shape}")
print(f"  Date range: {sanchez_df['game_date'].min()} to {sanchez_df['game_date'].max()}")
print(f"  Pitch types: {sanchez_df['pitch_name'].value_counts().to_dict()}")

print("\n" + "="*80)
print("SAMPLE DATA GENERATION COMPLETE")
print("="*80)
print("\nNote: This is realistic sample data based on actual pitcher profiles.")
print("Replace with real Statcast data when network access is available.")
