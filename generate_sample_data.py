#!/usr/bin/env python
"""
Generate sample Statcast-like data for pitchers.
Creates realistic pitch data for testing when live data isn't available.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Pitch type profiles (velocity, h_break, v_break, spin)
PITCH_PROFILES = {
    "4-Seam Fastball": {"velo": (93, 98), "pfx_x": (-0.8, 0.8), "pfx_z": (1.2, 1.8), "spin": (2200, 2500)},
    "Sinker": {"velo": (91, 96), "pfx_x": (-1.5, -0.5), "pfx_z": (0.5, 1.2), "spin": (2000, 2300)},
    "Cutter": {"velo": (87, 93), "pfx_x": (0.2, 1.2), "pfx_z": (0.8, 1.4), "spin": (2300, 2600)},
    "Slider": {"velo": (83, 89), "pfx_x": (0.5, 2.5), "pfx_z": (-0.5, 0.5), "spin": (2400, 2800)},
    "Sweeper": {"velo": (78, 84), "pfx_x": (1.5, 3.5), "pfx_z": (-0.3, 0.3), "spin": (2600, 3000)},
    "Curveball": {"velo": (77, 84), "pfx_x": (0.3, 1.2), "pfx_z": (-1.5, -0.5), "spin": (2500, 3000)},
    "Changeup": {"velo": (82, 88), "pfx_x": (-1.5, -0.5), "pfx_z": (0.3, 1.0), "spin": (1600, 2000)},
    "Splitter": {"velo": (85, 91), "pfx_x": (-0.5, 0.5), "pfx_z": (0.0, 0.6), "spin": (1300, 1700)},
}

OUTCOMES = {
    "ball": 0.35,
    "called_strike": 0.18,
    "swinging_strike": 0.12,
    "foul": 0.20,
    "hit_into_play": 0.15,
}


def generate_pitcher_data(pitcher_id: int, pitcher_name: str, pitch_types: list, 
                          throws: str, num_pitches: int = 2000) -> pd.DataFrame:
    """Generate realistic pitch data for a pitcher."""
    
    np.random.seed(pitcher_id % 10000)
    random.seed(pitcher_id % 10000)
    
    # Generate game dates (spread across season)
    start_date = datetime(2024, 3, 28)
    end_date = datetime(2024, 9, 29)
    num_games = num_pitches // 90  # ~90 pitches per start
    
    game_dates = []
    for _ in range(num_games):
        days = random.randint(0, (end_date - start_date).days)
        game_dates.append(start_date + timedelta(days=days))
    game_dates = sorted(set(game_dates))
    
    data = []
    pitch_num = 0
    at_bat_num = 0
    
    for game_date in game_dates:
        if pitch_num >= num_pitches:
            break
            
        game_pitches = min(random.randint(80, 110), num_pitches - pitch_num)
        pitch_in_game = 1
        
        for _ in range(game_pitches // 4):  # ~4 pitches per at-bat
            at_bat_num += 1
            at_bat_pitches = random.randint(1, 7)
            batter_hand = random.choice(["R", "L"])
            
            balls, strikes = 0, 0
            
            for p in range(at_bat_pitches):
                if pitch_num >= num_pitches:
                    break
                    
                pitch_num += 1
                pitch_in_game += 1
                
                # Select pitch type (weighted toward primary pitches)
                weights = [1.0 / (i + 1) for i in range(len(pitch_types))]
                pitch_type = random.choices(pitch_types, weights=weights)[0]
                profile = PITCH_PROFILES.get(pitch_type, PITCH_PROFILES["4-Seam Fastball"])
                
                # Generate pitch characteristics
                velo = np.random.uniform(*profile["velo"])
                pfx_x = np.random.uniform(*profile["pfx_x"])
                pfx_z = np.random.uniform(*profile["pfx_z"])
                spin = np.random.uniform(*profile["spin"])
                
                # Adjust for handedness
                if throws == "L":
                    pfx_x = -pfx_x
                
                # Plate location
                plate_x = np.random.normal(0, 0.7)
                plate_z = np.random.normal(2.5, 0.5)
                
                # Determine zone (1-9 in zone, 11-14 out of zone)
                in_zone = abs(plate_x) < 0.83 and 1.5 < plate_z < 3.5
                zone = random.randint(1, 9) if in_zone else random.randint(11, 14)
                
                # Determine outcome
                outcome = random.choices(
                    list(OUTCOMES.keys()),
                    weights=list(OUTCOMES.values())
                )[0]
                
                # Update count based on outcome
                if outcome == "ball":
                    balls = min(balls + 1, 4)
                elif outcome in ["called_strike", "swinging_strike", "foul"]:
                    if strikes < 2 or outcome != "foul":
                        strikes = min(strikes + 1, 3)
                
                # Release point
                release_x = 2.0 if throws == "R" else -2.0
                release_x += np.random.normal(0, 0.2)
                release_z = np.random.normal(6.0, 0.3)
                release_y = np.random.normal(54.5, 0.5)
                
                data.append({
                    "pitch_type": pitch_type[:2].upper() if pitch_type != "4-Seam Fastball" else "FF",
                    "pitch_name": pitch_type,
                    "game_date": game_date.strftime("%Y-%m-%d"),
                    "release_speed": round(velo, 1),
                    "release_spin_rate": int(spin),
                    "pfx_x": round(pfx_x, 2),
                    "pfx_z": round(pfx_z, 2),
                    "plate_x": round(plate_x, 2),
                    "plate_z": round(plate_z, 2),
                    "zone": zone,
                    "description": outcome,
                    "balls": balls,
                    "strikes": strikes,
                    "stand": batter_hand,
                    "p_throws": throws,
                    "at_bat_number": at_bat_num,
                    "pitch_number": pitch_in_game,
                    "pitcher": pitcher_id,
                    "player_name": pitcher_name,
                    "release_pos_x": round(release_x, 2),
                    "release_pos_y": round(release_y, 2),
                    "release_pos_z": round(release_z, 2),
                    "inning": random.randint(1, 9),
                    "outs_when_up": random.randint(0, 2),
                    "home_team": "UNK",
                    "away_team": "UNK",
                })
    
    return pd.DataFrame(data)


def main():
    cache_dir = Path(__file__).parent / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Define pitchers to generate data for
    pitchers = [
        (669373, "Tarik Skubal", ["4-Seam Fastball", "Slider", "Changeup", "Curveball"], "L", 3000),
        (554430, "Zack Wheeler", ["4-Seam Fastball", "Slider", "Curveball", "Changeup", "Sinker"], "R", 3200),
        (543037, "Gerrit Cole", ["4-Seam Fastball", "Slider", "Curveball", "Changeup", "Cutter"], "R", 2800),
        (605483, "Chris Sale", ["4-Seam Fastball", "Slider", "Changeup"], "L", 2600),
        (650556, "Jhoan Duran", ["4-Seam Fastball", "Splitter", "Slider"], "R", 1200),
        (650911, "Cristopher Sanchez", ["Sinker", "Changeup", "Sweeper", "4-Seam Fastball"], "L", 2700),
        (592789, "Aaron Nola", ["4-Seam Fastball", "Curveball", "Changeup", "Sinker", "Cutter"], "R", 3000),
        (680686, "Paul Skenes", ["4-Seam Fastball", "Slider", "Curveball", "Changeup"], "R", 2100),
        (656849, "Tyler Glasnow", ["4-Seam Fastball", "Curveball", "Slider", "Changeup"], "R", 2300),
        (571945, "Corbin Burnes", ["Cutter", "Curveball", "Sinker", "Changeup", "4-Seam Fastball"], "R", 3100),
    ]
    
    print("Generating sample pitch data...")
    
    for pid, name, pitches, throws, count in pitchers:
        print(f"  Generating {name}...", end=" ")
        df = generate_pitcher_data(pid, name, pitches, throws, count)
        
        # Save to cache
        cache_file = cache_dir / f"pitcher_{pid}_2024.csv"
        df.to_csv(cache_file, index=False)
        print(f"✓ {len(df)} pitches")
    
    print(f"\nDone! Files saved to: {cache_dir}")


if __name__ == "__main__":
    main()
