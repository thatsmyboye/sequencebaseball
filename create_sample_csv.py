"""Quick script to create sample pitch data - run this directly"""
import csv
import random
from pathlib import Path

def main():
    cache_dir = Path(__file__).parent / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    pitchers = [
        (554430, "Zack Wheeler", "R", ["4-Seam Fastball", "Slider", "Curveball", "Sinker", "Changeup"]),
        (669373, "Tarik Skubal", "L", ["4-Seam Fastball", "Slider", "Changeup", "Curveball"]),
        (543037, "Gerrit Cole", "R", ["4-Seam Fastball", "Slider", "Curveball", "Cutter", "Changeup"]),
        (650556, "Jhoan Duran", "R", ["4-Seam Fastball", "Splitter", "Slider"]),
    ]
    
    outcomes = ["ball", "called_strike", "swinging_strike", "foul", "hit_into_play", "foul_tip"]
    
    for pid, name, throws, pitch_types in pitchers:
        rows = []
        at_bat = 0
        
        for game in range(25):  # 25 games
            game_date = f"2024-{4 + game // 5:02d}-{1 + (game % 28):02d}"
            
            for ab in range(22):  # ~22 at-bats per game
                at_bat += 1
                stand = random.choice(["R", "L"])
                num_pitches = random.randint(1, 7)
                balls, strikes = 0, 0
                
                for p in range(num_pitches):
                    pitch = random.choice(pitch_types)
                    outcome = random.choice(outcomes)
                    
                    # Update count
                    if outcome == "ball":
                        balls = min(balls + 1, 3)
                    elif outcome in ["called_strike", "swinging_strike"]:
                        strikes = min(strikes + 1, 2)
                    elif outcome == "foul" and strikes < 2:
                        strikes += 1
                    
                    velo = {"4-Seam Fastball": 96, "Slider": 87, "Curveball": 80, 
                           "Sinker": 94, "Changeup": 85, "Cutter": 90, "Splitter": 88}.get(pitch, 90)
                    velo += random.uniform(-2, 2)
                    
                    rows.append({
                        "pitch_type": pitch[:2].upper() if "Seam" not in pitch else "FF",
                        "pitch_name": pitch,
                        "game_date": game_date,
                        "release_speed": round(velo, 1),
                        "release_spin_rate": random.randint(2000, 2800),
                        "pfx_x": round(random.uniform(-1.5, 1.5), 2),
                        "pfx_z": round(random.uniform(-0.5, 1.5), 2),
                        "plate_x": round(random.uniform(-1, 1), 2),
                        "plate_z": round(random.uniform(1.5, 4), 2),
                        "zone": random.randint(1, 14),
                        "description": outcome,
                        "balls": balls,
                        "strikes": strikes,
                        "stand": stand,
                        "p_throws": throws,
                        "at_bat_number": at_bat,
                        "pitch_number": p + 1,
                        "pitcher": pid,
                        "player_name": name,
                        "release_pos_x": round(2.0 if throws == "R" else -2.0, 2),
                        "release_pos_y": 54.5,
                        "release_pos_z": 6.0,
                        "inning": (ab // 3) + 1,
                        "outs_when_up": ab % 3,
                    })
        
        # Write CSV
        csv_file = cache_dir / f"pitcher_{pid}_2024.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Created {csv_file.name} with {len(rows)} pitches")

if __name__ == "__main__":
    main()
