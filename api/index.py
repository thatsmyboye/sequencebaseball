"""
Lightweight API for Vercel Serverless
Returns JSON data only - no heavy visualization dependencies
For full visualizations, use Railway/Render deployment
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import csv

app = FastAPI(
    title="Sequence Baseball API (Lite)",
    description="Lightweight API - JSON data only",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pitcher registry
PITCHERS = {
    669373: {"name": "Tarik Skubal", "team": "DET", "file": "skubal_statcast_2024.csv"},
    650556: {"name": "Jhoan Duran", "team": "MIN", "file": "duran_statcast_2024.csv"},
}

DATA_DIR = Path(__file__).parent.parent / "data"


def load_csv_as_dicts(filename: str) -> list:
    """Load CSV without pandas"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


@app.get("/")
def root():
    return {
        "status": "ok",
        "api": "Sequence Baseball API (Lite)",
        "note": "Lightweight version for Vercel. For visualizations, deploy to Railway."
    }


@app.get("/pitchers")
def list_pitchers():
    return [
        {"id": pid, "name": info["name"], "team": info["team"]}
        for pid, info in PITCHERS.items()
    ]


@app.get("/pitchers/{pitcher_id}/data")
def get_pitcher_data(pitcher_id: int, limit: int = 100):
    """Get raw pitch data (limited rows for performance)"""
    if pitcher_id not in PITCHERS:
        raise HTTPException(404, f"Pitcher {pitcher_id} not found")
    
    data = load_csv_as_dicts(PITCHERS[pitcher_id]["file"])
    return {
        "pitcher": PITCHERS[pitcher_id]["name"],
        "total_pitches": len(data),
        "data": data[:limit]
    }


@app.get("/pitchers/{pitcher_id}/summary")
def get_summary(pitcher_id: int):
    """Get pitch type summary without pandas"""
    if pitcher_id not in PITCHERS:
        raise HTTPException(404, f"Pitcher {pitcher_id} not found")
    
    data = load_csv_as_dicts(PITCHERS[pitcher_id]["file"])
    
    # Count pitch types
    pitch_counts = {}
    velocities = {}
    
    for row in data:
        pitch = row.get('pitch_name', 'Unknown')
        pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1
        
        try:
            velo = float(row.get('release_speed', 0))
            if pitch not in velocities:
                velocities[pitch] = []
            velocities[pitch].append(velo)
        except:
            pass
    
    # Calculate averages
    avg_velo = {
        p: round(sum(v)/len(v), 1) if v else 0 
        for p, v in velocities.items()
    }
    
    return {
        "pitcher": PITCHERS[pitcher_id]["name"],
        "team": PITCHERS[pitcher_id]["team"],
        "total_pitches": len(data),
        "pitch_counts": pitch_counts,
        "avg_velocity": avg_velo
    }




