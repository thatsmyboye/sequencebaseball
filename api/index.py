"""
Lightweight API for Vercel Serverless
Returns JSON data only - no heavy visualization dependencies
For full visualizations, use Railway/Render deployment

Includes SportRadar MLB API integration for live data.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import csv
import os
import requests
from typing import Optional

app = FastAPI(
    title="Sequence Baseball API (Lite)",
    description="Lightweight API - JSON data only. Includes SportRadar MLB integration.",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pitcher registry (local data)
PITCHERS = {
    669373: {"name": "Tarik Skubal", "team": "DET", "file": "skubal_statcast_2024.csv"},
    650556: {"name": "Jhoan Duran", "team": "MIN", "file": "duran_statcast_2024.csv"},
}

DATA_DIR = Path(__file__).parent.parent / "data"

# SportRadar API Configuration
# API Key valid until January 4, 2026
SPORTRADAR_API_KEY = os.environ.get(
    "SPORTRADAR_API_KEY",
    "MkPCdbssBXUy74XhOtt58Ed7Ov8GMrJnVGHk2YmN"
)
SPORTRADAR_BASE_URL = "https://api.sportradar.com/mlb/trial/v8/en"


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


# ============================================================================
# SPORTRADAR MLB API ENDPOINTS
# ============================================================================

def sportradar_request(endpoint: str) -> dict:
    """Make request to SportRadar API"""
    url = f"{SPORTRADAR_BASE_URL}/{endpoint}"
    params = {"api_key": SPORTRADAR_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            raise HTTPException(403, "SportRadar API access denied. Check API key.")
        elif response.status_code == 429:
            raise HTTPException(429, "SportRadar API rate limit exceeded. Please wait.")
        elif response.status_code == 404:
            raise HTTPException(404, f"SportRadar resource not found: {endpoint}")
        else:
            raise HTTPException(response.status_code, f"SportRadar API error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(503, f"SportRadar API connection error: {str(e)}")


@app.get("/sportradar/status")
def sportradar_status():
    """Check SportRadar API connection status"""
    try:
        glossary = sportradar_request("league/glossary.json")
        return {
            "status": "connected",
            "api": "SportRadar MLB v8",
            "pitch_types_available": len(glossary.get("pitch_types", [])),
            "note": "Trial API - does not include Statcast data"
        }
    except HTTPException as e:
        return {
            "status": "error",
            "error": e.detail
        }


@app.get("/sportradar/schedule/{date}")
def sportradar_daily_schedule(date: str):
    """
    Get MLB schedule for a date
    
    - **date**: Date in YYYY-MM-DD format (e.g., 2024-08-15)
    """
    try:
        year, month, day = date.split('-')
    except ValueError:
        raise HTTPException(400, "Date must be in YYYY-MM-DD format")
    
    endpoint = f"games/{year}/{month}/{day}/schedule.json"
    data = sportradar_request(endpoint)
    
    # Simplify response
    games = []
    for game in data.get("games", []):
        games.append({
            "id": game.get("id"),
            "status": game.get("status"),
            "scheduled": game.get("scheduled"),
            "home": {
                "name": game.get("home", {}).get("name"),
                "abbr": game.get("home", {}).get("abbr"),
            },
            "away": {
                "name": game.get("away", {}).get("name"),
                "abbr": game.get("away", {}).get("abbr"),
            },
            "venue": game.get("venue", {}).get("name"),
        })
    
    return {
        "date": date,
        "game_count": len(games),
        "games": games
    }


@app.get("/sportradar/game/{game_id}/boxscore")
def sportradar_game_boxscore(game_id: str):
    """
    Get boxscore for a specific game
    
    - **game_id**: SportRadar game ID (UUID format)
    """
    endpoint = f"games/{game_id}/boxscore.json"
    return sportradar_request(endpoint)


@app.get("/sportradar/game/{game_id}/pitch_metrics")
def sportradar_game_pitch_metrics(game_id: str):
    """
    Get pitch metrics for a game
    
    Includes pitch type, velocity, and results for all pitchers.
    Note: Does NOT include Statcast data (spin rate, movement, etc.)
    
    - **game_id**: SportRadar game ID (UUID format)
    """
    endpoint = f"games/{game_id}/pitch_metrics.json"
    return sportradar_request(endpoint)


@app.get("/sportradar/game/{game_id}/pbp")
def sportradar_game_pbp(game_id: str):
    """
    Get play-by-play data for a game
    
    - **game_id**: SportRadar game ID (UUID format)
    """
    endpoint = f"games/{game_id}/pbp.json"
    return sportradar_request(endpoint)


@app.get("/sportradar/teams")
def sportradar_teams():
    """Get all MLB teams"""
    endpoint = "league/teams.json"
    data = sportradar_request(endpoint)
    
    teams = []
    for team in data.get("teams", []):
        teams.append({
            "id": team.get("id"),
            "name": team.get("name"),
            "market": team.get("market"),
            "abbr": team.get("abbr"),
            "league": team.get("league", {}).get("name"),
            "division": team.get("division", {}).get("name"),
        })
    
    return {"teams": teams}


@app.get("/sportradar/team/{team_id}/profile")
def sportradar_team_profile(team_id: str):
    """
    Get team profile with roster
    
    - **team_id**: SportRadar team ID (UUID format)
    """
    endpoint = f"teams/{team_id}/profile.json"
    return sportradar_request(endpoint)


@app.get("/sportradar/player/{player_id}/profile")
def sportradar_player_profile(player_id: str):
    """
    Get player profile with stats
    
    - **player_id**: SportRadar player ID (UUID format)
    """
    endpoint = f"players/{player_id}/profile.json"
    return sportradar_request(endpoint)


@app.get("/sportradar/standings/{season}")
def sportradar_standings(season: int, season_type: str = "REG"):
    """
    Get league standings
    
    - **season**: Season year (e.g., 2024)
    - **season_type**: PRE (Spring Training), REG (Regular Season), PST (Postseason)
    """
    endpoint = f"seasons/{season}/{season_type}/standings.json"
    return sportradar_request(endpoint)


@app.get("/sportradar/leaders/{season}")
def sportradar_leaders(season: int, season_type: str = "REG"):
    """
    Get league leaders for various statistics
    
    - **season**: Season year (e.g., 2024)
    - **season_type**: PRE, REG, or PST
    """
    endpoint = f"seasons/{season}/{season_type}/leaders.json"
    return sportradar_request(endpoint)


@app.get("/sportradar/injuries")
def sportradar_injuries():
    """Get current MLB injury report"""
    endpoint = "league/injuries.json"
    return sportradar_request(endpoint)







