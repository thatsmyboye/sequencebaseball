"""
Sequence Baseball REST API - LITE VERSION
Optimized for Railway/low-memory deployment

This version:
- Uses pre-built pitcher registry (no pybaseball at startup)
- Lazy loads pitcher data only when requested
- Reduced memory footprint

Run with: uvicorn api.main_lite:app --reload
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict
from enum import Enum
import io
import json
import logging

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Sequence Baseball API",
    description="REST API for MLB pitch sequencing analysis - 2024 Season (Lite)",
    version="1.0.0-lite",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

# Find the most recent registry file
def get_registry_file():
    """Find the most recent pitcher registry file."""
    registry_files = list(DATA_DIR.glob("pitcher_registry_*.json"))
    if registry_files:
        return sorted(registry_files, reverse=True)[0]
    return DATA_DIR / "pitcher_registry_2024.json"

REGISTRY_FILE = get_registry_file()

# MLB Teams
MLB_TEAMS = {
    108: {"abbr": "LAA", "name": "Los Angeles Angels"},
    109: {"abbr": "ARI", "name": "Arizona Diamondbacks"},
    110: {"abbr": "BAL", "name": "Baltimore Orioles"},
    111: {"abbr": "BOS", "name": "Boston Red Sox"},
    112: {"abbr": "CHC", "name": "Chicago Cubs"},
    113: {"abbr": "CIN", "name": "Cincinnati Reds"},
    114: {"abbr": "CLE", "name": "Cleveland Guardians"},
    115: {"abbr": "COL", "name": "Colorado Rockies"},
    116: {"abbr": "DET", "name": "Detroit Tigers"},
    117: {"abbr": "HOU", "name": "Houston Astros"},
    118: {"abbr": "KC", "name": "Kansas City Royals"},
    119: {"abbr": "LAD", "name": "Los Angeles Dodgers"},
    120: {"abbr": "WSH", "name": "Washington Nationals"},
    121: {"abbr": "NYM", "name": "New York Mets"},
    133: {"abbr": "OAK", "name": "Oakland Athletics"},
    134: {"abbr": "PIT", "name": "Pittsburgh Pirates"},
    135: {"abbr": "SD", "name": "San Diego Padres"},
    136: {"abbr": "SEA", "name": "Seattle Mariners"},
    137: {"abbr": "SF", "name": "San Francisco Giants"},
    138: {"abbr": "STL", "name": "St. Louis Cardinals"},
    139: {"abbr": "TB", "name": "Tampa Bay Rays"},
    140: {"abbr": "TEX", "name": "Texas Rangers"},
    141: {"abbr": "TOR", "name": "Toronto Blue Jays"},
    142: {"abbr": "MIN", "name": "Minnesota Twins"},
    143: {"abbr": "PHI", "name": "Philadelphia Phillies"},
    144: {"abbr": "ATL", "name": "Atlanta Braves"},
    145: {"abbr": "CWS", "name": "Chicago White Sox"},
    146: {"abbr": "MIA", "name": "Miami Marlins"},
    147: {"abbr": "NYY", "name": "New York Yankees"},
    158: {"abbr": "MIL", "name": "Milwaukee Brewers"},
}

TEAM_ABBRS = [info["abbr"] for info in MLB_TEAMS.values()]

# Pitcher registry (loaded lazily)
_PITCHER_REGISTRY: Optional[Dict[int, Dict]] = None


def get_pitcher_registry() -> Dict[int, Dict]:
    """Load pitcher registry lazily."""
    global _PITCHER_REGISTRY
    
    if _PITCHER_REGISTRY is None:
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE, 'r') as f:
                _PITCHER_REGISTRY = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"Loaded {len(_PITCHER_REGISTRY)} pitchers from registry")
        else:
            logger.warning("No pitcher registry found!")
            _PITCHER_REGISTRY = {}
    
    return _PITCHER_REGISTRY


def load_pitcher_data(pitcher_id: int, season: int = 2024) -> pd.DataFrame:
    """Load pitcher data from cache or fetch live."""
    registry = get_pitcher_registry()
    
    if pitcher_id not in registry:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    
    # Check cache
    cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{season}.csv"
    
    if cache_file.exists():
        return pd.read_csv(cache_file)
    
    # Try to fetch live (only if pybaseball available)
    try:
        from pybaseball import statcast_pitcher
        from datetime import datetime
        
        start_date = f"{season}-03-27"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        data = statcast_pitcher(start_date, end_date, pitcher_id)
        
        if data is not None and len(data) > 0:
            CACHE_DIR.mkdir(exist_ok=True)
            data.to_csv(cache_file, index=False)
            return data
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
    
    raise HTTPException(
        status_code=503,
        detail=f"Data not available. Pre-cache data or ensure pybaseball is installed."
    )


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PitcherInfo(BaseModel):
    id: int
    name: str
    team: str
    team_full: str
    position: str
    throws: str
    pitch_types: List[str] = []
    available_seasons: List[int] = [2024]


class SearchResponse(BaseModel):
    query: str
    results: List[dict]
    total_count: int
    is_team_search: bool = False


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def read_root():
    """Serve the main web application"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>Sequence Baseball API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/api", tags=["Health"])
def api_info():
    """API health check"""
    registry = get_pitcher_registry()
    return {
        "status": "ok",
        "api": "Sequence Baseball API (Lite)",
        "version": "1.0.0-lite",
        "pitcher_count": len(registry),
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check for Railway"""
    return {"status": "healthy"}


@app.get("/pitchers", response_model=List[PitcherInfo], tags=["Pitchers"])
def list_pitchers(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """List all pitchers (alphabetized)"""
    registry = get_pitcher_registry()
    
    sorted_pitchers = sorted(
        registry.items(),
        key=lambda x: x[1]["name"].lower()
    )
    
    paginated = sorted_pitchers[offset:offset + limit]
    
    return [
        PitcherInfo(
            id=pid,
            name=info["name"],
            team=info["team"],
            team_full=info["team_full"],
            position=info["position"],
            throws=info["throws"],
            pitch_types=info.get("pitch_types", []),
            available_seasons=info.get("available_seasons", [2024])
        )
        for pid, info in paginated
    ]


@app.get("/pitchers/search", response_model=SearchResponse, tags=["Pitchers"])
def search_pitchers(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=50)
):
    """Search pitchers by name or team (alphabetized results)"""
    registry = get_pitcher_registry()
    query_lower = q.lower().strip()
    
    # Detect team search
    is_team_search = False
    team_match = None
    
    if query_lower.upper() in TEAM_ABBRS:
        is_team_search = True
        team_match = query_lower.upper()
    else:
        for team_id, team_info in MLB_TEAMS.items():
            if query_lower in team_info["name"].lower():
                is_team_search = True
                team_match = team_info["abbr"]
                break
    
    matches = []
    for pid, info in registry.items():
        match_type = None
        
        if is_team_search:
            if team_match and info["team"] == team_match:
                match_type = "team"
        else:
            if query_lower in info["name"].lower():
                match_type = "name"
            elif query_lower == info["team"].lower():
                match_type = "team"
        
        if match_type:
            matches.append({
                "id": pid,
                "name": info["name"],
                "team": info["team"],
                "team_full": info["team_full"],
                "position": info["position"],
                "throws": info["throws"],
                "match_type": match_type
            })
    
    # Sort alphabetically
    matches.sort(key=lambda x: x["name"].lower())
    
    return SearchResponse(
        query=q,
        results=matches[:limit],
        total_count=len(matches),
        is_team_search=is_team_search
    )


@app.get("/pitchers/{pitcher_id}", response_model=PitcherInfo, tags=["Pitchers"])
def get_pitcher(pitcher_id: int):
    """Get specific pitcher details"""
    registry = get_pitcher_registry()
    
    if pitcher_id not in registry:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    info = registry[pitcher_id]
    return PitcherInfo(
        id=pitcher_id,
        name=info["name"],
        team=info["team"],
        team_full=info["team_full"],
        position=info["position"],
        throws=info["throws"],
        pitch_types=info.get("pitch_types", []),
        available_seasons=info.get("available_seasons", [2024])
    )


@app.get("/teams", tags=["Teams"])
def list_teams():
    """List all teams with pitcher counts"""
    registry = get_pitcher_registry()
    
    team_counts = {}
    for info in registry.values():
        team = info["team"]
        team_counts[team] = team_counts.get(team, 0) + 1
    
    return [
        {
            "abbr": info["abbr"],
            "name": info["name"],
            "pitcher_count": team_counts.get(info["abbr"], 0)
        }
        for info in sorted(MLB_TEAMS.values(), key=lambda x: x["name"])
    ]


@app.get("/teams/{team}/pitchers", tags=["Teams"])
def get_team_pitchers(team: str):
    """Get all pitchers for a team"""
    registry = get_pitcher_registry()
    
    team_upper = team.upper()
    target_abbr = team_upper if team_upper in TEAM_ABBRS else None
    
    if not target_abbr:
        for info in MLB_TEAMS.values():
            if team.lower() in info["name"].lower():
                target_abbr = info["abbr"]
                break
    
    if not target_abbr:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found")
    
    pitchers = [
        {"id": pid, **info}
        for pid, info in registry.items()
        if info["team"] == target_abbr
    ]
    
    return sorted(pitchers, key=lambda x: x["name"].lower())


@app.get("/data/{pitcher_id}/summary", tags=["Data"])
def get_pitcher_summary(pitcher_id: int, season: int = 2024):
    """Get pitcher summary statistics"""
    registry = get_pitcher_registry()
    if pitcher_id not in registry:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    info = registry[pitcher_id]
    
    try:
        df = load_pitcher_data(pitcher_id, season)
        
        # Handle missing columns gracefully
        pitch_counts = {}
        avg_velo = {}
        
        if 'pitch_name' in df.columns:
            pitch_counts = df['pitch_name'].value_counts().to_dict()
        if 'pitch_name' in df.columns and 'release_speed' in df.columns:
            avg_velo = df.groupby('pitch_name')['release_speed'].mean().round(1).to_dict()
        
        return {
            "pitcher_id": pitcher_id,
            "pitcher_name": info["name"],
            "team": info["team"],
            "season": season,
            "total_pitches": len(df),
            "pitch_arsenal": pitch_counts,
            "avg_velocity_by_pitch": avg_velo,
            "available_seasons": info.get("available_seasons", [2024])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summary for pitcher {pitcher_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualizations/sequences", tags=["Visualizations"])
def generate_sequences(
    pitcher_id: int = Body(...),
    batter_hand: Optional[str] = Body(None),
    min_sample_size: int = Body(10),
    top_n: int = Body(15),
    season: int = Body(2024)
):
    """Generate sequence analysis"""
    # Import with error handling
    try:
        from pitch_viz import analyze_pitch_sequences
    except ImportError as e:
        logger.error(f"Failed to import pitch_viz: {e}")
        raise HTTPException(status_code=503, detail="Pitch analysis module not available")
    
    registry = get_pitcher_registry()
    if pitcher_id not in registry:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    info = registry[pitcher_id]
    
    try:
        df = load_pitcher_data(pitcher_id, season)
        
        seq_df = analyze_pitch_sequences(
            df=df,
            pitcher_name=info["name"],
            batter_hand=batter_hand,
            min_sample_size=min_sample_size,
            success_metric='overall'
        )
        
        if seq_df is None or len(seq_df) == 0:
            return {"pitcher_name": info["name"], "sequences": [], "total_sequences": 0, "season": season}
        
        sequences = [
            {
                "sequence": row['Sequence'],
                "usage": int(row['Usage']),
                "whiff_rate": row['Whiff Rate'],
                "chase_rate": row['Chase Rate'],
                "weak_contact_rate": row['Weak Contact Rate'],
                "overall_score": row['Overall Score']
            }
            for _, row in seq_df.head(top_n).iterrows()
        ]
        
        return {
            "pitcher_name": info["name"],
            "sequences": sequences,
            "total_sequences": len(seq_df),
            "season": season
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating sequences for pitcher {pitcher_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualizations/tunnel", response_class=HTMLResponse, tags=["Visualizations"])
def generate_tunnel(
    pitcher_id: int = Body(...),
    pitch_types: List[str] = Body(...),
    batter_hand: Optional[str] = Body(None),
    max_pitches_per_type: int = Body(30),
    season: int = Body(2024)
):
    """Generate 3D tunnel visualization"""
    # Import with error handling
    try:
        from pitch_viz import visualize_pitch_trajectories_3d
    except ImportError as e:
        logger.error(f"Failed to import pitch_viz: {e}")
        raise HTTPException(status_code=503, detail="Visualization module not available")
    
    registry = get_pitcher_registry()
    if pitcher_id not in registry:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    info = registry[pitcher_id]
    
    try:
        df = load_pitcher_data(pitcher_id, season)
        
        fig = visualize_pitch_trajectories_3d(
            df=df,
            pitcher_name=info["name"],
            pitch_types=pitch_types,
            batter_hand=batter_hand,
            max_pitches_per_type=max_pitches_per_type
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating tunnel for pitcher {pitcher_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    registry = get_pitcher_registry()
    
    print("\n" + "=" * 50)
    print("SEQUENCE BASEBALL API (LITE)")
    print("=" * 50)
    print(f"Pitchers loaded: {len(registry)}")
    print("=" * 50 + "\n")
