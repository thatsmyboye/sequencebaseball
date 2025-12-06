"""
Sequence Baseball REST API
FastAPI endpoints for pitch sequencing analysis and visualizations
Integrated with MLB 2025 season data via pybaseball

Run with: uvicorn api.main:app --reload
API docs: http://localhost:8000/docs
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
import io
import json
import logging

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitch_viz import visualize_pitch_trajectories_3d, analyze_pitch_sequences
from sequence_visualizations import (
    create_sequence_visualization,
    create_interactive_table,
    CompositeScoreConfig,
    get_available_chart_types
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Sequence Baseball API",
    description="REST API for MLB pitch sequencing analysis and visualizations - 2025 Season",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
REGISTRY_FILE = DATA_DIR / "pitcher_registry_2025.json"

# MLB Team mappings
MLB_TEAMS = {
    108: {"abbr": "LAA", "name": "Los Angeles Angels", "league": "AL", "division": "West"},
    109: {"abbr": "ARI", "name": "Arizona Diamondbacks", "league": "NL", "division": "West"},
    110: {"abbr": "BAL", "name": "Baltimore Orioles", "league": "AL", "division": "East"},
    111: {"abbr": "BOS", "name": "Boston Red Sox", "league": "AL", "division": "East"},
    112: {"abbr": "CHC", "name": "Chicago Cubs", "league": "NL", "division": "Central"},
    113: {"abbr": "CIN", "name": "Cincinnati Reds", "league": "NL", "division": "Central"},
    114: {"abbr": "CLE", "name": "Cleveland Guardians", "league": "AL", "division": "Central"},
    115: {"abbr": "COL", "name": "Colorado Rockies", "league": "NL", "division": "West"},
    116: {"abbr": "DET", "name": "Detroit Tigers", "league": "AL", "division": "Central"},
    117: {"abbr": "HOU", "name": "Houston Astros", "league": "AL", "division": "West"},
    118: {"abbr": "KC", "name": "Kansas City Royals", "league": "AL", "division": "Central"},
    119: {"abbr": "LAD", "name": "Los Angeles Dodgers", "league": "NL", "division": "West"},
    120: {"abbr": "WSH", "name": "Washington Nationals", "league": "NL", "division": "East"},
    121: {"abbr": "NYM", "name": "New York Mets", "league": "NL", "division": "East"},
    133: {"abbr": "OAK", "name": "Oakland Athletics", "league": "AL", "division": "West"},
    134: {"abbr": "PIT", "name": "Pittsburgh Pirates", "league": "NL", "division": "Central"},
    135: {"abbr": "SD", "name": "San Diego Padres", "league": "NL", "division": "West"},
    136: {"abbr": "SEA", "name": "Seattle Mariners", "league": "AL", "division": "West"},
    137: {"abbr": "SF", "name": "San Francisco Giants", "league": "NL", "division": "West"},
    138: {"abbr": "STL", "name": "St. Louis Cardinals", "league": "NL", "division": "Central"},
    139: {"abbr": "TB", "name": "Tampa Bay Rays", "league": "AL", "division": "East"},
    140: {"abbr": "TEX", "name": "Texas Rangers", "league": "AL", "division": "West"},
    141: {"abbr": "TOR", "name": "Toronto Blue Jays", "league": "AL", "division": "East"},
    142: {"abbr": "MIN", "name": "Minnesota Twins", "league": "AL", "division": "Central"},
    143: {"abbr": "PHI", "name": "Philadelphia Phillies", "league": "NL", "division": "East"},
    144: {"abbr": "ATL", "name": "Atlanta Braves", "league": "NL", "division": "East"},
    145: {"abbr": "CWS", "name": "Chicago White Sox", "league": "AL", "division": "Central"},
    146: {"abbr": "MIA", "name": "Miami Marlins", "league": "NL", "division": "East"},
    147: {"abbr": "NYY", "name": "New York Yankees", "league": "AL", "division": "East"},
    158: {"abbr": "MIL", "name": "Milwaukee Brewers", "league": "NL", "division": "Central"},
}

TEAM_ABBR_TO_ID = {info["abbr"]: team_id for team_id, info in MLB_TEAMS.items()}
TEAM_NAMES = [info["name"] for info in MLB_TEAMS.values()]
TEAM_ABBRS = list(TEAM_ABBR_TO_ID.keys())

# In-memory pitcher registry (loaded on startup)
PITCHER_REGISTRY: Dict[int, Dict] = {}


def load_pitcher_registry() -> Dict[int, Dict]:
    """Load pitcher registry from cache file or build it."""
    global PITCHER_REGISTRY
    
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, 'r') as f:
                PITCHER_REGISTRY = {int(k): v for k, v in json.load(f).items()}
                logger.info(f"Loaded pitcher registry with {len(PITCHER_REGISTRY)} pitchers")
                return PITCHER_REGISTRY
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
    
    # If no registry exists, try to build it
    try:
        from mlb_2025_integration import build_pitcher_registry
        PITCHER_REGISTRY = build_pitcher_registry(min_pitches=100)
        return PITCHER_REGISTRY
    except Exception as e:
        logger.warning(f"Could not build registry: {e}")
        return {}


def fetch_pitcher_data_live(pitcher_id: int, season: int = 2025) -> pd.DataFrame:
    """Fetch pitcher data from pybaseball (live API call)."""
    try:
        from pybaseball import statcast_pitcher
        from datetime import datetime
        
        # Season date ranges
        if season == 2025:
            start_date = "2025-03-27"
            end_date = datetime.now().strftime("%Y-%m-%d")
        else:
            start_date = f"{season}-03-20"
            end_date = f"{season}-10-15"
        
        logger.info(f"Fetching data for pitcher {pitcher_id}, season {season}")
        data = statcast_pitcher(start_date, end_date, pitcher_id)
        
        if data is not None and len(data) > 0:
            data['game_date'] = pd.to_datetime(data['game_date'])
            data['season'] = data['game_date'].dt.year
            
            # Cache the data
            cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{season}.csv"
            CACHE_DIR.mkdir(exist_ok=True)
            data.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(data)} pitches for pitcher {pitcher_id}")
        
        return data
        
    except ImportError:
        logger.error("pybaseball not installed")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def load_pitcher_data(pitcher_id: int, season: Optional[int] = 2025,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load pitcher data - from cache if available, otherwise fetch live.
    """
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Pitcher ID {pitcher_id} not found in registry"
        )
    
    pitcher_info = PITCHER_REGISTRY[pitcher_id]
    season = season or 2025
    
    # Check cache first
    cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{season}.csv"
    
    if cache_file.exists():
        logger.info(f"Loading cached data for pitcher {pitcher_id}")
        df = pd.read_csv(cache_file)
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        # Fetch live data
        df = fetch_pitcher_data_live(pitcher_id, season)
        
        if df is None or len(df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {pitcher_info['name']} in {season} season. "
                       f"Data may need to be fetched from MLB."
            )
    
    # Filter by date range if specified
    if start_date or end_date:
        if start_date:
            df = df[df['game_date'] >= start_date]
        if end_date:
            df = df[df['game_date'] <= end_date]
        
        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No data found for the specified date range"
            )
    
    return df


# =============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# =============================================================================

class BatterHand(str, Enum):
    right = "R"
    left = "L"


class ChartType(str, Enum):
    grouped_bar = "grouped_bar"
    composite_score = "composite_score"
    heatmap_matrix = "heatmap_matrix"
    scatter_bubble = "scatter_bubble"
    lollipop = "lollipop"
    small_multiples = "small_multiples"


class SeasonInfo(BaseModel):
    season: int
    note: Optional[str] = None


class PitcherInfo(BaseModel):
    id: int
    name: str
    team: str
    team_full: str
    position: str
    throws: str
    pitch_types: List[str]
    available_seasons: List[int] = []
    total_pitches: Optional[int] = None


class PitcherSearchResult(BaseModel):
    id: int
    name: str
    team: str
    team_full: str
    position: str
    throws: str
    match_type: Optional[str] = None  # "name" or "team"


class SearchResponse(BaseModel):
    query: str
    results: List[PitcherSearchResult]
    total_count: int
    is_team_search: bool = False


class TeamInfo(BaseModel):
    abbr: str
    name: str
    league: str
    division: str
    pitcher_count: int


class TunnelRequest(BaseModel):
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    pitch_types: List[str] = Field(..., description="List of pitch types to visualize")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    max_pitches_per_type: int = Field(30, ge=5, le=100, description="Max pitches per type")
    season: Optional[int] = Field(2025, ge=2015, le=2025, description="Season year")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class SequenceRequest(BaseModel):
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    min_sample_size: int = Field(10, ge=1, le=100, description="Minimum sample size")
    chart_type: ChartType = Field(ChartType.composite_score, description="Visualization type")
    top_n: int = Field(10, ge=3, le=20, description="Number of top sequences")
    sequence_position: str = Field("any", description="Sequence position: 'any', 'start', or 'end'")
    season: Optional[int] = Field(2025, ge=2015, le=2025, description="Season year")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class CompositeScoreRequest(BaseModel):
    whiff_weight: float = Field(0.40, ge=0, le=1, description="Weight for whiff rate")
    chase_weight: float = Field(0.35, ge=0, le=1, description="Weight for chase rate")
    weak_contact_weight: float = Field(0.25, ge=0, le=1, description="Weight for weak contact")


class SequenceData(BaseModel):
    sequence: str
    usage: int
    whiff_rate: float
    chase_rate: float
    weak_contact_rate: float
    overall_score: float


class SequenceAnalysisResponse(BaseModel):
    pitcher_name: str
    batter_hand: Optional[str]
    season: Optional[int] = None
    date_range: Optional[dict] = None
    sequences: List[SequenceData]
    total_sequences: int
    message: Optional[str] = None


class TableRequest(BaseModel):
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    min_sample_size: int = Field(10, ge=1, le=100, description="Minimum sample size")
    top_n: int = Field(10, ge=3, le=20, description="Number of top sequences")
    season: Optional[int] = Field(2025, ge=2015, le=2025, description="Season year")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def read_root():
    """Serve the main web application"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>Sequence Baseball API</h1><p>Frontend not found. Visit <a href='/docs'>/docs</a> for API.</p>")


@app.get("/api", tags=["Health"])
def api_info():
    """API health check and version info"""
    return {
        "status": "ok",
        "api": "Sequence Baseball API",
        "version": "1.0.0",
        "season": "2025",
        "features": ["2025 season data", "live data fetching", "team search", "alphabetized results"],
        "pitcher_count": len(PITCHER_REGISTRY),
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sequence-baseball-api", "pitcher_count": len(PITCHER_REGISTRY)}


@app.get("/pitchers", response_model=List[PitcherInfo], tags=["Pitchers"])
def list_pitchers(
    limit: int = Query(100, ge=1, le=500, description="Maximum number of pitchers to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List all available pitchers with their data (alphabetized by name).
    """
    # Sort alphabetically by name
    sorted_pitchers = sorted(
        PITCHER_REGISTRY.items(),
        key=lambda x: x[1]["name"].lower()
    )
    
    # Apply pagination
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
            available_seasons=info.get("available_seasons", [2025]),
            total_pitches=info.get("total_pitches")
        )
        for pid, info in paginated
    ]


@app.get("/pitchers/search", response_model=SearchResponse, tags=["Pitchers"])
def search_pitchers(
    q: str = Query(..., min_length=1, description="Search query (pitcher name or team)"),
    limit: int = Query(20, ge=1, le=50, description="Maximum results to return")
):
    """
    Search pitchers by name or team. Results are alphabetized.
    
    - Search by pitcher name: Returns matching pitchers
    - Search by team (abbr or full name): Returns all pitchers for that team
    """
    query_lower = q.lower().strip()
    
    # Detect if this is a team search
    is_team_search = False
    team_match = None
    
    # Check if query matches a team abbreviation exactly
    if query_lower.upper() in TEAM_ABBRS:
        is_team_search = True
        team_match = query_lower.upper()
    
    # Check if query matches a team name
    for team_id, team_info in MLB_TEAMS.items():
        if query_lower in team_info["name"].lower():
            is_team_search = True
            team_match = team_info["abbr"]
            break
    
    matches = []
    
    for pid, info in PITCHER_REGISTRY.items():
        match_type = None
        
        if is_team_search:
            # Match by team
            if team_match and info["team"] == team_match:
                match_type = "team"
        else:
            # Match by name
            if query_lower in info["name"].lower():
                match_type = "name"
            # Also allow team search within name search
            elif query_lower == info["team"].lower():
                match_type = "team"
            elif query_lower in info["team_full"].lower():
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
    
    # Sort alphabetically by name
    matches.sort(key=lambda x: x["name"].lower())
    
    return SearchResponse(
        query=q,
        results=[PitcherSearchResult(**m) for m in matches[:limit]],
        total_count=len(matches),
        is_team_search=is_team_search
    )


@app.get("/teams", response_model=List[TeamInfo], tags=["Teams"])
def list_teams():
    """
    List all MLB teams with pitcher counts.
    """
    # Count pitchers per team
    team_counts = {}
    for info in PITCHER_REGISTRY.values():
        team = info["team"]
        team_counts[team] = team_counts.get(team, 0) + 1
    
    teams = []
    for team_id, team_info in sorted(MLB_TEAMS.items(), key=lambda x: x[1]["name"]):
        abbr = team_info["abbr"]
        teams.append(TeamInfo(
            abbr=abbr,
            name=team_info["name"],
            league=team_info["league"],
            division=team_info["division"],
            pitcher_count=team_counts.get(abbr, 0)
        ))
    
    return teams


@app.get("/teams/{team}/pitchers", response_model=List[PitcherInfo], tags=["Teams"])
def get_team_pitchers(team: str):
    """
    Get all pitchers for a specific team (alphabetized).
    
    Team can be abbreviation (NYY) or full name search term (Yankees).
    """
    team_lower = team.lower().strip()
    
    # Find matching team
    target_abbr = None
    if team.upper() in TEAM_ABBRS:
        target_abbr = team.upper()
    else:
        for team_id, team_info in MLB_TEAMS.items():
            if team_lower in team_info["name"].lower():
                target_abbr = team_info["abbr"]
                break
    
    if not target_abbr:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found")
    
    # Get pitchers for this team
    pitchers = []
    for pid, info in PITCHER_REGISTRY.items():
        if info["team"] == target_abbr:
            pitchers.append(PitcherInfo(
                id=pid,
                name=info["name"],
                team=info["team"],
                team_full=info["team_full"],
                position=info["position"],
                throws=info["throws"],
                pitch_types=info.get("pitch_types", []),
                available_seasons=info.get("available_seasons", [2025]),
                total_pitches=info.get("total_pitches")
            ))
    
    # Sort alphabetically
    pitchers.sort(key=lambda x: x.name.lower())
    
    return pitchers


@app.get("/pitchers/{pitcher_id}", response_model=PitcherInfo, tags=["Pitchers"])
def get_pitcher(pitcher_id: int):
    """Get details for a specific pitcher"""
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    
    info = PITCHER_REGISTRY[pitcher_id]
    return PitcherInfo(
        id=pitcher_id,
        name=info["name"],
        team=info["team"],
        team_full=info["team_full"],
        position=info["position"],
        throws=info["throws"],
        pitch_types=info.get("pitch_types", []),
        available_seasons=info.get("available_seasons", [2025]),
        total_pitches=info.get("total_pitches")
    )


@app.get("/pitchers/{pitcher_id}/seasons", tags=["Pitchers"])
def get_pitcher_seasons(pitcher_id: int):
    """Get available seasons for a pitcher."""
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    
    info = PITCHER_REGISTRY[pitcher_id]
    seasons = info.get("available_seasons", [2025])
    
    return {
        "pitcher_id": pitcher_id,
        "pitcher_name": info["name"],
        "available_seasons": [{"season": s, "note": None} for s in sorted(seasons, reverse=True)]
    }


@app.get("/pitchers/{pitcher_id}/pitch-types", tags=["Pitchers"])
def get_pitch_types(pitcher_id: int):
    """Get available pitch types for a pitcher"""
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    
    return {
        "pitcher_id": pitcher_id,
        "pitcher_name": PITCHER_REGISTRY[pitcher_id]["name"],
        "pitch_types": PITCHER_REGISTRY[pitcher_id].get("pitch_types", [])
    }


@app.get("/chart-types", tags=["Visualizations"])
def list_chart_types():
    """List available visualization chart types"""
    return {
        "chart_types": get_available_chart_types(),
        "default": "composite_score"
    }


# =============================================================================
# VISUALIZATION ENDPOINTS
# =============================================================================

@app.post("/visualizations/tunnel", response_class=HTMLResponse, tags=["Visualizations"])
def generate_tunnel_visualization(request: TunnelRequest):
    """
    Generate 3D pitch tunnel visualization.
    Returns interactive Plotly HTML.
    """
    try:
        df = load_pitcher_data(
            request.pitcher_id,
            season=request.season,
            start_date=request.start_date,
            end_date=request.end_date
        )
        pitcher_info = PITCHER_REGISTRY[request.pitcher_id]
        
        # Validate pitch types
        available_types = df['pitch_name'].unique().tolist()
        invalid_types = [pt for pt in request.pitch_types if pt not in available_types]
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pitch types: {invalid_types}. Available: {available_types}"
            )
        
        # Build title
        title_parts = [pitcher_info["name"]]
        if request.season:
            title_parts.append(f"({request.season})")
        title = " ".join(title_parts)
        
        # Generate visualization
        fig = visualize_pitch_trajectories_3d(
            df=df,
            pitcher_name=title,
            pitch_types=request.pitch_types,
            batter_hand=request.batter_hand.value if request.batter_hand else None,
            max_pitches_per_type=request.max_pitches_per_type,
            output_html=None,
            output_png=None
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=True)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


@app.post("/visualizations/sequences", tags=["Visualizations"])
def generate_sequence_analysis(request: SequenceRequest):
    """
    Generate pitch sequence analysis.
    """
    try:
        df = load_pitcher_data(
            request.pitcher_id,
            season=request.season,
            start_date=request.start_date,
            end_date=request.end_date
        )
        pitcher_info = PITCHER_REGISTRY[request.pitcher_id]
        
        batter_hand = request.batter_hand.value if request.batter_hand else None
        
        seq_df = analyze_pitch_sequences(
            df=df,
            pitcher_name=pitcher_info["name"],
            batter_hand=batter_hand,
            min_sample_size=request.min_sample_size,
            success_metric='overall',
            sequence_position=request.sequence_position
        )
        
        if len(seq_df) == 0:
            return SequenceAnalysisResponse(
                pitcher_name=pitcher_info["name"],
                batter_hand=batter_hand,
                season=request.season,
                sequences=[],
                total_sequences=0,
                message="No sequences found with specified criteria"
            )
        
        date_range = None
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            date_range = {
                "start": str(df['game_date'].min().date()),
                "end": str(df['game_date'].max().date())
            }
        
        sequences = [
            SequenceData(
                sequence=row['Sequence'],
                usage=int(row['Usage']),
                whiff_rate=row['Whiff Rate'],
                chase_rate=row['Chase Rate'],
                weak_contact_rate=row['Weak Contact Rate'],
                overall_score=row['Overall Score']
            )
            for _, row in seq_df.head(request.top_n).iterrows()
        ]
        
        return SequenceAnalysisResponse(
            pitcher_name=pitcher_info["name"],
            batter_hand=batter_hand,
            season=request.season,
            date_range=date_range,
            sequences=sequences,
            total_sequences=len(seq_df)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/visualizations/sequences/chart", response_class=HTMLResponse, tags=["Visualizations"])
def generate_sequence_chart(
    request: TableRequest,
    score_config: Optional[CompositeScoreRequest] = Body(None)
):
    """
    Generate interactive HTML table for pitch sequence analysis.
    """
    try:
        df = load_pitcher_data(
            request.pitcher_id,
            season=request.season,
            start_date=request.start_date,
            end_date=request.end_date
        )
        pitcher_info = PITCHER_REGISTRY[request.pitcher_id]
        
        batter_hand = request.batter_hand.value if request.batter_hand else None
        
        seq_df = analyze_pitch_sequences(
            df=df,
            pitcher_name=pitcher_info["name"],
            batter_hand=batter_hand,
            min_sample_size=request.min_sample_size,
            success_metric='overall'
        )
        
        if len(seq_df) == 0:
            raise HTTPException(
                status_code=400,
                detail="No sequences found with specified criteria"
            )
        
        config = None
        if score_config:
            config = CompositeScoreConfig(
                whiff_weight=score_config.whiff_weight,
                chase_weight=score_config.chase_weight,
                weak_contact_weight=score_config.weak_contact_weight
            )
        
        html_content = create_interactive_table(
            sequence_df=seq_df.head(request.top_n),
            pitcher_name=pitcher_info["name"],
            batter_hand=batter_hand,
            score_config=config
        )
        
        return html_content
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation error: {str(e)}")


@app.post("/visualizations/sequences/chart-image", tags=["Visualizations"])
def generate_sequence_chart_image(
    request: SequenceRequest,
    score_config: Optional[CompositeScoreRequest] = Body(None)
):
    """
    Generate sequence visualization chart as base64-encoded PNG.
    """
    import base64
    
    try:
        df = load_pitcher_data(
            request.pitcher_id,
            season=request.season,
            start_date=request.start_date,
            end_date=request.end_date
        )
        pitcher_info = PITCHER_REGISTRY[request.pitcher_id]
        
        batter_hand = request.batter_hand.value if request.batter_hand else None
        
        seq_df = analyze_pitch_sequences(
            df=df,
            pitcher_name=pitcher_info["name"],
            batter_hand=batter_hand,
            min_sample_size=request.min_sample_size,
            success_metric='overall'
        )
        
        if len(seq_df) == 0:
            raise HTTPException(
                status_code=400,
                detail="No sequences found with specified criteria"
            )
        
        config = None
        if score_config:
            config = CompositeScoreConfig(
                whiff_weight=score_config.whiff_weight,
                chase_weight=score_config.chase_weight,
                weak_contact_weight=score_config.weak_contact_weight
            )
        
        fig = create_sequence_visualization(
            sequence_df=seq_df,
            pitcher_name=pitcher_info["name"],
            chart_type=request.chart_type.value,
            batter_hand=batter_hand,
            top_n=request.top_n,
            score_config=config
        )
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return {
            "pitcher_name": pitcher_info["name"],
            "chart_type": request.chart_type.value,
            "batter_hand": batter_hand,
            "season": request.season,
            "image_format": "png",
            "image_data": image_base64
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation error: {str(e)}")


# =============================================================================
# RAW DATA ENDPOINTS
# =============================================================================

@app.get("/data/{pitcher_id}/summary", tags=["Data"])
def get_pitcher_summary(
    pitcher_id: int,
    season: Optional[int] = Query(2025, ge=2015, le=2025, description="Season year"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get summary statistics for a pitcher's data"""
    try:
        df = load_pitcher_data(pitcher_id, season=season, start_date=start_date, end_date=end_date)
        pitcher_info = PITCHER_REGISTRY[pitcher_id]
        
        pitch_counts = df['pitch_name'].value_counts().to_dict()
        avg_velo_by_pitch = df.groupby('pitch_name')['release_speed'].mean().round(1).to_dict()
        
        total_pitches = len(df)
        swing_types = ['swinging_strike', 'foul', 'foul_tip', 'hit_into_play', 'swinging_strike_blocked']
        swings = df[df['description'].isin(swing_types)]
        whiffs = df[df['description'] == 'swinging_strike']
        whiff_rate = round(len(whiffs) / len(swings) * 100, 1) if len(swings) > 0 else 0
        
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        return {
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher_info["name"],
            "team": pitcher_info["team"],
            "season": season,
            "total_pitches": total_pitches,
            "pitch_arsenal": pitch_counts,
            "avg_velocity_by_pitch": avg_velo_by_pitch,
            "overall_whiff_rate": whiff_rate,
            "available_seasons": pitcher_info.get("available_seasons", [2025]),
            "date_range": {
                "start": str(df['game_date'].min().date()),
                "end": str(df['game_date'].max().date())
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/{pitcher_id}/movement", tags=["Data"])
def get_movement_profile(
    pitcher_id: int,
    season: Optional[int] = Query(2025, ge=2015, le=2025, description="Season year")
):
    """Get pitch movement profile data"""
    try:
        df = load_pitcher_data(pitcher_id, season=season)
        pitcher_info = PITCHER_REGISTRY[pitcher_id]
        
        movement_data = df.groupby('pitch_name').agg({
            'pfx_x': ['mean', 'std'],
            'pfx_z': ['mean', 'std'],
            'release_speed': 'mean'
        }).round(2)
        
        movement_data.columns = ['h_break_avg', 'h_break_std', 'v_break_avg', 'v_break_std', 'avg_velo']
        movement_data = movement_data.reset_index()
        
        return {
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher_info["name"],
            "season": season,
            "movement": movement_data.to_dict(orient='records')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DATA MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/data/{pitcher_id}/refresh", tags=["Data"])
def refresh_pitcher_data(pitcher_id: int, season: int = 2025):
    """
    Force refresh pitcher data from MLB API.
    Use this to get the latest data for a pitcher.
    """
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    
    pitcher_info = PITCHER_REGISTRY[pitcher_id]
    
    # Remove cached file if exists
    cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{season}.csv"
    if cache_file.exists():
        cache_file.unlink()
    
    # Fetch fresh data
    df = fetch_pitcher_data_live(pitcher_id, season)
    
    if df is None or len(df) == 0:
        raise HTTPException(
            status_code=503,
            detail=f"Could not fetch data for {pitcher_info['name']}. MLB API may be unavailable."
        )
    
    return {
        "status": "success",
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_info["name"],
        "season": season,
        "pitches_fetched": len(df)
    }


@app.post("/registry/refresh", tags=["Admin"])
def refresh_pitcher_registry():
    """
    Refresh the pitcher registry from MLB data.
    Warning: This may take several minutes.
    """
    global PITCHER_REGISTRY
    
    try:
        from mlb_2025_integration import build_pitcher_registry
        
        # Remove old registry file
        if REGISTRY_FILE.exists():
            REGISTRY_FILE.unlink()
        
        PITCHER_REGISTRY = build_pitcher_registry(min_pitches=100)
        
        return {
            "status": "success",
            "pitcher_count": len(PITCHER_REGISTRY),
            "message": "Registry refreshed successfully"
        }
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="pybaseball not installed. Cannot refresh registry."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing registry: {str(e)}"
        )


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load pitcher registry on startup"""
    global PITCHER_REGISTRY
    
    print("\n" + "=" * 60)
    print("SEQUENCE BASEBALL API - 2025 Season")
    print("=" * 60)
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Load pitcher registry
    PITCHER_REGISTRY = load_pitcher_registry()
    
    if PITCHER_REGISTRY:
        print(f"\n✓ Loaded {len(PITCHER_REGISTRY)} pitchers")
        
        # Count by team
        team_counts = {}
        for info in PITCHER_REGISTRY.values():
            team = info["team"]
            team_counts[team] = team_counts.get(team, 0) + 1
        
        print(f"✓ Teams represented: {len(team_counts)}")
        print(f"✓ Top teams: {sorted(team_counts.items(), key=lambda x: -x[1])[:5]}")
    else:
        print("\n⚠️  No pitcher registry loaded!")
        print("   Run: POST /registry/refresh to build registry from MLB data")
        print("   Or ensure pybaseball is installed: pip install pybaseball")
    
    print("\n" + "=" * 60)
    print("API ready at http://localhost:8000")
    print("Interactive docs at http://localhost:8000/docs")
    print("=" * 60 + "\n")
