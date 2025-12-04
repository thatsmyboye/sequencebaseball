"""
Sequence Baseball REST API
FastAPI endpoints for pitch sequencing analysis and visualizations

Run with: uvicorn api.main:app --reload
API docs: http://localhost:8000/docs
"""

import sys
from pathlib import Path
from typing import Optional, List
from enum import Enum
import io

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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

# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Sequence Baseball API",
    description="REST API for MLB pitch sequencing analysis and visualizations",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allow frontend to call API
# Note: allow_credentials=True requires specific origins, not wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production to specific origins
    allow_credentials=False,  # Cannot use True with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Pitcher registry with multi-season support
# Each pitcher has available_seasons list with years they pitched in MLB (Statcast era: 2015+)
PITCHER_REGISTRY = {
    669373: {
        "name": "Tarik Skubal",
        "team": "DET",
        "team_full": "Detroit Tigers",
        "position": "SP",
        "throws": "L",
        "data_file": "skubal_statcast_all.csv",  # Multi-season file
        "data_file_2024": "skubal_statcast_2024.csv",  # Legacy single-season
        "pitch_types": ["4-Seam Fastball", "Slider", "Changeup"],
        "available_seasons": [2020, 2021, 2022, 2023, 2024],
        "season_notes": {
            2020: "Rookie debut (shortened COVID season)",
            2021: "First full season",
            2022: "Breakout year",
            2023: "Limited due to injury",
            2024: "Cy Young season"
        }
    },
    650556: {
        "name": "Jhoan Duran",
        "team": "MIN",
        "team_full": "Minnesota Twins",
        "position": "RP",
        "throws": "R",
        "data_file": "duran_statcast_all.csv",
        "data_file_2024": "duran_statcast_2024.csv",
        "pitch_types": ["4-Seam Fastball", "Splitter", "Slider"],
        "available_seasons": [2022, 2023, 2024],
        "season_notes": {
            2022: "Rookie season",
            2023: "All-Star breakout",
            2024: "Elite closer"
        }
    },
    554430: {
        "name": "Zack Wheeler",
        "team": "PHI",
        "team_full": "Philadelphia Phillies",
        "position": "SP",
        "throws": "R",
        "data_file": "wheeler_statcast_all.csv",
        "data_file_2024": "wheeler_statcast_2024.csv",
        "pitch_types": ["4-Seam Fastball", "Slider", "Curveball", "Changeup", "Sinker"],
        "available_seasons": [2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "season_notes": {
            2015: "With Mets",
            2017: "Return from injury",
            2018: "Strong comeback",
            2019: "Career year with Mets",
            2020: "First year with Phillies (COVID)",
            2021: "Cy Young runner-up",
            2022: "World Series run",
            2023: "Consistent ace",
            2024: "Elite performance"
        }
    },
    650911: {
        "name": "Cristopher Sanchez",
        "team": "PHI",
        "team_full": "Philadelphia Phillies",
        "position": "SP",
        "throws": "L",
        "data_file": "sanchez_statcast_all.csv",
        "data_file_2024": "sanchez_statcast_2024.csv",
        "pitch_types": ["Sinker", "Changeup", "Sweeper", "4-Seam Fastball"],
        "available_seasons": [2021, 2022, 2023, 2024],
        "season_notes": {
            2021: "Brief MLB debut",
            2022: "Spot starts/long relief",
            2023: "Rotation spot",
            2024: "Established starter"
        }
    },
}

DATA_DIR = Path(__file__).parent.parent / "data"


def load_pitcher_data(pitcher_id: int, season: Optional[int] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load pitcher data from CSV file with optional season/date filtering.
    
    Parameters:
    - pitcher_id: MLB pitcher ID
    - season: Optional season year to filter (e.g., 2024)
    - start_date: Optional start date (YYYY-MM-DD)
    - end_date: Optional end date (YYYY-MM-DD)
    """
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Pitcher ID {pitcher_id} not found. Available: {list(PITCHER_REGISTRY.keys())}"
        )
    
    pitcher_info = PITCHER_REGISTRY[pitcher_id]
    
    # Try multi-season file first, fall back to 2024-only file
    data_path = DATA_DIR / pitcher_info["data_file"]
    if not data_path.exists():
        data_path = DATA_DIR / pitcher_info.get("data_file_2024", pitcher_info["data_file"])
    
    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found for {pitcher_info['name']}. "
                   f"Expected path: {data_path}. "
                   f"DATA_DIR exists: {DATA_DIR.exists()}. "
                   f"Files in DATA_DIR: {list(DATA_DIR.glob('*')) if DATA_DIR.exists() else 'N/A'}"
        )
    
    df = pd.read_csv(data_path)
    
    # Filter by season if specified
    if season is not None:
        if 'season' in df.columns:
            df = df[df['season'] == season]
        else:
            # Fall back to extracting year from game_date
            df['game_date'] = pd.to_datetime(df['game_date'])
            df = df[df['game_date'].dt.year == season]
        
        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No data found for {pitcher_info['name']} in season {season}. "
                       f"Available seasons: {pitcher_info.get('available_seasons', ['unknown'])}"
            )
    
    # Filter by date range if specified
    if start_date or end_date:
        df['game_date'] = pd.to_datetime(df['game_date'])
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


class PitcherSeasonsResponse(BaseModel):
    pitcher_id: int
    pitcher_name: str
    available_seasons: List[SeasonInfo]


class TunnelRequest(BaseModel):
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    pitch_types: List[str] = Field(..., description="List of pitch types to visualize")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    max_pitches_per_type: int = Field(30, ge=5, le=100, description="Max pitches per type")
    season: Optional[int] = Field(None, ge=2015, le=2024, description="Season year (2015-2024)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class SequenceRequest(BaseModel):
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    min_sample_size: int = Field(10, ge=1, le=100, description="Minimum sample size")
    chart_type: ChartType = Field(ChartType.composite_score, description="Visualization type")
    top_n: int = Field(10, ge=3, le=20, description="Number of top sequences")
    sequence_position: str = Field("any", description="Sequence position: 'any', 'start', or 'end'")
    season: Optional[int] = Field(None, ge=2015, le=2024, description="Season year (2015-2024)")
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
    """Request model for interactive table endpoint (no chart_type needed)"""
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    min_sample_size: int = Field(10, ge=1, le=100, description="Minimum sample size")
    top_n: int = Field(10, ge=3, le=20, description="Number of top sequences")
    season: Optional[int] = Field(None, ge=2015, le=2024, description="Season year (2015-2024)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
def read_root():
    """API health check and version info"""
    return {
        "status": "ok",
        "api": "Sequence Baseball API",
        "version": "0.2.0",
        "features": ["multi-season support", "date range filtering"],
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for Railway/container orchestration"""
    return {"status": "healthy", "service": "sequence-baseball-api"}


@app.get("/pitchers", response_model=List[PitcherInfo], tags=["Pitchers"])
def list_pitchers():
    """
    List all available pitchers with their data
    
    Returns pitcher ID, name, team, available pitch types, and available seasons.
    """
    return [
        PitcherInfo(
            id=pid,
            name=info["name"],
            team=info["team"],
            team_full=info["team_full"],
            position=info["position"],
            throws=info["throws"],
            pitch_types=info["pitch_types"],
            available_seasons=info.get("available_seasons", [])
        )
        for pid, info in PITCHER_REGISTRY.items()
    ]


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
        pitch_types=info["pitch_types"],
        available_seasons=info.get("available_seasons", [])
    )


@app.get("/pitchers/{pitcher_id}/seasons", response_model=PitcherSeasonsResponse, tags=["Pitchers"])
def get_pitcher_seasons(pitcher_id: int):
    """
    Get available seasons for a pitcher with metadata.
    
    Returns list of seasons where the pitcher has MLB data (Statcast era: 2015+).
    Seasons where the pitcher did not pitch are not included.
    """
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    
    info = PITCHER_REGISTRY[pitcher_id]
    season_notes = info.get("season_notes", {})
    
    return PitcherSeasonsResponse(
        pitcher_id=pitcher_id,
        pitcher_name=info["name"],
        available_seasons=[
            SeasonInfo(season=s, note=season_notes.get(s))
            for s in sorted(info.get("available_seasons", []), reverse=True)
        ]
    )


@app.get("/pitchers/{pitcher_id}/pitch-types", tags=["Pitchers"])
def get_pitch_types(pitcher_id: int):
    """Get available pitch types for a pitcher"""
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    
    return {
        "pitcher_id": pitcher_id,
        "pitcher_name": PITCHER_REGISTRY[pitcher_id]["name"],
        "pitch_types": PITCHER_REGISTRY[pitcher_id]["pitch_types"]
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
    Generate 3D pitch tunnel visualization
    
    Returns interactive Plotly HTML that can be embedded in a webpage.
    Shows pitch trajectories from release point to home plate.
    
    Supports filtering by:
    - season: Single season year (2015-2024)
    - start_date/end_date: Date range within a season (YYYY-MM-DD)
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
        
        # Build title with season/date info
        title_parts = [pitcher_info["name"]]
        if request.season:
            title_parts.append(f"({request.season})")
        elif request.start_date or request.end_date:
            date_str = f"{request.start_date or 'start'} to {request.end_date or 'end'}"
            title_parts.append(f"({date_str})")
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
        
        # Return HTML string
        return fig.to_html(include_plotlyjs='cdn', full_html=True)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


@app.post("/visualizations/sequences", tags=["Visualizations"])
def generate_sequence_analysis(request: SequenceRequest):
    """
    Generate pitch sequence analysis
    
    Returns sequence data and optionally a visualization.
    Analyzes which pitch sequences are most effective.
    
    Supports filtering by:
    - season: Single season year (2015-2024)
    - start_date/end_date: Date range within a season (YYYY-MM-DD)
    - sequence_position: 'any', 'start' (first 2 pitches), or 'end' (last 2 pitches)
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
        
        # Analyze sequences with position filter
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
        
        # Get date range from filtered data
        date_range = None
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            date_range = {
                "start": str(df['game_date'].min().date()),
                "end": str(df['game_date'].max().date())
            }
        
        # Convert to response format
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
    Generate interactive HTML table for pitch sequence analysis
    
    Returns a sortable HTML table with sequence metrics.
    For PNG chart images, use /visualizations/sequences/chart-image instead.
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
        
        # Analyze sequences
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
        
        # Build score config if provided
        config = None
        if score_config:
            config = CompositeScoreConfig(
                whiff_weight=score_config.whiff_weight,
                chase_weight=score_config.chase_weight,
                weak_contact_weight=score_config.weak_contact_weight
            )
        
        # Generate interactive table (returns HTML)
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
    Generate sequence visualization chart as base64-encoded PNG
    
    Returns JSON with base64 image data that can be displayed with:
    <img src="data:image/png;base64,{image_data}">
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
        
        # Analyze sequences
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
        
        # Build score config if provided
        config = None
        if score_config:
            config = CompositeScoreConfig(
                whiff_weight=score_config.whiff_weight,
                chase_weight=score_config.chase_weight,
                weak_contact_weight=score_config.weak_contact_weight
            )
        
        # Generate matplotlib figure
        fig = create_sequence_visualization(
            sequence_df=seq_df,
            pitcher_name=pitcher_info["name"],
            chart_type=request.chart_type.value,
            batter_hand=batter_hand,
            top_n=request.top_n,
            score_config=config
        )
        
        # Convert to base64 PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        # Close figure to free memory
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
    season: Optional[int] = Query(None, ge=2015, le=2024, description="Season year"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get summary statistics for a pitcher's data with optional season/date filtering"""
    try:
        df = load_pitcher_data(pitcher_id, season=season, start_date=start_date, end_date=end_date)
        pitcher_info = PITCHER_REGISTRY[pitcher_id]
        
        # Calculate summary stats
        pitch_counts = df['pitch_name'].value_counts().to_dict()
        avg_velo_by_pitch = df.groupby('pitch_name')['release_speed'].mean().round(1).to_dict()
        
        total_pitches = len(df)
        # Swing types must match analyze_pitch_sequences definition
        swing_types = ['swinging_strike', 'foul', 'foul_tip', 'hit_into_play', 'swinging_strike_blocked']
        swings = df[df['description'].isin(swing_types)]
        whiffs = df[df['description'] == 'swinging_strike']
        whiff_rate = round(len(whiffs) / len(swings) * 100, 1) if len(swings) > 0 else 0
        
        # Get date range
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
            "available_seasons": pitcher_info.get("available_seasons", []),
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
    season: Optional[int] = Query(None, ge=2015, le=2024, description="Season year")
):
    """Get pitch movement profile data for visualization"""
    try:
        df = load_pitcher_data(pitcher_id, season=season)
        pitcher_info = PITCHER_REGISTRY[pitcher_id]
        
        # Calculate movement stats by pitch type
        movement_data = df.groupby('pitch_name').agg({
            'pfx_x': ['mean', 'std'],
            'pfx_z': ['mean', 'std'],
            'release_speed': 'mean'
        }).round(2)
        
        # Flatten column names
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
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Verify data files exist on startup"""
    print("\n" + "="*60)
    print("SEQUENCE BASEBALL API - Starting up...")
    print("="*60)
    
    for pitcher_id, info in PITCHER_REGISTRY.items():
        # Check for multi-season file first
        data_path = DATA_DIR / info["data_file"]
        if data_path.exists():
            status = "OK (multi-season)"
        else:
            # Fall back to 2024-only file
            data_path = DATA_DIR / info.get("data_file_2024", info["data_file"])
            status = "OK (2024 only)" if data_path.exists() else "MISSING"
        
        seasons_str = f"[{min(info.get('available_seasons', []))} - {max(info.get('available_seasons', []))}]" if info.get('available_seasons') else "[unknown]"
        print(f"  [{status}] {info['name']} ({pitcher_id}): {seasons_str}")
    
    print("="*60)
    print("API ready at http://localhost:8000")
    print("Interactive docs at http://localhost:8000/docs")
    print("="*60 + "\n")
