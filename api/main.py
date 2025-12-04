"""
Sequence Baseball REST API
FastAPI endpoints for pitch sequencing analysis and visualizations

Run with: uvicorn api.main:app --reload
API docs: http://localhost:8000/docs
Frontend: http://localhost:8000/app
"""

import sys
from pathlib import Path
from typing import Optional, List
from enum import Enum
import io

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

# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Sequence Baseball API",
    description="REST API for MLB pitch sequencing analysis and visualizations",
    version="0.1.0",
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

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Pitcher registry - maps pitcher info to data files
PITCHER_REGISTRY = {
    669373: {
        "name": "Tarik Skubal",
        "team": "DET",
        "team_full": "Detroit Tigers",
        "position": "SP",
        "throws": "L",
        "data_file": "skubal_statcast_2024.csv",
        "pitch_types": ["4-Seam Fastball", "Slider", "Changeup", "Curveball"]
    },
    650556: {
        "name": "Jhoan Duran",
        "team": "MIN",
        "team_full": "Minnesota Twins",
        "position": "RP",
        "throws": "R",
        "data_file": "duran_statcast_2024.csv",
        "pitch_types": ["4-Seam Fastball", "Splitter", "Slider", "Sinker"]
    },
    # Add more pitchers as data becomes available
}

DATA_DIR = Path(__file__).parent.parent / "data"


def load_pitcher_data(pitcher_id: int) -> pd.DataFrame:
    """Load pitcher data from CSV file"""
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Pitcher ID {pitcher_id} not found. Available: {list(PITCHER_REGISTRY.keys())}"
        )
    
    pitcher_info = PITCHER_REGISTRY[pitcher_id]
    data_path = DATA_DIR / pitcher_info["data_file"]
    
    if not data_path.exists():
        # Provide helpful debug info for serverless environments
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found for {pitcher_info['name']}. "
                   f"Expected path: {data_path}. "
                   f"DATA_DIR exists: {DATA_DIR.exists()}. "
                   f"Files in DATA_DIR: {list(DATA_DIR.glob('*')) if DATA_DIR.exists() else 'N/A'}"
        )
    
    return pd.read_csv(data_path)


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


class PitcherInfo(BaseModel):
    id: int
    name: str
    team: str
    team_full: str
    position: str
    throws: str
    pitch_types: List[str]


class TunnelRequest(BaseModel):
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    pitch_types: List[str] = Field(..., description="List of pitch types to visualize")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    max_pitches_per_type: int = Field(30, ge=5, le=100, description="Max pitches per type")


class SequenceRequest(BaseModel):
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    min_sample_size: int = Field(10, ge=1, le=100, description="Minimum sample size")
    chart_type: ChartType = Field(ChartType.composite_score, description="Visualization type")
    top_n: int = Field(10, ge=3, le=20, description="Number of top sequences")


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
    sequences: List[SequenceData]
    total_sequences: int
    message: Optional[str] = None


class TableRequest(BaseModel):
    """Request model for interactive table endpoint (no chart_type needed)"""
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    batter_hand: Optional[BatterHand] = Field(None, description="Filter by batter handedness")
    min_sample_size: int = Field(10, ge=1, le=100, description="Minimum sample size")
    top_n: int = Field(10, ge=3, le=20, description="Number of top sequences")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
def read_root():
    """API health check and version info"""
    return {
        "status": "ok",
        "api": "Sequence Baseball API",
        "version": "0.1.0",
        "docs": "/docs",
        "app": "/app"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for Railway/container orchestration"""
    return {"status": "healthy", "service": "sequence-baseball-api"}


@app.get("/app", response_class=HTMLResponse, tags=["Frontend"])
def serve_frontend():
    """
    Serve the interactive web application
    
    This is the main user interface for exploring pitch sequencing data.
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    
    with open(index_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


@app.get("/pitchers", response_model=List[PitcherInfo], tags=["Pitchers"])
def list_pitchers():
    """
    List all available pitchers with their data
    
    Returns pitcher ID, name, team, and available pitch types.
    """
    return [
        PitcherInfo(id=pid, **{k: v for k, v in info.items() if k != "data_file"})
        for pid, info in PITCHER_REGISTRY.items()
    ]


@app.get("/pitchers/{pitcher_id}", response_model=PitcherInfo, tags=["Pitchers"])
def get_pitcher(pitcher_id: int):
    """Get details for a specific pitcher"""
    if pitcher_id not in PITCHER_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    
    info = PITCHER_REGISTRY[pitcher_id]
    return PitcherInfo(id=pitcher_id, **{k: v for k, v in info.items() if k != "data_file"})


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
    """
    try:
        df = load_pitcher_data(request.pitcher_id)
        pitcher_info = PITCHER_REGISTRY[request.pitcher_id]
        
        # Validate pitch types
        available_types = df['pitch_name'].unique().tolist()
        invalid_types = [pt for pt in request.pitch_types if pt not in available_types]
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pitch types: {invalid_types}. Available: {available_types}"
            )
        
        # Generate visualization
        fig = visualize_pitch_trajectories_3d(
            df=df,
            pitcher_name=pitcher_info["name"],
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
    """
    try:
        df = load_pitcher_data(request.pitcher_id)
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
            return SequenceAnalysisResponse(
                pitcher_name=pitcher_info["name"],
                batter_hand=batter_hand,
                sequences=[],
                total_sequences=0,
                message="No sequences found with specified criteria"
            )
        
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
        df = load_pitcher_data(request.pitcher_id)
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
        df = load_pitcher_data(request.pitcher_id)
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
def get_pitcher_summary(pitcher_id: int):
    """Get summary statistics for a pitcher's data"""
    try:
        df = load_pitcher_data(pitcher_id)
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
        
        return {
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher_info["name"],
            "team": pitcher_info["team"],
            "total_pitches": total_pitches,
            "pitch_arsenal": pitch_counts,
            "avg_velocity_by_pitch": avg_velo_by_pitch,
            "overall_whiff_rate": whiff_rate,
            "date_range": {
                "start": str(df['game_date'].min()),
                "end": str(df['game_date'].max())
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/{pitcher_id}/movement", tags=["Data"])
def get_movement_profile(pitcher_id: int):
    """Get pitch movement profile data for visualization"""
    try:
        df = load_pitcher_data(pitcher_id)
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
        data_path = DATA_DIR / info["data_file"]
        status = "OK" if data_path.exists() else "MISSING"
        print(f"  [{status}] {info['name']} ({pitcher_id}): {info['data_file']}")
    
    # Check static files
    static_status = "OK" if (STATIC_DIR / "index.html").exists() else "MISSING"
    print(f"  [{static_status}] Frontend: api/static/index.html")
    
    print("="*60)
    print("🎯 Web App:       http://localhost:8000/app")
    print("📚 API Docs:      http://localhost:8000/docs")
    print("🔧 Health Check:  http://localhost:8000/health")
    print("="*60 + "\n")

