# Sequence Baseball - 2025 MLB Pitch Analysis

A comprehensive system for analyzing and visualizing MLB pitcher data using Statcast metrics. Features 3D pitch trajectory visualization, sequence analysis, and live data integration for the **2025 MLB season**.

## 🎯 Overview

This project analyzes pitch sequencing strategies and creates portfolio-ready visualizations for **all MLB pitchers** with data from the 2025 season. Built with Python, FastAPI, Plotly, and matplotlib, it provides:

- **Live 2025 Data**: Automatically fetches current season data from Baseball Savant via pybaseball
- **All Pitchers**: Search and analyze any pitcher with 100+ pitches in 2025
- **Team Search**: Find all pitchers for a specific team
- **Alphabetized Search**: Results sorted A-Z for easy navigation
- **De-duped Registry**: Clean, distinct pitcher records

---

## ✨ Features

### 🔍 Pitcher Search
- Search by **pitcher name** (e.g., "Skubal", "Wheeler")
- Search by **team** (e.g., "Yankees", "NYY", "Detroit Tigers")
- **Alphabetized results** in search lookahead
- **Auto-complete** with instant local filtering + API search

### 📊 Analysis Tools
- **3D Pitch Trajectories** - Interactive tunneling visualization
- **Sequence Analysis** - Calculate whiff rate, chase rate, weak contact
- **Summary Statistics** - Pitch arsenal, velocity, and more
- **Date Range Filtering** - Analyze specific time periods

### 🗄️ Data Management
- **Live Data Fetching** - Gets real Statcast data from Baseball Savant
- **Intelligent Caching** - Caches pitcher data for fast access
- **Registry System** - Maintains list of all 2025 pitchers

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd sequencebaseball
pip install -r requirements.txt
```

### 2. Initialize 2025 Data

```bash
# Build pitcher registry from 2025 MLB data
python init_2025_data.py

# Optional: Pre-fetch data for all pitchers (takes longer)
python init_2025_data.py --fetch-all
```

### 3. Start the API

```bash
python run_api.py
```

Then open http://localhost:8000 in your browser.

---

## 📁 Project Structure

```
sequencebaseball/
│
├── api/
│   ├── main.py              # FastAPI application
│   └── static/
│       └── index.html       # Web frontend
│
├── data/
│   ├── cache/               # Cached pitcher data (auto-generated)
│   └── pitcher_registry_2025.json  # Pitcher registry (auto-generated)
│
├── init_2025_data.py        # Initialize 2025 pitcher data
├── mlb_2025_integration.py  # Data fetching from pybaseball
├── pitch_viz.py             # Core visualization functions
├── sequence_visualizations.py  # Chart generation
├── run_api.py               # API startup script
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🔧 API Endpoints

### Pitchers

| Endpoint | Description |
|----------|-------------|
| `GET /pitchers` | List all pitchers (alphabetized) |
| `GET /pitchers/search?q={query}` | Search by name or team |
| `GET /pitchers/{id}` | Get specific pitcher details |
| `GET /pitchers/{id}/seasons` | Available seasons |
| `GET /pitchers/{id}/pitch-types` | Pitcher's pitch arsenal |

### Teams

| Endpoint | Description |
|----------|-------------|
| `GET /teams` | List all MLB teams with pitcher counts |
| `GET /teams/{team}/pitchers` | All pitchers for a team |

### Visualizations

| Endpoint | Description |
|----------|-------------|
| `POST /visualizations/tunnel` | 3D pitch trajectory visualization |
| `POST /visualizations/sequences` | Sequence analysis data |
| `POST /visualizations/sequences/chart` | Interactive sequence table |

### Data

| Endpoint | Description |
|----------|-------------|
| `GET /data/{id}/summary` | Pitcher summary statistics |
| `GET /data/{id}/movement` | Pitch movement profile |
| `POST /data/{id}/refresh` | Force refresh pitcher data |

---

## 💻 Usage Examples

### Search for Pitchers

```python
import requests

# Search by name
response = requests.get("http://localhost:8000/pitchers/search?q=Skubal")
print(response.json())
# Returns: {"results": [{"name": "Tarik Skubal", "team": "DET", ...}], "is_team_search": false}

# Search by team
response = requests.get("http://localhost:8000/pitchers/search?q=Yankees")
print(response.json())
# Returns: {"results": [...], "is_team_search": true}
```

### Get All Pitchers for a Team

```python
response = requests.get("http://localhost:8000/teams/NYY/pitchers")
yankees_pitchers = response.json()
# Returns alphabetized list of all Yankees pitchers
```

### Generate 3D Tunnel Visualization

```python
response = requests.post("http://localhost:8000/visualizations/tunnel", json={
    "pitcher_id": 669373,  # Tarik Skubal
    "pitch_types": ["4-Seam Fastball", "Slider"],
    "season": 2025,
    "max_pitches_per_type": 30
})
# Returns HTML visualization
```

---

## 📊 Search Functionality

### Name Search
- Searches pitcher names (partial match)
- Returns alphabetized results
- Example: "Wheeler" → Zack Wheeler, Kyle Wheeler, etc.

### Team Search
- Detects team abbreviations (NYY, LAD, etc.)
- Matches partial team names ("Yankees", "Dodgers")
- Returns all pitchers for that team, alphabetized
- Example: "Yankees" → All NY Yankees pitchers

### Search Response

```json
{
  "query": "Yankees",
  "results": [
    {"id": 123456, "name": "Aaron Judge", "team": "NYY", ...},
    {"id": 234567, "name": "Gerrit Cole", "team": "NYY", ...}
  ],
  "total_count": 15,
  "is_team_search": true
}
```

---

## 🗄️ Data Sources

### Primary: pybaseball (Baseball Savant)
- Official MLB Statcast data
- All 35+ Statcast fields
- Release point, spin rate, movement, velocity
- Game outcomes and contexts

### Pitch Registry
- Built automatically from 2025 season data
- Includes all pitchers with 100+ pitches
- De-duped and distinct entries
- Updated via `init_2025_data.py`

---

## ⚙️ Configuration

### Minimum Pitch Threshold
```bash
# Include pitchers with 50+ pitches instead of 100
python init_2025_data.py --min-pitches 50
```

### Refresh Data
```bash
# Via API
curl -X POST http://localhost:8000/registry/refresh

# Or run init script again
python init_2025_data.py
```

---

## 🔧 Requirements

- Python 3.8+
- pybaseball >= 2.2.7
- FastAPI
- pandas, numpy
- plotly, matplotlib, seaborn
- requests

See `requirements.txt` for complete list.

---

## 📝 Notes

### No Sample Data
This version uses **only real MLB data** from the 2025 season. Sample/generated data has been removed. If the 2025 season hasn't started or you need offline access, you'll need to initialize the registry when you have network access.

### Caching
- Pitcher data is cached in `data/cache/`
- Cache expires after 1 day
- Use `POST /data/{id}/refresh` to force update

### Rate Limiting
When fetching many pitchers, pybaseball may be rate-limited by Baseball Savant. The initialization script handles this gracefully.

---

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive Swagger docs
- [pybaseball Docs](https://github.com/jldbc/pybaseball) - Data source library
- [Baseball Savant](https://baseballsavant.mlb.com/) - Official MLB Statcast

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- **MLB** - For Statcast data via Baseball Savant
- **pybaseball** - For Python API wrapper
- **Plotly** - For interactive visualizations
- **FastAPI** - For the web framework

---

**Last Updated:** December 2025  
**Version:** 1.0.0 - 2025 MLB Season Integration
