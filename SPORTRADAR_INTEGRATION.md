# SportRadar MLB API Integration

## Overview

This project includes integration with [SportRadar's MLB API v8](https://developer.sportradar.com/baseball/reference/mlb-overview), providing access to live MLB data including schedules, game statistics, player profiles, and pitch metrics.

**API Key:** `MkPCdbssBXUy74XhOtt58Ed7Ov8GMrJnVGHk2YmN`  
**Valid Until:** January 4, 2026

> ⚠️ **Important:** This integration does NOT include Statcast data (spin rate, exit velocity, launch angle, etc.). Statcast is a separate premium package from SportRadar. For Statcast data, use the `pybaseball` library or Baseball Savant.

---

## What's Included

### SportRadar Data (via this integration)
- ✅ Daily/Season schedules
- ✅ Game boxscores and summaries
- ✅ Play-by-play data
- ✅ Basic pitch metrics (type, velocity, results)
- ✅ Player profiles and stats
- ✅ Team rosters and profiles
- ✅ League standings and leaders
- ✅ Injury reports
- ✅ Transactions

### NOT Included (Requires Statcast Package)
- ❌ Spin rate
- ❌ Pitch movement (pfx_x, pfx_z)
- ❌ Exit velocity
- ❌ Launch angle
- ❌ Release point coordinates
- ❌ Expected stats (xwOBA, xBA)

---

## Installation

The SportRadar integration requires the `requests` library:

```bash
pip install requests
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

---

## Usage

### Python Module

```python
from sportradar_integration import SportRadarMLB

# Initialize client
client = SportRadarMLB()

# Get daily schedule
schedule = client.get_daily_schedule("2024-08-15")

# Get game details
game_id = "abc123-..."  # SportRadar game ID from schedule
boxscore = client.get_game_boxscore(game_id)
pitch_metrics = client.get_game_pitch_metrics(game_id)
pbp = client.get_game_play_by_play(game_id)

# Get team info
hierarchy = client.get_league_hierarchy()
teams = client.get_teams()

# Get player profile
player = client.get_player_profile(player_id)

# Get standings and leaders
standings = client.get_standings(season=2024)
leaders = client.get_league_leaders(season=2024)

# Get injuries
injuries = client.get_injuries()
```

### API Endpoints

The FastAPI server exposes SportRadar endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /sportradar/status` | Check API connection |
| `GET /sportradar/schedule/{date}` | Daily schedule (YYYY-MM-DD) |
| `GET /sportradar/game/{game_id}/boxscore` | Game boxscore |
| `GET /sportradar/game/{game_id}/pitch_metrics` | Pitch metrics |
| `GET /sportradar/game/{game_id}/pbp` | Play-by-play |
| `GET /sportradar/teams` | All MLB teams |
| `GET /sportradar/team/{team_id}/profile` | Team roster |
| `GET /sportradar/player/{player_id}/profile` | Player stats |
| `GET /sportradar/standings/{season}` | Standings |
| `GET /sportradar/leaders/{season}` | League leaders |
| `GET /sportradar/injuries` | Injury report |

---

## Data Extraction Examples

### Get Pitcher Season Data

```python
from sportradar_integration import (
    SportRadarMLB, 
    get_pitcher_season_pitches,
    convert_sportradar_to_statcast_format
)

client = SportRadarMLB()

# Get all pitches for Tarik Skubal in 2024
pitches = get_pitcher_season_pitches(
    client,
    pitcher_name="Skubal",
    season=2024,
    max_games=10  # Limit for testing
)

# Convert to Statcast-compatible format
statcast_format = convert_sportradar_to_statcast_format(pitches)
```

### Extract Pitch Metrics from Game

```python
from sportradar_integration import SportRadarMLB, extract_pitch_metrics

client = SportRadarMLB()

# Get pitch metrics for a game
metrics_data = client.get_game_pitch_metrics(game_id)
metrics_df = extract_pitch_metrics(metrics_data)

print(metrics_df)
# Columns: player_name, pitch_type, count, avg_speed, max_speed, strikes, balls, etc.
```

### Find Team Pitchers

```python
from sportradar_integration import SportRadarMLB, lookup_team_id, get_team_pitchers

client = SportRadarMLB()

# Look up team ID
team_id = lookup_team_id(client, "Tigers")

# Get pitchers
pitchers = get_team_pitchers(client, team_id)
for p in pitchers:
    print(f"{p['name']} - {p['throw_hand']}-handed")
```

---

## Caching

The client automatically caches API responses to avoid redundant requests:

- Cache location: `data/sportradar_cache/`
- Default TTL: 24 hours
- Disable with: `SportRadarMLB(use_cache=False)`

---

## Rate Limiting

SportRadar has rate limits on API requests. The client includes:

- Automatic delay between requests (1 second)
- Retry logic with exponential backoff
- Graceful handling of 429 (rate limit) errors

---

## Environment Variables

You can override the API key via environment variable:

```bash
export SPORTRADAR_API_KEY="your-api-key-here"
```

---

## Error Handling

```python
from sportradar_integration import SportRadarMLB

client = SportRadarMLB()

try:
    schedule = client.get_daily_schedule("2024-08-15")
except PermissionError as e:
    print(f"API access denied: {e}")
except ValueError as e:
    print(f"Resource not found: {e}")
except ConnectionError as e:
    print(f"Network error: {e}")
```

---

## Combining with Existing Data

SportRadar can supplement your existing Statcast data:

```python
import pandas as pd
from sportradar_integration import SportRadarMLB

# Load existing Statcast data
statcast_df = pd.read_csv("data/skubal_statcast_2024.csv")

# Get current injuries
client = SportRadarMLB()
injuries = client.get_injuries()

# Get latest standings
standings = client.get_standings(2024)

# Get league leaders
leaders = client.get_league_leaders(2024)
```

---

## API Documentation Reference

Full SportRadar MLB API documentation:
- [MLB v8 Overview](https://developer.sportradar.com/baseball/reference/mlb-overview)
- [Endpoint Reference](https://developer.sportradar.com/baseball/reference/mlb-overview#endpoints)

---

## Limitations

1. **No Statcast Data**: This trial key doesn't include Statcast metrics
2. **Rate Limits**: Be mindful of API rate limits
3. **Trial Environment**: Using trial endpoint (`/trial/`) - production apps need production keys
4. **Historical Data**: Some historical data may have limitations

---

## Files

| File | Description |
|------|-------------|
| `sportradar_integration.py` | Main integration module |
| `api/index.py` | FastAPI endpoints |
| `SPORTRADAR_INTEGRATION.md` | This documentation |

---

## Quick Test

```bash
# Test the integration
python sportradar_integration.py
```

Or via API:
```bash
curl http://localhost:8000/sportradar/status
```

---

## Support

- SportRadar Support: [Contact Us](https://developer.sportradar.com/contact-us)
- API Documentation: [developer.sportradar.com](https://developer.sportradar.com/baseball/reference/mlb-overview)
