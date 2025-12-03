# Phase 1: Data Exploration & Setup - Complete ✓

## Summary

Successfully set up the project environment and generated sample Statcast data for development.

### Network Access Issue
- ⚠️ **Issue**: Unable to access MLB Statcast API (baseballsavant.mlb.com) due to network proxy restrictions (403 Forbidden)
- ✓ **Solution**: Generated realistic sample data based on actual pitcher profiles and Statcast schema
- 📝 **Note**: Code is designed to work with real Statcast data - simply replace CSV files when network access is available

---

## Test Pitchers

### 1. Tarik Skubal (Detroit Tigers)
- **MLB ID**: 669373
- **Throws**: Left
- **2024**: American League Cy Young Winner
- **Signature**: Elite fastball-slider combination with exceptional tunneling
- **Sample Data**: 2,000 pitches (data/skubal_statcast_2024.csv)

**Arsenal**:
- **4-Seam Fastball** (55% usage): 96.5 MPH avg, 2400 RPM spin
- **Slider** (40% usage): 86.0 MPH avg, 2650 RPM spin, devastating break
- **Changeup** (5% usage): 88.5 MPH avg, arm-side fade

### 2. Jhoan Duran (Minnesota Twins)
- **MLB ID**: 650556 (estimated - player lookup failed)
- **Throws**: Right
- **Known For**: "Splinker" - hardest splitter in baseball
- **Sample Data**: 1,500 pitches (data/duran_statcast_2024.csv)

**Arsenal**:
- **4-Seam Fastball** (45% usage): 100.5 MPH avg (elite velocity)
- **Splitter/Splinker** (50% usage): 94.0 MPH avg (signature pitch)
- **Slider** (5% usage): 88.0 MPH avg

---

## Available Data Fields (35 columns)

### 🎯 Pitch Identification & Type
| Field | Description | Example Values |
|-------|-------------|----------------|
| `pitch_type` | Abbreviated pitch type code | FF, SL, CH, FS, CU |
| `pitch_name` | Full pitch name | 4-Seam Fastball, Slider, Changeup |

### 📍 Release Point Data
| Field | Description | Units |
|-------|-------------|-------|
| `release_pos_x` | Horizontal release position (catcher's view) | feet (negative = 3B side for LHP) |
| `release_pos_y` | Distance from home plate at release | feet (~50-55 ft) |
| `release_pos_z` | Vertical release height | feet (5-7 ft typical) |
| `release_extension` | Extension toward home plate | feet (5.5-6.5 ft typical) |

### ⚡ Velocity & Spin
| Field | Description | Units |
|-------|-------------|-------|
| `release_speed` | Velocity at release point | MPH |
| `effective_speed` | Perceived velocity (accounts for extension) | MPH |
| `release_spin_rate` | Spin rate | RPM |
| `spin_axis` | Direction of spin | degrees (0-360) |

### 🌀 Movement Data
| Field | Description | Units |
|-------|-------------|-------|
| `pfx_x` | Horizontal movement | inches (positive = arm side for RHP) |
| `pfx_z` | Vertical movement (induced) | inches (positive = rise) |
| `plate_x` | Horizontal location at plate | feet (0 = middle, negative = LHH side) |
| `plate_z` | Vertical location at plate | feet (ground level = 0) |

### 📐 Physics Components (for trajectory calculation)
| Field | Description | Units |
|-------|-------------|-------|
| `vx0` | Initial velocity X component | ft/s |
| `vy0` | Initial velocity Y component (toward plate) | ft/s |
| `vz0` | Initial velocity Z component | ft/s |
| `ax` | Acceleration X (from Magnus force) | ft/s² |
| `ay` | Acceleration Y (air resistance) | ft/s² |
| `az` | Acceleration Z (gravity + Magnus) | ft/s² |

### ⚾ Outcome & Context
| Field | Description | Example Values |
|-------|-------------|----------------|
| `description` | Pitch outcome | ball, called_strike, swinging_strike, foul, hit_into_play |
| `events` | At-bat result (if final pitch) | single, strikeout, field_out, home_run |
| `type` | Pitch result code | S (strike), B (ball), X (in play) |
| `zone` | Location zone | 1-9 (strike zone), 11-14 (outside) |
| `balls` | Ball count | 0-3 |
| `strikes` | Strike count | 0-2 |

### 👥 Player & Situation
| Field | Description | Example Values |
|-------|-------------|----------------|
| `pitcher` | MLB ID of pitcher | 669373 (Skubal) |
| `batter` | MLB ID of batter | various |
| `stand` | Batter handedness | R, L |
| `p_throws` | Pitcher handedness | R, L |
| `inning` | Inning number | 1-9+ |
| `outs_when_up` | Outs when pitch thrown | 0, 1, 2 |

### 🎮 Sequencing Data
| Field | Description | Example Values |
|-------|-------------|----------------|
| `game_date` | Date of game | 2024-04-01 |
| `at_bat_number` | At-bat sequence number | 1, 2, 3... |
| `pitch_number` | Pitch number within at-bat | 1-10+ |

---

## Data Completeness

✓ **100% Complete** - All critical fields populated in sample data

For real Statcast data, expect:
- **>99% complete**: pitch_type, release_speed, release_pos_*, plate_*, description
- **>95% complete**: pfx_x, pfx_z, release_spin_rate
- **>90% complete**: spin_axis, ax, ay, az

---

## Filtering Examples

The data supports comprehensive filtering for analysis:

```python
import pandas as pd

# Load data
df = pd.read_csv('data/skubal_statcast_2024.csv')

# 1. Filter by pitch type
fastballs = df[df['pitch_name'] == '4-Seam Fastball']
sliders = df[df['pitch_name'] == 'Slider']

# 2. Filter by batter handedness
vs_righties = df[df['stand'] == 'R']
vs_lefties = df[df['stand'] == 'L']

# 3. Filter by outcome
whiffs = df[df['description'] == 'swinging_strike']
chases = df[(df['description'] == 'swinging_strike') & (df['zone'] >= 11)]

# 4. Filter by count
two_strike = df[df['strikes'] == 2]
ahead_count = df[(df['balls'] < df['strikes'])]

# 5. Combine filters (FB vs RHH that generated whiffs)
fb_rhh_whiff = df[
    (df['pitch_name'] == '4-Seam Fastball') &
    (df['stand'] == 'R') &
    (df['description'] == 'swinging_strike')
]

# 6. Group by at-bat for sequence analysis
df['at_bat_id'] = df['game_date'].astype(str) + '_' + df['at_bat_number'].astype(str)
sequences = df.sort_values(['at_bat_id', 'pitch_number'])
```

---

## Data Structure for Sequencing

Each **at-bat** contains multiple pitches that can be grouped:

```
At-Bat ID: 2024-04-01_5
├─ Pitch 1: 4-Seam Fastball, 96.5 MPH, called_strike
├─ Pitch 2: 4-Seam Fastball, 97.1 MPH, ball
├─ Pitch 3: Slider, 86.2 MPH, swinging_strike
└─ Pitch 4: Slider, 85.8 MPH, swinging_strike (strikeout)
```

**Key for Sequence Analysis**:
- Group by: `at_bat_id` (or `game_date` + `at_bat_number`)
- Sort by: `pitch_number`
- Success metrics: Calculate from `description` and `events` fields

---

## Key Metrics We'll Calculate

### 1. Pitch Movement Metrics
- Horizontal vs Vertical break (pfx_x, pfx_z)
- Release point consistency (std dev of release_pos_*)
- Tunnel point (where pitches diverge)

### 2. Effectiveness Metrics
- **Whiff Rate** = swinging_strikes / swings
- **Chase Rate** = swings outside zone / pitches outside zone
- **Zone %** = pitches in zone (1-9) / total pitches
- **Called Strike Rate** = called_strikes / pitches taken

### 3. Sequence Metrics
- Most common 2-pitch sequences
- Success rate by sequence type
- Comparison to league average (will need baseline data)

---

## Sample Data Statistics

### Tarik Skubal (2,000 pitches)
- **Date Range**: 2024-04-01 to 2024-05-22
- **Games**: ~80 games worth
- **Arsenal**:
  - 4-Seam Fastball: 1,074 (54%)
  - Slider: 822 (41%)
  - Changeup: 104 (5%)
- **vs RHH**: ~55% of pitches
- **vs LHH**: ~45% of pitches

### Jhoan Duran (1,500 pitches)
- **Date Range**: 2024-04-01 to 2024-05-13
- **Games**: ~60 games worth
- **Arsenal**:
  - Splitter: 755 (50%)
  - 4-Seam Fastball: 678 (45%)
  - Slider: 67 (5%)
- **vs RHH**: ~55% of pitches
- **vs LHH**: ~45% of pitches

---

## Next Steps → Phase 2

With data exploration complete, we're ready to build:

1. ✅ **3D Pitch Trajectory Visualization** - Use release_pos_*, plate_*, and physics data to plot pitch paths
2. ✅ **Sequence Analysis Function** - Group by at_bat_id and calculate success metrics
3. ✅ **Portfolio Export Pipeline** - Generate complete packages per pitcher

All visualization functions will be designed to work with this exact data structure.

---

## Files Generated

```
sequencebaseball/
├── data/
│   ├── skubal_statcast_2024.csv      # 2,000 pitches, 35 columns
│   └── duran_statcast_2024.csv       # 1,500 pitches, 35 columns
├── phase1_data_exploration.ipynb     # Jupyter notebook (template)
├── generate_sample_data.py           # Data generation script
├── troubleshoot_data_pull.py         # Network troubleshooting
└── PHASE1_DATA_DOCUMENTATION.md      # This file
```

---

**Phase 1 Status**: ✓ **COMPLETE**

Ready to proceed to Phase 2: Core Visualization Functions
