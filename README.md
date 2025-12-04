# Baseball Pitch Sequencing Visualization System

A comprehensive Python-based system for analyzing and visualizing MLB pitcher data using Statcast metrics. Features 3D pitch trajectory visualization, sequence analysis, and automated portfolio generation.

## 🎯 Project Overview

This project analyzes pitch sequencing strategies and creates portfolio-ready visualizations for MLB pitchers. Built with Python, Plotly, and matplotlib, it transforms raw Statcast data into compelling visual stories.

### Featured Pitchers

1. **Tarik Skubal (DET)** - "The Tunnel Master"
   - 2024 & 2025 AL Cy Young Winner
   - Elite fastball-slider tunnel combination
   - Focus: How his pitch sequencing dominated hitters

2. **Jhoan Duran (MIN/PHI)** - "The Splinker"
   - Hardest splitter in baseball (94 MPH avg)
   - Elite 100+ MPH fastball
   - Focus: Why his splitter is nearly unhittable

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Core Functions](#core-functions)
- [Generated Visualizations](#generated-visualizations)
- [Data Documentation](#data-documentation)
- [Usage Examples](#usage-examples)
- [Technical Specifications](#technical-specifications)
- [Known Issues](#known-issues)
- [Future Enhancements](#future-enhancements)

---

## ✨ Features

### Phase 1: Data Exploration
- ✅ Automated Statcast data retrieval using `pybaseball`
- ✅ Sample data generation for offline development
- ✅ Comprehensive data field documentation
- ✅ Data quality assessment

### Phase 2: Core Visualization Functions
- ✅ **3D Pitch Trajectory Visualization** - Interactive tunneling analysis
- ✅ **Sequence Analysis** - Calculate effectiveness metrics (whiff rate, chase rate, weak contact)
- ✅ **Portfolio Export Pipeline** - Automated batch processing

### Phase 3: Showcase Pieces
- ✅ Pitcher-specific narrative visualizations
- ✅ Movement profiles and velocity distributions
- ✅ Usage analysis by count situation
- ✅ Automated narrative generation

---

## 🚀 Installation

### Requirements

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone or navigate to project directory
cd sequencebaseball

# Install dependencies
pip install pybaseball jupyter plotly pandas numpy matplotlib seaborn scipy kaleido

# Optional: For PNG export from Plotly (requires Chrome)
# Run: plotly_get_chrome
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `pybaseball` | MLB Statcast data retrieval |
| `plotly` | Interactive 3D visualizations |
| `matplotlib` | Static charts and graphs |
| `seaborn` | Statistical visualizations |
| `pandas` | Data manipulation |
| `numpy` | Numerical computations |
| `scipy` | Statistical analysis |

---

## 🏃 Quick Start

### Option 1: Run Complete Showcase (Recommended)

```bash
# Generate complete showcases for both pitchers
python phase3_showcase.py
```

This will create:
- 3D tunnel visualizations
- Sequence analysis charts
- Movement profiles
- Velocity distributions
- Usage breakdowns
- Narrative summaries

Output: `showcase_portfolios/skubal_tarik/` and `showcase_portfolios/duran_jhoan/`

### Option 2: Test Individual Functions

```bash
# Test all visualization functions with sample data
python test_visualizations.py
```

Output: `test_output/` directory with all test visualizations

### Option 3: Use Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open: phase1_data_exploration.ipynb
```

---

## 📁 Project Structure

```
sequencebaseball/
│
├── data/                                  # Sample Statcast data
│   ├── skubal_statcast_2024.csv          # Skubal pitches (2,000)
│   └── duran_statcast_2024.csv           # Duran pitches (1,500)
│
├── showcase_portfolios/                   # Phase 3 output
│   ├── skubal_tarik/
│   │   ├── 01_tunnel_fastball_slider_RHH.html
│   │   ├── 02_sequences_chart_RHH.png
│   │   ├── 03_sequences_chart_LHH.png
│   │   ├── 04_movement_profile.png
│   │   ├── 05_velocity_distribution.png
│   │   ├── 06_usage_by_count.png
│   │   ├── NARRATIVE.md
│   │   └── [CSV data files]
│   │
│   └── duran_jhoan/
│       └── [Similar structure]
│
├── test_output/                          # Phase 2 test output
│   ├── portfolio_skubal/
│   └── portfolio_duran/
│
├── pitch_viz.py                          # Core visualization module
├── phase3_showcase.py                    # Showcase generation script
├── test_visualizations.py                # Testing script
├── generate_sample_data.py               # Sample data generator
├── phase1_data_exploration.ipynb         # Jupyter notebook
│
├── PHASE1_DATA_DOCUMENTATION.md          # Data field reference
├── README.md                             # This file
└── LICENSE                               # MIT License
```

---

## 🎨 Core Functions

### 1. 3D Pitch Trajectory Visualization

```python
from pitch_viz import visualize_pitch_trajectories_3d

fig = visualize_pitch_trajectories_3d(
    df=pitcher_data,
    pitcher_name="Tarik Skubal",
    pitch_types=["4-Seam Fastball", "Slider"],
    batter_hand='R',  # Optional: 'R', 'L', or None for all
    max_pitches_per_type=25,
    output_html="output.html",
    output_png="output.png"  # Requires Chrome/Kaleido
)
```

**Features:**
- Interactive 3D scatter/line plots
- Catcher's perspective camera angle
- Strike zone overlay
- Release point annotations
- Pitch statistics sidebar

### 2. Sequence Analysis

```python
from pitch_viz import analyze_pitch_sequences

sequences = analyze_pitch_sequences(
    df=pitcher_data,
    pitcher_name="Tarik Skubal",
    min_sample_size=20,
    success_metric='whiff_rate',  # 'whiff_rate', 'chase_rate', 'overall'
    batter_hand='R'  # Optional
)
```

**Returns DataFrame with:**
- Sequence (e.g., "Fastball → Slider")
- Usage count
- Whiff Rate (%)
- Chase Rate (%)
- Weak Contact Rate (%)
- Overall Score

### 3. Portfolio Package Generation

```python
from pitch_viz import generate_portfolio_package

manifest = generate_portfolio_package(
    df=pitcher_data,
    pitcher_name="Tarik Skubal",
    pitcher_id=669373,
    output_dir="portfolios/skubal",
    pitch_types_for_tunnel=["4-Seam Fastball", "Slider"],
    batter_hands=['R', 'L'],
    include_interactive=True
)
```

**Generates:**
- Hero tunnel visualization (HTML + PNG)
- Sequence analysis by batter handedness (CSV + charts)
- Summary statistics (JSON + TXT)
- Complete organized folder structure

---

## 📊 Generated Visualizations

### 3D Tunnel Visualizations

**Shows:**
- Pitch trajectories from release to plate
- Color-coded by pitch type
- Release point clusters
- Strike zone reference

### Sequence Analysis Charts

**Displays:**
- Top 10 most effective sequences
- Whiff rate, chase rate, weak contact rate
- Usage frequency (n=X)
- Color-coded metrics

### Movement Profiles

**Illustrates:**
- Horizontal vs vertical break
- Scatter plot by pitch type
- Mean markers for each pitch

### Velocity Distributions

**Features:**
- Violin plots by pitch type
- Mean velocity annotations
- Distribution shapes

---

## 📖 Data Documentation

### Available Statcast Fields (35 total)

**Core Metrics:**
- `pitch_type`, `pitch_name` - Pitch identification
- `release_speed`, `release_spin_rate` - Velocity and spin
- `release_pos_x`, `release_pos_y`, `release_pos_z` - Release point (3D)
- `pfx_x`, `pfx_z` - Movement (horizontal, vertical)
- `plate_x`, `plate_z` - Location at plate
- `vx0`, `vy0`, `vz0` - Initial velocity components
- `ax`, `ay`, `az` - Acceleration components

**Context:**
- `description` - Pitch outcome (ball, strike, whiff, etc.)
- `events` - At-bat result
- `zone` - Location zone (1-9 in strike zone)
- `stand`, `p_throws` - Batter/pitcher handedness
- `balls`, `strikes` - Count

**Sequencing:**
- `game_date`, `at_bat_number`, `pitch_number`

See `PHASE1_DATA_DOCUMENTATION.md` for complete field reference.

---

## 💡 Usage Examples

### Example 1: Create Custom Tunnel Viz

```python
import pandas as pd
from pitch_viz import visualize_pitch_trajectories_3d

# Load your data
df = pd.read_csv('data/skubal_statcast_2024.csv')

# Create visualization
visualize_pitch_trajectories_3d(
    df=df,
    pitcher_name="Tarik Skubal",
    pitch_types=["4-Seam Fastball", "Slider", "Changeup"],
    batter_hand='L',  # vs lefties only
    output_html="skubal_vs_LHH.html"
)
```

### Example 2: Find Best Sequences

```python
from pitch_viz import analyze_pitch_sequences, create_sequence_chart

# Analyze
sequences = analyze_pitch_sequences(
    df=df,
    pitcher_name="Tarik Skubal",
    min_sample_size=15,
    success_metric='whiff_rate'
)

# Display top 5
print(sequences.head(5))

# Create chart
create_sequence_chart(
    sequence_df=sequences,
    pitcher_name="Tarik Skubal",
    top_n=8,
    output_png="top_sequences.png"
)
```

### Example 3: Batch Process Multiple Pitchers

```python
from pitch_viz import generate_portfolio_package
import pandas as pd

pitchers = [
    {'name': 'Tarik Skubal', 'id': 669373, 'file': 'skubal_statcast_2024.csv',
     'pitches': ['4-Seam Fastball', 'Slider']},
    {'name': 'Jhoan Duran', 'id': 650556, 'file': 'duran_statcast_2024.csv',
     'pitches': ['4-Seam Fastball', 'Splitter']},
]

for p in pitchers:
    df = pd.read_csv(f'data/{p["file"]}')
    generate_portfolio_package(
        df=df,
        pitcher_name=p['name'],
        pitcher_id=p['id'],
        output_dir=f'portfolios/{p["name"].replace(" ", "_").lower()}',
        pitch_types_for_tunnel=p['pitches']
    )
```

---

## 🔧 Technical Specifications

### Visualization Standards

**Colors (Colorblind-Friendly):**
- Fastball: #E63946 (Red)
- Slider: #FFD166 (Yellow)
- Curveball: #06AED5 (Blue)
- Changeup: #06A77D (Green)
- Splitter/Sinker: #F77F00 (Orange)

**File Formats:**
- Interactive: HTML (Plotly)
- Static charts: PNG (300 DPI)
- Data exports: CSV
- Metadata: JSON

**Dimensions:**
- 3D visualizations: 1920x1080
- Charts: 1200x800 to 1200x1000
- DPI: 100 (screen), 300 (print)

### Performance

- Single pitcher analysis: < 2 minutes
- Full portfolio generation: < 5 minutes
- Batch processing (10 pitchers): < 30 minutes

### Data Requirements

- Minimum 500 pitches per pitcher for reliable analysis
- Minimum 20 occurrences per sequence for statistical validity
- At least 200 pitches per pitch type for movement analysis

---

## ⚠️ Known Issues

### 1. Network Access to Statcast API

**Issue:** Cannot access baseballsavant.mlb.com due to network proxy restrictions (403 Forbidden)

**Solution:** Project includes realistic sample data generator
- Use `generate_sample_data.py` to create test data
- Code is designed to work with real Statcast data once network access is available
- Simply replace CSV files in `data/` directory

### 2. PNG Export from Plotly

**Issue:** Kaleido requires Chrome for PNG export

**Solution:**
- Interactive HTML files work perfectly (primary deliverable)
- Install Chrome with `plotly_get_chrome` for PNG export
- Alternative: Use matplotlib-based charts (included)

### 3. Jhoan Duran Player Lookup

**Issue:** `playerid_lookup('duran', 'jhoan')` returns empty

**Workaround:** Using estimated MLB ID (650556) in sample data

---

## 🚀 Future Enhancements

### Planned Features

1. **League Average Comparisons**
   - Compare pitcher metrics to MLB averages
   - Percentile rankings for each metric

2. **Multi-Pitcher Comparisons**
   - Side-by-side tunnel visualizations
   - Comparative sequence analysis

3. **Advanced Metrics**
   - Expected stats (xwOBA, xBA)
   - Pitch quality scoring
   - Sequencing entropy

4. **Interactive Dashboard**
   - Streamlit or Dash web interface
   - Real-time data updates
   - Custom date range selection

5. **Video Integration**
   - Sync pitch data with video clips
   - Side-by-side visual/data analysis

6. **Report Generation**
   - PDF export with all visualizations
   - LaTeX formatting for academic papers
   - PowerPoint template integration

---

## 📝 Citation

If you use this project in your research or analysis, please cite:

```
Baseball Pitch Sequencing Visualization System (2025)
Data Source: MLB Statcast via pybaseball
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional pitcher showcases
- New visualization types
- Performance optimizations
- Bug fixes

---

## 📧 Contact

For questions or feedback about this project, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- **MLB** - For Statcast data
- **pybaseball** - For Python API wrapper
- **Plotly** - For interactive visualizations
- **matplotlib/seaborn** - For static charts

---

## 📚 References

- [pybaseball Documentation](https://github.com/jldbc/pybaseball)
- [MLB Statcast Search](https://baseballsavant.mlb.com/statcast_search)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Baseball Savant](https://baseballsavant.mlb.com/)

---

**Last Updated:** December 2025

**Version:** 0.1.0
