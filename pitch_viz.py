"""
Pitch Sequencing Visualization Module
Core functions for 3D trajectory visualization, sequence analysis, and portfolio generation
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Optional, Dict, Tuple

# Color palette (colorblind-friendly)
PITCH_COLORS = {
    '4-Seam Fastball': '#E63946',  # Red
    'Fastball': '#E63946',
    'FF': '#E63946',
    'Slider': '#FFD166',  # Yellow
    'SL': '#FFD166',
    'Curveball': '#06AED5',  # Blue
    'CU': '#06AED5',
    'Changeup': '#06A77D',  # Green
    'CH': '#06A77D',
    'Splitter': '#F77F00',  # Orange
    'FS': '#F77F00',
    'Sinker': '#F77F00',
    'SI': '#F77F00',
}

# Strike zone dimensions (feet)
STRIKE_ZONE = {
    'width': 17/12,  # 17 inches in feet
    'top': 3.5,
    'bottom': 1.5
}


def calculate_pitch_trajectory(row: pd.Series, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate 3D pitch trajectory from release point to home plate using physics.
    Uses actual plate crossing data for endpoint accuracy.

    Parameters:
    -----------
    row : pd.Series
        Single pitch data row with release position, velocity, and acceleration data
    num_points : int
        Number of points to calculate along trajectory

    Returns:
    --------
    x, y, z : np.ndarray
        Arrays of x, y, z coordinates along pitch path
    """
    # Release point
    x0 = row['release_pos_x']
    y0 = row['release_pos_y']
    z0 = row['release_pos_z']

    # Actual plate crossing point (from Statcast)
    plate_x_final = row['plate_x']
    plate_z_final = row['plate_z']

    # Initial velocities (ft/s)
    vx0 = row['vx0']
    vy0 = row['vy0']
    vz0 = row['vz0']

    # Accelerations (ft/s²) - Statcast values are already signed
    # (az is typically negative, around -32 ft/s² including gravity)
    ax = row['ax']
    ay = row['ay']
    az = row['az']

    # Solve quadratic for time: y0 + vy0*t + 0.5*ay*t² = 0
    # ay is positive (drag slowing forward motion)
    a_coef = 0.5 * ay
    b_coef = vy0
    c_coef = y0
    
    discriminant = b_coef**2 - 4 * a_coef * c_coef
    if discriminant < 0 or a_coef == 0:
        # Fallback to simple estimate
        total_time = abs(y0 / vy0) if vy0 != 0 else 0.4
    else:
        # Take the smaller positive root
        t1 = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)
        t2 = (-b_coef - np.sqrt(discriminant)) / (2 * a_coef)
        positive_times = [t for t in [t1, t2] if t > 0]
        total_time = min(positive_times) if positive_times else 0.4

    # Time points
    t = np.linspace(0, total_time, num_points)

    # Calculate y position using kinematic equation
    y = y0 + vy0 * t + 0.5 * ay * t**2

    # For x and z, blend physics with actual endpoint for accuracy
    # Physics-based trajectory
    x_physics = x0 + vx0 * t + 0.5 * ax * t**2
    z_physics = z0 + vz0 * t + 0.5 * az * t**2

    # Blend factor (0 at start, 1 at end) to ensure we hit actual plate location
    blend = (t / total_time) ** 2  # Quadratic blend for smooth curve
    
    # Calculate where physics says we'd end up
    x_physics_end = x_physics[-1] if len(x_physics) > 0 else x0
    z_physics_end = z_physics[-1] if len(z_physics) > 0 else z0
    
    # Blend to actual plate crossing (only if plate data is valid)
    if not np.isnan(plate_x_final) and not np.isnan(plate_z_final):
        x_correction = (plate_x_final - x_physics_end) * blend
        z_correction = (plate_z_final - z_physics_end) * blend
        x = x_physics + x_correction
        z = z_physics + z_correction
    else:
        x = x_physics
        z = z_physics

    # Clip to home plate (y >= 0)
    mask = y >= 0
    x = x[mask]
    y = y[mask]
    z = z[mask]

    return x, y, z


def create_strike_zone_traces() -> List[go.Scatter3d]:
    """
    Create 3D traces for the strike zone outline and home plate

    Returns:
    --------
    list of go.Scatter3d
        Plotly 3D scatter traces for strike zone and plate
    """
    width = STRIKE_ZONE['width'] / 2
    top = STRIKE_ZONE['top']
    bottom = STRIKE_ZONE['bottom']
    
    traces = []

    # Strike zone rectangle at y=0 (home plate) - thicker, more visible
    x = [-width, width, width, -width, -width]
    y = [0, 0, 0, 0, 0]
    z = [bottom, bottom, top, top, bottom]

    traces.append(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='black', width=5),
        name='Strike Zone',
        showlegend=True
    ))
    
    # Add horizontal lines at top and bottom of zone for depth reference
    # These extend slightly back to help with depth perception
    for z_val in [bottom, top]:
        traces.append(go.Scatter3d(
            x=[-width, -width], y=[0, 3], z=[z_val, z_val],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        traces.append(go.Scatter3d(
            x=[width, width], y=[0, 3], z=[z_val, z_val],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add vertical corner lines extending back for depth reference
    for x_val in [-width, width]:
        traces.append(go.Scatter3d(
            x=[x_val, x_val], y=[0, 3], z=[bottom, bottom],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.2)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Home plate outline (pentagon shape at ground level)
    plate_half = 8.5/12  # 8.5 inches = half width at front
    plate_back = 17/12   # 17 inches back point
    plate_x = [-plate_half, plate_half, plate_half, 0, -plate_half, -plate_half]
    plate_y = [0, 0, plate_back/2, plate_back, plate_back/2, 0]
    plate_z = [0, 0, 0, 0, 0, 0]
    
    traces.append(go.Scatter3d(
        x=plate_x, y=plate_y, z=plate_z,
        mode='lines',
        line=dict(color='white', width=4),
        name='Home Plate',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add filled surface for strike zone (semi-transparent)
    traces.append(go.Mesh3d(
        x=[-width, width, width, -width],
        y=[0, 0, 0, 0],
        z=[bottom, bottom, top, top],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='lightblue',
        opacity=0.15,
        name='Strike Zone Area',
        showlegend=False,
        hoverinfo='skip'
    ))

    return traces


def visualize_pitch_trajectories_3d(
    df: pd.DataFrame,
    pitcher_name: str,
    pitch_types: List[str],
    date_range: Optional[Tuple[str, str]] = None,
    batter_hand: Optional[str] = None,
    max_pitches_per_type: int = 30,
    output_html: Optional[str] = None,
    output_png: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Function 1: 3D Pitch Trajectory Visualization (Tunneling)

    Create interactive 3D visualization showing pitch paths from release to plate

    Parameters:
    -----------
    df : pd.DataFrame
        Statcast pitch data
    pitcher_name : str
        Name of pitcher for title
    pitch_types : list of str
        Pitch types to include (e.g., ["4-Seam Fastball", "Slider"])
    date_range : tuple of str, optional
        (start_date, end_date) to filter data
    batter_hand : str, optional
        'R' or 'L' to filter by batter handedness
    max_pitches_per_type : int
        Maximum number of pitches to show per type (prevents overcrowding)
    output_html : str, optional
        Path to save interactive HTML file
    output_png : str, optional
        Path to save static PNG image (1920x1080, 300 DPI)
    title : str, optional
        Custom title for the plot

    Returns:
    --------
    fig : go.Figure
        Plotly figure object
    """
    # Filter data
    filtered_df = df.copy()

    if date_range:
        filtered_df = filtered_df[
            (filtered_df['game_date'] >= date_range[0]) &
            (filtered_df['game_date'] <= date_range[1])
        ]

    if batter_hand:
        filtered_df = filtered_df[filtered_df['stand'] == batter_hand]

    # Filter by pitch types
    filtered_df = filtered_df[filtered_df['pitch_name'].isin(pitch_types)]

    if len(filtered_df) == 0:
        raise ValueError("No pitches match the specified filters")

    # Create figure
    fig = go.Figure()

    # Add strike zone and plate reference traces
    for trace in create_strike_zone_traces():
        fig.add_trace(trace)

    # Calculate stats for annotations
    stats_text = []

    # Plot trajectories for each pitch type
    for pitch_type in pitch_types:
        pitch_df = filtered_df[filtered_df['pitch_name'] == pitch_type]

        if len(pitch_df) == 0:
            continue

        # Sample pitches if too many
        if len(pitch_df) > max_pitches_per_type:
            pitch_df = pitch_df.sample(n=max_pitches_per_type, random_state=42)

        color = PITCH_COLORS.get(pitch_type, '#999999')

        # Calculate trajectories
        all_x, all_y, all_z = [], [], []
        release_x, release_z = [], []

        for idx, row in pitch_df.iterrows():
            try:
                x, y, z = calculate_pitch_trajectory(row)

                # Add trajectory line
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color=color, width=2),
                    opacity=0.6,
                    name=pitch_type,
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # Collect points for release point cluster
                release_x.append(row['release_pos_x'])
                release_z.append(row['release_pos_z'])

                # Store last point (plate location)
                all_x.append(x[-1])
                all_y.append(y[-1])
                all_z.append(z[-1])

            except Exception as e:
                continue

        # Add marker for average release point
        if release_x:
            avg_rel_x = np.mean(release_x)
            avg_rel_z = np.mean(release_z)
            rel_std = np.std(release_x)**2 + np.std(release_z)**2
            rel_std = np.sqrt(rel_std)

            fig.add_trace(go.Scatter3d(
                x=[avg_rel_x],
                y=[55],  # Approximate release distance
                z=[avg_rel_z],
                mode='markers',
                marker=dict(size=8, color=color, symbol='diamond'),
                name=f'{pitch_type} Release',
                showlegend=True
            ))

            # Collect stats
            avg_velo = pitch_df['release_speed'].mean()
            avg_spin = pitch_df['release_spin_rate'].mean()
            avg_h_break = pitch_df['pfx_x'].mean()
            avg_v_break = pitch_df['pfx_z'].mean()

            stats_text.append(
                f"<b>{pitch_type}</b><br>"
                f"Velo: {avg_velo:.1f} MPH<br>"
                f"Spin: {int(avg_spin)} RPM<br>"
                f"Movement: {avg_h_break:+.1f}\" H / {avg_v_break:+.1f}\" V<br>"
                f"Release Var: {rel_std:.2f} ft<br>"
                f"Pitches: {len(pitch_df)}"
            )

    # Layout configuration
    if title is None:
        hand_txt = f" vs {'RHH' if batter_hand == 'R' else 'LHH' if batter_hand == 'L' else 'All'}"
        title = f"{pitcher_name} - Pitch Tunneling Visualization{hand_txt}"

    # Add annotations
    annotation_text = "<br><br>".join(stats_text)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family='Arial, sans-serif')
        ),
        scene=dict(
            xaxis=dict(
                title='← 1B          Horizontal Position (ft)          3B →',
                range=[3, -3],  # Catcher's view: 1B on left (positive x), 3B on right (negative x)
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=2
            ),
            yaxis=dict(
                title='Distance from Plate (ft)',
                range=[0, 60],
                showgrid=True,
                gridcolor='lightgray'
            ),
            zaxis=dict(
                title='Height (ft)',
                range=[0, 6],
                showgrid=True,
                gridcolor='lightgray'
            ),
            camera=dict(
                eye=dict(x=0.1, y=-1.2, z=0.5),  # Slightly offset, behind plate, eye level
                center=dict(x=0, y=0.3, z=0.4),  # Focus on strike zone area
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=4, z=1.2)  # Better proportions for pitch visualization
        ),
        width=1920,
        height=1080,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        annotations=[
            dict(
                text=annotation_text,
                xref='paper', yref='paper',
                x=0.98, y=0.02,
                xanchor='right', yanchor='bottom',
                showarrow=False,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=11)
            )
        ]
    )

    # Save outputs
    if output_html:
        fig.write_html(output_html)
        print(f"✓ Saved interactive HTML: {output_html}")

    if output_png:
        try:
            fig.write_image(output_png, width=1920, height=1080, scale=2)
            print(f"✓ Saved PNG image: {output_png}")
        except Exception as e:
            print(f"⚠️  PNG export requires Chrome/Kaleido (use HTML for now): {output_png}")

    return fig


def analyze_pitch_sequences(
    df: pd.DataFrame,
    pitcher_name: str,
    min_sample_size: int = 20,
    success_metric: str = 'whiff_rate',
    batter_hand: Optional[str] = None,
    sequence_length: int = 2,
    sequence_position: str = 'any'
) -> pd.DataFrame:
    """
    Function 2: Sequence Analysis & Success Metrics

    Analyze and identify most effective pitch sequences

    Parameters:
    -----------
    df : pd.DataFrame
        Statcast pitch data
    pitcher_name : str
        Name of pitcher
    min_sample_size : int
        Minimum occurrences for a sequence to be included
    success_metric : str
        'whiff_rate', 'chase_rate', 'weak_contact', or 'overall'
    batter_hand : str, optional
        'R' or 'L' to filter by batter handedness
    sequence_length : int
        Length of sequences to analyze (2 or 3 pitches)
    sequence_position : str
        'any' - all consecutive sequences (default)
        'start' - only first 2 pitches of PA (excludes 1-pitch PAs)
        'end' - only last 2 pitches of PA (excludes 1-pitch PAs)

    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with top sequences and their success metrics
    """
    # Validate required columns exist
    required_cols = ['pitch_name', 'game_date', 'at_bat_number', 'pitch_number', 'description']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, returning empty DataFrame")
        return pd.DataFrame()
    
    # Filter by batter hand if specified
    filtered_df = df.copy()
    if batter_hand and 'stand' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['stand'] == batter_hand]

    # Create at-bat identifier
    filtered_df['at_bat_id'] = (
        filtered_df['game_date'].astype(str) + '_' +
        filtered_df['at_bat_number'].astype(str)
    )

    # Sort by at-bat and pitch number
    filtered_df = filtered_df.sort_values(['at_bat_id', 'pitch_number'])

    # Build sequences
    sequences = []

    for ab_id, ab_group in filtered_df.groupby('at_bat_id'):
        ab_group = ab_group.sort_values('pitch_number')
        pitches = ab_group['pitch_name'].values
        num_pitches = len(pitches)

        # Skip PAs with only 1 pitch for start/end filters
        if sequence_position in ['start', 'end'] and num_pitches < 2:
            continue

        # Determine which indices to use based on sequence_position
        if sequence_position == 'start':
            # Only the first 2 pitches (index 0)
            indices = [0] if num_pitches >= sequence_length else []
        elif sequence_position == 'end':
            # Only the last 2 pitches (last valid index)
            indices = [num_pitches - sequence_length] if num_pitches >= sequence_length else []
        else:  # 'any'
            # All consecutive sequences
            indices = range(num_pitches - sequence_length + 1)

        # Create sequences for selected indices
        for i in indices:
            # Filter out NaN values and convert to strings
            sequence_raw = pitches[i:i + sequence_length]
            sequence = tuple(str(p) for p in sequence_raw if pd.notna(p))

            # Skip if sequence is incomplete
            if len(sequence) != sequence_length:
                continue

            # Get outcome of final pitch in sequence
            final_pitch = ab_group.iloc[i + sequence_length - 1]

            # Determine if pitch was in zone (handle missing 'zone' column)
            if 'zone' in final_pitch.index and pd.notna(final_pitch['zone']):
                try:
                    in_zone = float(final_pitch['zone']) <= 9
                except (ValueError, TypeError):
                    in_zone = True  # Default to in-zone if can't determine
            else:
                # Estimate from plate location if available
                if 'plate_x' in final_pitch.index and 'plate_z' in final_pitch.index:
                    px, pz = final_pitch['plate_x'], final_pitch['plate_z']
                    in_zone = abs(px) <= 0.83 and 1.5 <= pz <= 3.5
                else:
                    in_zone = True  # Default assumption

            # Determine if swing
            swing = final_pitch['description'] in [
                'swinging_strike', 'foul', 'foul_tip',
                'hit_into_play', 'swinging_strike_blocked'
            ]

            # Determine outcomes
            whiff = final_pitch['description'] == 'swinging_strike'
            chase = (not in_zone) and swing
            
            # Check for weak contact (handle missing 'events' column)
            weak_contact = False
            if final_pitch['description'] == 'hit_into_play':
                if 'events' in final_pitch.index and pd.notna(final_pitch['events']):
                    weak_contact = final_pitch['events'] in ['field_out', 'force_out', 'grounded_into_double_play', 'double_play', 'sac_fly', 'fielders_choice']
                else:
                    # If no events column, estimate weak contact from other indicators
                    weak_contact = True  # Assume hit_into_play is weak contact as fallback

            sequences.append({
                'sequence': sequence,
                'sequence_str': ' → '.join(sequence),
                'swing': swing,
                'whiff': whiff,
                'in_zone': in_zone,
                'chase': chase,
                'weak_contact': weak_contact,
                'final_outcome': final_pitch['description']
            })

    # Create DataFrame
    seq_df = pd.DataFrame(sequences)

    if len(seq_df) == 0:
        return pd.DataFrame()

    # Calculate success metrics by sequence
    grouped = seq_df.groupby('sequence_str')

    results = []
    for seq_name, seq_group in grouped:
        if len(seq_group) < min_sample_size:
            continue

        total = len(seq_group)
        swings = seq_group['swing'].sum()
        whiffs = seq_group['whiff'].sum()

        outside_zone = (~seq_group['in_zone']).sum()
        chases = seq_group['chase'].sum()

        weak_contacts = seq_group['weak_contact'].sum()

        # Calculate rates
        whiff_rate = (whiffs / swings * 100) if swings > 0 else 0
        chase_rate = (chases / outside_zone * 100) if outside_zone > 0 else 0
        weak_contact_rate = (weak_contacts / total * 100)

        # Overall effectiveness score (weighted combination)
        overall_score = (whiff_rate * 0.5 + chase_rate * 0.3 + weak_contact_rate * 0.2)

        results.append({
            'Sequence': seq_name,
            'Usage': total,
            'Whiff Rate': round(whiff_rate, 1),
            'Chase Rate': round(chase_rate, 1),
            'Weak Contact Rate': round(weak_contact_rate, 1),
            'Overall Score': round(overall_score, 1)
        })

    results_df = pd.DataFrame(results)

    # Sort by selected metric
    sort_col = {
        'whiff_rate': 'Whiff Rate',
        'chase_rate': 'Chase Rate',
        'weak_contact': 'Weak Contact Rate',
        'overall': 'Overall Score'
    }.get(success_metric, 'Overall Score')

    results_df = results_df.sort_values(sort_col, ascending=False)

    return results_df


def create_sequence_chart(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    batter_hand: Optional[str] = None,
    top_n: int = 10,
    output_png: Optional[str] = None
) -> plt.Figure:
    """
    Create bar chart visualization of top pitch sequences

    Parameters:
    -----------
    sequence_df : pd.DataFrame
        Output from analyze_pitch_sequences()
    pitcher_name : str
        Name of pitcher
    batter_hand : str, optional
        Batter handedness for title
    top_n : int
        Number of top sequences to display
    output_png : str, optional
        Path to save PNG image

    Returns:
    --------
    fig : matplotlib Figure
    """
    # Take top N sequences
    plot_df = sequence_df.head(top_n).copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    # Bar positions
    x = np.arange(len(plot_df))
    width = 0.25

    # Create bars
    ax.bar(x - width, plot_df['Whiff Rate'], width, label='Whiff Rate', color='#E63946', alpha=0.8)
    ax.bar(x, plot_df['Chase Rate'], width, label='Chase Rate', color='#FFD166', alpha=0.8)
    ax.bar(x + width, plot_df['Weak Contact Rate'], width, label='Weak Contact Rate', color='#06AED5', alpha=0.8)

    # Customization
    hand_text = f" vs {'RHH' if batter_hand == 'R' else 'LHH' if batter_hand == 'L' else 'All Batters'}"
    ax.set_title(f'{pitcher_name} - Top Pitch Sequences{hand_text}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Pitch Sequence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Sequence'], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(plot_df[['Whiff Rate', 'Chase Rate', 'Weak Contact Rate']].max()) * 1.1)

    # Add usage annotations
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(i, -5, f"n={row['Usage']}", ha='center', va='top', fontsize=8, style='italic')

    plt.tight_layout()

    # Save if specified
    if output_png:
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sequence chart: {output_png}")

    return fig


def generate_portfolio_package(
    df: pd.DataFrame,
    pitcher_name: str,
    pitcher_id: int,
    output_dir: str,
    pitch_types_for_tunnel: List[str],
    batter_hands: List[str] = ['R', 'L'],
    include_interactive: bool = True
) -> Dict[str, str]:
    """
    Function 3: Portfolio Export Pipeline

    Generate complete portfolio package for a pitcher

    Parameters:
    -----------
    df : pd.DataFrame
        Complete Statcast data for pitcher
    pitcher_name : str
        Pitcher's name
    pitcher_id : int
        MLB ID
    output_dir : str
        Directory to save portfolio files
    pitch_types_for_tunnel : list of str
        Pitch types to feature in tunnel visualization
    batter_hands : list of str
        Batter handedness splits to analyze
    include_interactive : bool
        Whether to generate interactive HTML files

    Returns:
    --------
    manifest : dict
        Dictionary of generated files and their paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = {}

    print(f"\n{'='*80}")
    print(f"GENERATING PORTFOLIO: {pitcher_name}")
    print(f"{'='*80}\n")

    # 1. Generate hero tunnel visualization
    print("1. Creating 3D tunnel visualization...")
    hero_html = str(output_path / "interactive_demo.html") if include_interactive else None
    hero_png = str(output_path / "hero_image.png")

    try:
        visualize_pitch_trajectories_3d(
            df=df,
            pitcher_name=pitcher_name,
            pitch_types=pitch_types_for_tunnel,
            output_html=hero_html,
            output_png=hero_png,
            max_pitches_per_type=25
        )
        manifest['hero_png'] = hero_png
        if hero_html:
            manifest['hero_html'] = hero_html
    except Exception as e:
        print(f"  ⚠️  Error creating tunnel viz: {e}")

    # 2. Generate sequence analysis for each batter hand
    print("\n2. Analyzing pitch sequences...")

    for hand in batter_hands:
        hand_label = 'RHH' if hand == 'R' else 'LHH'
        print(f"   Analyzing vs {hand_label}...")

        # Analyze sequences
        seq_df = analyze_pitch_sequences(
            df=df,
            pitcher_name=pitcher_name,
            batter_hand=hand,
            min_sample_size=15,
            success_metric='overall'
        )

        if len(seq_df) > 0:
            # Save sequence data
            seq_csv = str(output_path / f"sequences_vs_{hand_label}.csv")
            seq_df.to_csv(seq_csv, index=False)
            manifest[f'sequences_{hand_label}_csv'] = seq_csv

            # Create chart
            chart_png = str(output_path / f"sequence_chart_vs_{hand_label}.png")
            create_sequence_chart(
                sequence_df=seq_df,
                pitcher_name=pitcher_name,
                batter_hand=hand,
                output_png=chart_png
            )
            manifest[f'sequences_{hand_label}_png'] = chart_png

    # 3. Generate summary statistics
    print("\n3. Generating summary statistics...")

    stats = {
        'pitcher_name': pitcher_name,
        'pitcher_id': pitcher_id,
        'total_pitches': len(df),
        'date_range': f"{df['game_date'].min()} to {df['game_date'].max()}",
        'arsenal': df.groupby('pitch_name').size().to_dict(),
        'avg_velocity_by_pitch': df.groupby('pitch_name')['release_speed'].mean().round(1).to_dict(),
        'whiff_rate_overall': round(
            len(df[df['description'] == 'swinging_strike']) /
            len(df[df['description'].isin(['swinging_strike', 'foul', 'foul_tip', 'hit_into_play', 'swinging_strike_blocked'])]) * 100,
            1
        ) if len(df[df['description'].isin(['swinging_strike', 'foul', 'foul_tip', 'hit_into_play', 'swinging_strike_blocked'])]) > 0 else 0
    }

    # Save metadata
    metadata_json = str(output_path / "metadata.json")
    with open(metadata_json, 'w') as f:
        json.dump(stats, f, indent=2)
    manifest['metadata'] = metadata_json

    # Save text summary
    summary_txt = str(output_path / "stats_summary.txt")
    with open(summary_txt, 'w') as f:
        f.write(f"PITCHER PORTFOLIO: {pitcher_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"MLB ID: {pitcher_id}\n")
        f.write(f"Total Pitches Analyzed: {stats['total_pitches']:,}\n")
        f.write(f"Date Range: {stats['date_range']}\n")
        f.write(f"Overall Whiff Rate: {stats['whiff_rate_overall']}%\n\n")
        f.write(f"PITCH ARSENAL:\n")
        f.write(f"{'-'*60}\n")
        for pitch, count in sorted(stats['arsenal'].items(), key=lambda x: x[1], reverse=True):
            pct = count / stats['total_pitches'] * 100
            avg_velo = stats['avg_velocity_by_pitch'].get(pitch, 0)
            f.write(f"{pitch:20s} {count:5d} ({pct:5.1f}%)  Avg: {avg_velo:5.1f} MPH\n")
        f.write(f"\nData Source: MLB Statcast via pybaseball\n")

    manifest['summary'] = summary_txt

    print(f"\n{'='*80}")
    print(f"PORTFOLIO COMPLETE: {len(manifest)} files generated")
    print(f"Output directory: {output_path}")
    print(f"{'='*80}\n")

    return manifest


if __name__ == "__main__":
    print("Pitch Visualization Module")
    print("Import this module to use visualization functions")
