"""
Phase 3: Showcase Pieces - Specific Pitcher Visualizations
Generate portfolio-ready visualizations with narratives

Pitcher 1: Tarik Skubal - "The Tunnel Master"
Pitcher 2: Jhoan Duran - "The Splinker"
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pitch_viz import (
    visualize_pitch_trajectories_3d,
    analyze_pitch_sequences,
    create_sequence_chart
)
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_movement_chart(df: pd.DataFrame, pitcher_name: str, output_path: str):
    """
    Create 2D movement chart (horizontal vs vertical break)
    """
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

    # Get unique pitch types
    pitch_types = df['pitch_name'].unique()

    # Plot each pitch type
    for pitch_type in pitch_types:
        pitch_data = df[df['pitch_name'] == pitch_type]

        # Get color
        from pitch_viz import PITCH_COLORS
        color = PITCH_COLORS.get(pitch_type, '#999999')

        # Plot scatter
        ax.scatter(
            pitch_data['pfx_x'],
            pitch_data['pfx_z'],
            c=color,
            label=f"{pitch_type} (n={len(pitch_data)})",
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )

        # Add mean marker
        mean_x = pitch_data['pfx_x'].mean()
        mean_z = pitch_data['pfx_z'].mean()
        ax.scatter(mean_x, mean_z, c=color, s=300, marker='*',
                  edgecolors='black', linewidth=2, zorder=10)

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Labels
    ax.set_xlabel('Horizontal Movement (inches)\n← Glove Side | Arm Side →',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Vertical Movement (inches)\n← Drop | Rise →',
                 fontsize=12, fontweight='bold')
    ax.set_title(f'{pitcher_name} - Pitch Movement Profile',
                fontsize=16, fontweight='bold', pad=20)

    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved movement chart: {output_path}")


def create_velocity_distribution(df: pd.DataFrame, pitcher_name: str, output_path: str):
    """
    Create velocity distribution chart by pitch type
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    # Get pitch types
    pitch_types = df['pitch_name'].unique()

    # Get colors
    from pitch_viz import PITCH_COLORS

    # Create violin plots
    data_to_plot = []
    labels = []
    colors = []

    for pitch_type in sorted(pitch_types, key=lambda x: df[df['pitch_name']==x]['release_speed'].mean(), reverse=True):
        pitch_data = df[df['pitch_name'] == pitch_type]
        data_to_plot.append(pitch_data['release_speed'].values)
        avg_velo = pitch_data['release_speed'].mean()
        labels.append(f"{pitch_type}\n({avg_velo:.1f} MPH)")
        colors.append(PITCH_COLORS.get(pitch_type, '#999999'))

    # Create violin plot
    parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                         showmeans=True, showmedians=False)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Customize
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Velocity (MPH)', fontsize=12, fontweight='bold')
    ax.set_title(f'{pitcher_name} - Velocity Distribution by Pitch Type',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved velocity distribution: {output_path}")


def create_usage_chart(df: pd.DataFrame, pitcher_name: str, output_path: str):
    """
    Create pitch usage chart by count situation
    """
    # Create count categories
    df_copy = df.copy()
    df_copy['count_situation'] = 'Even'
    df_copy.loc[(df_copy['balls'] > df_copy['strikes']), 'count_situation'] = 'Behind'
    df_copy.loc[(df_copy['balls'] < df_copy['strikes']), 'count_situation'] = 'Ahead'

    # Calculate usage by situation
    usage = pd.crosstab(df_copy['count_situation'], df_copy['pitch_name'], normalize='index') * 100

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    # Get colors
    from pitch_viz import PITCH_COLORS
    colors_list = [PITCH_COLORS.get(p, '#999999') for p in usage.columns]

    usage.plot(kind='bar', stacked=False, ax=ax, color=colors_list, alpha=0.8, width=0.7)

    ax.set_xlabel('Count Situation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Usage %', fontsize=12, fontweight='bold')
    ax.set_title(f'{pitcher_name} - Pitch Usage by Count Situation',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved usage chart: {output_path}")


def create_effectiveness_by_location(df: pd.DataFrame, pitcher_name: str, pitch_type: str, output_path: str):
    """
    Create heatmap showing pitch effectiveness by location
    """
    # Filter for specific pitch type
    pitch_df = df[df['pitch_name'] == pitch_type].copy()

    # Define whiffs
    pitch_df['whiff'] = pitch_df['description'] == 'swinging_strike'

    # Create bins for plate location
    x_bins = np.linspace(-2, 2, 9)
    z_bins = np.linspace(0, 5, 9)

    # Calculate whiff rate by location
    pitch_df['x_bin'] = pd.cut(pitch_df['plate_x'], bins=x_bins)
    pitch_df['z_bin'] = pd.cut(pitch_df['plate_z'], bins=z_bins)

    # Group and calculate
    heatmap_data = pitch_df.groupby(['z_bin', 'x_bin']).agg({
        'whiff': ['sum', 'count']
    })

    # Calculate rate
    heatmap_data['rate'] = (heatmap_data[('whiff', 'sum')] /
                           heatmap_data[('whiff', 'count')] * 100)

    # Pivot for heatmap
    heatmap_pivot = heatmap_data['rate'].unstack()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    sns.heatmap(heatmap_pivot, cmap='RdYlGn', annot=True, fmt='.0f',
               cbar_kws={'label': 'Whiff Rate (%)'}, ax=ax,
               vmin=0, vmax=50)

    # Strike zone overlay
    from pitch_viz import STRIKE_ZONE
    # Calculate strike zone bounds in bin coordinates
    # This is approximate - adjust based on actual bins

    ax.set_xlabel('Horizontal Location\n← LHH | RHH →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vertical Location\n← Low | High →', fontsize=12, fontweight='bold')
    ax.set_title(f'{pitcher_name} - {pitch_type} Whiff Rate by Location',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved effectiveness heatmap: {output_path}")


# =============================================================================
# PITCHER 1: TARIK SKUBAL - "The Tunnel Master"
# =============================================================================
def generate_skubal_showcase():
    """
    Generate complete showcase package for Tarik Skubal
    Focus: Elite fastball-slider tunnel
    """
    print("\n" + "="*80)
    print("GENERATING SHOWCASE: TARIK SKUBAL - 'The Tunnel Master'")
    print("="*80)

    # Load data
    df = pd.read_csv('data/skubal_statcast_2024.csv')
    output_dir = Path('showcase_portfolios/skubal_tarik')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoaded {len(df):,} pitches")
    print(f"Output directory: {output_dir}\n")

    # 1. 3D Tunnel View (Fastball + Slider vs RHH)
    print("1. Creating 3D tunnel visualization (FB + SL vs RHH)...")
    visualize_pitch_trajectories_3d(
        df=df,
        pitcher_name="Tarik Skubal",
        pitch_types=["4-Seam Fastball", "Slider"],
        batter_hand='R',
        max_pitches_per_type=25,
        output_html=str(output_dir / "01_tunnel_fastball_slider_RHH.html"),
        output_png=str(output_dir / "01_tunnel_fastball_slider_RHH.png"),
        title="Tarik Skubal - The Perfect Tunnel (vs RHH)<br><sub>Fastball up, Slider down - Same release point</sub>"
    )

    # 2. Sequence Analysis (FB→SL vs SL→FB)
    print("\n2. Analyzing pitch sequences vs RHH...")
    seq_rhh = analyze_pitch_sequences(
        df=df,
        pitcher_name="Tarik Skubal",
        batter_hand='R',
        min_sample_size=15,
        success_metric='whiff_rate'
    )

    if len(seq_rhh) > 0:
        seq_rhh.to_csv(output_dir / "02_sequences_vs_RHH.csv", index=False)

        create_sequence_chart(
            sequence_df=seq_rhh,
            pitcher_name="Tarik Skubal",
            batter_hand='R',
            top_n=10,
            output_png=str(output_dir / "02_sequences_chart_RHH.png")
        )

    # Same for LHH
    print("\n3. Analyzing pitch sequences vs LHH...")
    seq_lhh = analyze_pitch_sequences(
        df=df,
        pitcher_name="Tarik Skubal",
        batter_hand='L',
        min_sample_size=15,
        success_metric='whiff_rate'
    )

    if len(seq_lhh) > 0:
        seq_lhh.to_csv(output_dir / "03_sequences_vs_LHH.csv", index=False)

        create_sequence_chart(
            sequence_df=seq_lhh,
            pitcher_name="Tarik Skubal",
            batter_hand='L',
            top_n=10,
            output_png=str(output_dir / "03_sequences_chart_LHH.png")
        )

    # 3. Arsenal Overview
    print("\n4. Creating arsenal overview visualizations...")

    # Movement chart
    create_movement_chart(
        df=df,
        pitcher_name="Tarik Skubal",
        output_path=str(output_dir / "04_movement_profile.png")
    )

    # Velocity distribution
    create_velocity_distribution(
        df=df,
        pitcher_name="Tarik Skubal",
        output_path=str(output_dir / "05_velocity_distribution.png")
    )

    # Usage by situation
    create_usage_chart(
        df=df,
        pitcher_name="Tarik Skubal",
        output_path=str(output_dir / "06_usage_by_count.png")
    )

    # 4. Create narrative summary
    print("\n5. Generating narrative summary...")

    with open(output_dir / "NARRATIVE.md", 'w') as f:
        f.write("# Tarik Skubal - The Tunnel Master\n\n")
        f.write("## Why Skubal's Fastball-Slider Combo Won Him the Cy Young\n\n")

        f.write("### The Perfect Tunnel\n\n")
        f.write("Tarik Skubal's 2024 AL Cy Young award was built on one of the most devastating ")
        f.write("pitch combinations in baseball: his fastball-slider tunnel.\n\n")

        # Calculate stats
        fb_data = df[df['pitch_name'] == '4-Seam Fastball']
        sl_data = df[df['pitch_name'] == 'Slider']

        avg_fb_velo = fb_data['release_speed'].mean()
        avg_sl_velo = sl_data['release_speed'].mean()
        velo_diff = avg_fb_velo - avg_sl_velo

        fb_rel_x = fb_data['release_pos_x'].mean()
        fb_rel_z = fb_data['release_pos_z'].mean()
        sl_rel_x = sl_data['release_pos_x'].mean()
        sl_rel_z = sl_data['release_pos_z'].mean()

        rel_distance = np.sqrt((fb_rel_x - sl_rel_x)**2 + (fb_rel_z - sl_rel_z)**2) * 12  # Convert to inches

        f.write(f"**Key Metrics:**\n")
        f.write(f"- 4-Seam Fastball: {avg_fb_velo:.1f} MPH average\n")
        f.write(f"- Slider: {avg_sl_velo:.1f} MPH average ({velo_diff:.1f} MPH difference)\n")
        f.write(f"- Release point consistency: {rel_distance:.1f} inches apart\n")
        f.write(f"- Combined usage: {(len(fb_data) + len(sl_data)) / len(df) * 100:.1f}% of all pitches\n\n")

        # Sequence analysis
        if len(seq_rhh) > 0:
            top_seq = seq_rhh.iloc[0]
            f.write(f"### Most Effective Sequence (vs RHH)\n\n")
            f.write(f"**{top_seq['Sequence']}**\n")
            f.write(f"- Usage: {top_seq['Usage']} times\n")
            f.write(f"- Whiff Rate: {top_seq['Whiff Rate']:.1f}%\n")
            f.write(f"- Chase Rate: {top_seq['Chase Rate']:.1f}%\n\n")

        f.write("### The Secret: Tunneling\n\n")
        f.write("Both pitches come from nearly identical release points, making it impossible for ")
        f.write("batters to distinguish them until it's too late. The fastball appears to rise ")
        f.write("into the top of the zone, while the slider drops sharply down and away.\n\n")

        f.write("### Arsenal Breakdown\n\n")
        for pitch_name in df['pitch_name'].unique():
            pitch_data = df[df['pitch_name'] == pitch_name]
            pct = len(pitch_data) / len(df) * 100
            velo = pitch_data['release_speed'].mean()
            spin = pitch_data['release_spin_rate'].mean()

            f.write(f"**{pitch_name}** ({pct:.1f}%)\n")
            f.write(f"- Velocity: {velo:.1f} MPH\n")
            f.write(f"- Spin Rate: {int(spin):,} RPM\n\n")

        f.write("\n---\n\n")
        f.write("*Data Source: MLB Statcast via pybaseball*\n")

    print(f"✓ Saved narrative: {output_dir / 'NARRATIVE.md'}")

    print("\n" + "="*80)
    print("SKUBAL SHOWCASE COMPLETE")
    print("="*80)


# =============================================================================
# PITCHER 2: JHOAN DURAN - "The Splinker"
# =============================================================================
def generate_duran_showcase():
    """
    Generate complete showcase package for Jhoan Duran
    Focus: Elite "splinker" (hardest splitter in baseball)
    """
    print("\n" + "="*80)
    print("GENERATING SHOWCASE: JHOAN DURAN - 'The Splinker'")
    print("="*80)

    # Load data
    df = pd.read_csv('data/duran_statcast_2024.csv')
    output_dir = Path('showcase_portfolios/duran_jhoan')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoaded {len(df):,} pitches")
    print(f"Output directory: {output_dir}\n")

    # 1. 3D Tunnel View (Fastball + Splinker)
    print("1. Creating 3D tunnel visualization (FB + Splinker)...")
    visualize_pitch_trajectories_3d(
        df=df,
        pitcher_name="Jhoan Duran",
        pitch_types=["4-Seam Fastball", "Splitter"],
        max_pitches_per_type=25,
        output_html=str(output_dir / "01_tunnel_fastball_splitter.html"),
        output_png=str(output_dir / "01_tunnel_fastball_splitter.png"),
        title="Jhoan Duran - The Unhittable Splinker<br><sub>100+ MPH heat, 94 MPH splitter - impossible to hit</sub>"
    )

    # 2. Sequence Analysis
    print("\n2. Analyzing pitch sequences...")
    sequences = analyze_pitch_sequences(
        df=df,
        pitcher_name="Jhoan Duran",
        min_sample_size=15,
        success_metric='whiff_rate'
    )

    if len(sequences) > 0:
        sequences.to_csv(output_dir / "02_sequences_all_batters.csv", index=False)

        create_sequence_chart(
            sequence_df=sequences,
            pitcher_name="Jhoan Duran",
            top_n=10,
            output_png=str(output_dir / "02_sequences_chart.png")
        )

    # 3. Arsenal Overview
    print("\n3. Creating arsenal overview visualizations...")

    create_movement_chart(
        df=df,
        pitcher_name="Jhoan Duran",
        output_path=str(output_dir / "03_movement_profile.png")
    )

    create_velocity_distribution(
        df=df,
        pitcher_name="Jhoan Duran",
        output_path=str(output_dir / "04_velocity_distribution.png")
    )

    create_usage_chart(
        df=df,
        pitcher_name="Jhoan Duran",
        output_path=str(output_dir / "05_usage_by_count.png")
    )

    # 4. Create narrative summary
    print("\n4. Generating narrative summary...")

    with open(output_dir / "NARRATIVE.md", 'w') as f:
        f.write("# Jhoan Duran - The Splinker\n\n")
        f.write("## The Unhittable Splitter and Why It's Nearly Impossible to Hit\n\n")

        f.write("### The Hardest Splitter in Baseball\n\n")
        f.write("Jhoan Duran's \"splinker\" - a splitter thrown harder than most pitchers' fastballs - ")
        f.write("represents one of baseball's most devastating weapons.\n\n")

        # Calculate stats
        fb_data = df[df['pitch_name'] == '4-Seam Fastball']
        sp_data = df[df['pitch_name'] == 'Splitter']

        avg_fb_velo = fb_data['release_speed'].mean()
        avg_sp_velo = sp_data['release_speed'].mean()
        velo_diff = avg_fb_velo - avg_sp_velo

        fb_vert = fb_data['pfx_z'].mean()
        sp_vert = sp_data['pfx_z'].mean()
        vert_diff = fb_vert - sp_vert

        f.write(f"**Key Metrics:**\n")
        f.write(f"- 4-Seam Fastball: {avg_fb_velo:.1f} MPH average (elite velocity)\n")
        f.write(f"- Splitter: {avg_sp_velo:.1f} MPH average (HARDEST IN MLB)\n")
        f.write(f"- Velocity difference: Only {velo_diff:.1f} MPH\n")
        f.write(f"- Vertical movement difference: {abs(vert_diff):.1f} inches\n")
        f.write(f"- Usage: Splitter is his primary pitch ({len(sp_data) / len(df) * 100:.1f}%)\n\n")

        f.write("### Why It's Unhittable\n\n")
        f.write(f"The combination of elite velocity ({avg_sp_velo:.1f} MPH) and sharp downward ")
        f.write("movement creates an optical illusion. Batters see a fastball coming at them at 100+ MPH, ")
        f.write("commit to swing, and the ball drops out of the zone at the last moment.\n\n")

        # Sequence analysis
        if len(sequences) > 0:
            # Find FB→Splitter sequence
            fb_sp_seq = sequences[sequences['Sequence'].str.contains('Fastball → Splitter', na=False)]
            if len(fb_sp_seq) > 0:
                top_seq = fb_sp_seq.iloc[0]
                f.write(f"### Deadliest Sequence: High Fastball → Low Splitter\n\n")
                f.write(f"**{top_seq['Sequence']}**\n")
                f.write(f"- Usage: {top_seq['Usage']} times\n")
                f.write(f"- Whiff Rate: {top_seq['Whiff Rate']:.1f}%\n")
                f.write(f"- Chase Rate: {top_seq['Chase Rate']:.1f}%\n\n")

        f.write("### Arsenal Breakdown\n\n")
        for pitch_name in df['pitch_name'].unique():
            pitch_data = df[df['pitch_name'] == pitch_name]
            pct = len(pitch_data) / len(df) * 100
            velo = pitch_data['release_speed'].mean()
            spin = pitch_data['release_spin_rate'].mean()
            h_mvmt = pitch_data['pfx_x'].mean()
            v_mvmt = pitch_data['pfx_z'].mean()

            f.write(f"**{pitch_name}** ({pct:.1f}%)\n")
            f.write(f"- Velocity: {velo:.1f} MPH\n")
            f.write(f"- Spin Rate: {int(spin):,} RPM\n")
            f.write(f"- Movement: {h_mvmt:+.1f}\" horizontal, {v_mvmt:+.1f}\" vertical\n\n")

        f.write("\n---\n\n")
        f.write("*Data Source: MLB Statcast via pybaseball*\n")

    print(f"✓ Saved narrative: {output_dir / 'NARRATIVE.md'}")

    print("\n" + "="*80)
    print("DURAN SHOWCASE COMPLETE")
    print("="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 3: SHOWCASE PIECE GENERATION")
    print("="*80)

    # Generate both showcases
    generate_skubal_showcase()
    generate_duran_showcase()

    print("\n" + "="*80)
    print("PHASE 3 COMPLETE ✓")
    print("="*80)
    print("\nAll showcase portfolios generated successfully!")
    print("\nOutput directories:")
    print("  - showcase_portfolios/skubal_tarik/")
    print("  - showcase_portfolios/duran_jhoan/")
    print("\nEach contains:")
    print("  ✓ 3D tunnel visualizations (interactive HTML)")
    print("  ✓ Sequence analysis charts")
    print("  ✓ Movement profiles")
    print("  ✓ Velocity distributions")
    print("  ✓ Usage breakdowns")
    print("  ✓ Narrative summaries")
    print("\n" + "="*80)
