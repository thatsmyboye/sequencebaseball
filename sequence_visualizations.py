"""
Pitch Sequence Visualization Module
Multiple visualization options for presenting pitch sequence effectiveness data

Available Visualizations:
1. grouped_bar - Traditional grouped bar chart (current default)
2. composite_score - Single-bar ranking by weighted composite score
3. heatmap_matrix - Pitch-to-pitch transition effectiveness matrix
4. scatter_bubble - Multi-dimensional scatter with bubble size
5. lollipop - Clean dot-and-stem visualization
6. small_multiples - Faceted charts by metric
7. interactive_table - HTML sortable table

Usage:
    from sequence_visualizations import create_sequence_visualization
    
    fig = create_sequence_visualization(
        sequence_df=df,
        pitcher_name="Tarik Skubal",
        chart_type="composite_score",
        batter_hand="R"
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import html

# Color schemes
METRIC_COLORS = {
    'whiff': '#E63946',      # Red - Whiff Rate
    'chase': '#FFD166',      # Yellow/Gold - Chase Rate  
    'weak_contact': '#06AED5', # Cyan - Weak Contact
    'composite': '#7B2D8E',  # Purple - Composite Score
}

# Effectiveness gradient (red=bad, yellow=mid, green=good)
EFFECTIVENESS_CMAP = LinearSegmentedColormap.from_list(
    'effectiveness', ['#E63946', '#FFD166', '#06A77D']
)


class CompositeScoreConfig:
    """
    Configuration for composite score calculation
    Allows customization of metric weights
    """
    def __init__(
        self,
        whiff_weight: float = 0.40,
        chase_weight: float = 0.35,
        weak_contact_weight: float = 0.25,
        min_sample_size: int = 10
    ):
        """
        Initialize composite score configuration
        
        Parameters:
        -----------
        whiff_weight : float
            Weight for whiff rate (0-1)
        chase_weight : float
            Weight for chase rate (0-1)
        weak_contact_weight : float
            Weight for weak contact rate (0-1)
        min_sample_size : int
            Minimum sample size for confidence adjustment
            
        Raises:
        -------
        ValueError
            If all weights are zero or negative
        """
        # Validate weights sum to positive value
        total = whiff_weight + chase_weight + weak_contact_weight
        if total <= 0:
            raise ValueError(
                "Weights must sum to a positive value. "
                f"Got: whiff={whiff_weight}, chase={chase_weight}, weak_contact={weak_contact_weight}"
            )
        
        # Normalize weights to sum to 1
        self.whiff_weight = whiff_weight / total
        self.chase_weight = chase_weight / total
        self.weak_contact_weight = weak_contact_weight / total
        self.min_sample_size = min_sample_size
    
    def calculate_score(
        self, 
        whiff_rate: float, 
        chase_rate: float, 
        weak_contact_rate: float,
        sample_size: int
    ) -> Tuple[float, float]:
        """
        Calculate composite score with confidence adjustment
        
        Returns:
        --------
        (raw_score, confidence_adjusted_score)
        """
        raw_score = (
            whiff_rate * self.whiff_weight +
            chase_rate * self.chase_weight +
            weak_contact_rate * self.weak_contact_weight
        )
        
        # Confidence adjustment: penalize small samples
        # Uses logarithmic scaling so large samples don't get excessive boost
        if sample_size < self.min_sample_size:
            confidence = sample_size / self.min_sample_size
        else:
            # Gentle boost for larger samples, capped at ~1.15x
            confidence = 1.0 + 0.1 * np.log10(sample_size / self.min_sample_size)
            confidence = min(confidence, 1.15)
        
        adjusted_score = raw_score * confidence
        
        return raw_score, adjusted_score


def create_sequence_visualization(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    chart_type: str = 'grouped_bar',
    batter_hand: Optional[str] = None,
    top_n: int = 10,
    output_path: Optional[str] = None,
    score_config: Optional[CompositeScoreConfig] = None,
    **kwargs
) -> plt.Figure:
    """
    Create pitch sequence visualization with selected chart type
    
    Parameters:
    -----------
    sequence_df : pd.DataFrame
        DataFrame from analyze_pitch_sequences() with columns:
        - Sequence, Usage, Whiff Rate, Chase Rate, Weak Contact Rate, Overall Score
    pitcher_name : str
        Name of pitcher for title
    chart_type : str
        One of: 'grouped_bar', 'composite_score', 'heatmap_matrix', 
        'scatter_bubble', 'lollipop', 'small_multiples'
    batter_hand : str, optional
        'R' or 'L' for title annotation
    top_n : int
        Number of sequences to display
    output_path : str, optional
        Path to save PNG file
    score_config : CompositeScoreConfig, optional
        Custom configuration for composite scoring
    **kwargs : dict
        Additional chart-specific parameters
        
    Returns:
    --------
    fig : matplotlib.Figure
    """
    if score_config is None:
        score_config = CompositeScoreConfig()
    
    chart_functions = {
        'grouped_bar': _create_grouped_bar,
        'composite_score': _create_composite_score,
        'heatmap_matrix': _create_heatmap_matrix,
        'scatter_bubble': _create_scatter_bubble,
        'lollipop': _create_lollipop,
        'small_multiples': _create_small_multiples,
    }
    
    if chart_type not in chart_functions:
        available = ', '.join(chart_functions.keys())
        raise ValueError(f"Unknown chart_type '{chart_type}'. Available: {available}")
    
    # Create the visualization
    fig = chart_functions[chart_type](
        sequence_df=sequence_df,
        pitcher_name=pitcher_name,
        batter_hand=batter_hand,
        top_n=top_n,
        score_config=score_config,
        **kwargs
    )
    
    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved {chart_type} chart: {output_path}")
    
    return fig


def _get_title_suffix(batter_hand: Optional[str]) -> str:
    """Get title suffix for batter handedness"""
    if batter_hand == 'R':
        return ' vs RHH'
    elif batter_hand == 'L':
        return ' vs LHH'
    return ''


# =============================================================================
# CHART TYPE 1: GROUPED BAR (Original)
# =============================================================================

def _create_grouped_bar(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    batter_hand: Optional[str],
    top_n: int,
    score_config: CompositeScoreConfig,
    **kwargs
) -> plt.Figure:
    """
    Traditional grouped bar chart showing all three metrics side by side
    """
    plot_df = sequence_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    
    x = np.arange(len(plot_df))
    width = 0.25
    
    bars1 = ax.bar(x - width, plot_df['Whiff Rate'], width, 
                   label='Whiff Rate', color=METRIC_COLORS['whiff'], alpha=0.85)
    bars2 = ax.bar(x, plot_df['Chase Rate'], width,
                   label='Chase Rate', color=METRIC_COLORS['chase'], alpha=0.85)
    bars3 = ax.bar(x + width, plot_df['Weak Contact Rate'], width,
                   label='Weak Contact Rate', color=METRIC_COLORS['weak_contact'], alpha=0.85)
    
    # Styling
    title_suffix = _get_title_suffix(batter_hand)
    ax.set_title(f'{pitcher_name} - Top Pitch Sequences{title_suffix}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Pitch Sequence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Sequence'], rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(plot_df[['Whiff Rate', 'Chase Rate', 'Weak Contact Rate']].max()) * 1.15)
    ax.set_xlim(-0.5, len(plot_df) - 0.5)
    
    # Sample size annotations
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(i, -3, f'n={row.Usage}', ha='center', va='top', 
                fontsize=9, style='italic', color='#666666')
    
    plt.tight_layout()
    return fig


# =============================================================================
# CHART TYPE 2: COMPOSITE SCORE RANKING (User Priority)
# =============================================================================

def _create_composite_score(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    batter_hand: Optional[str],
    top_n: int,
    score_config: CompositeScoreConfig,
    show_components: bool = True,
    show_confidence: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Single horizontal bar chart ranked by composite effectiveness score
    
    Features:
    - Weighted composite score combining all metrics
    - Color gradient based on score
    - Optional component breakdown
    - Confidence indicators based on sample size
    - Clear ranking visualization
    """
    plot_df = sequence_df.head(top_n).copy()
    
    # Calculate composite scores
    scores = []
    for _, row in plot_df.iterrows():
        raw, adjusted = score_config.calculate_score(
            row['Whiff Rate'],
            row['Chase Rate'],
            row['Weak Contact Rate'],
            row['Usage']
        )
        scores.append({
            'raw_score': raw,
            'adjusted_score': adjusted,
            'confidence': adjusted / raw if raw > 0 else 1.0
        })
    
    plot_df['Raw Score'] = [s['raw_score'] for s in scores]
    plot_df['Adjusted Score'] = [s['adjusted_score'] for s in scores]
    plot_df['Confidence'] = [s['confidence'] for s in scores]
    
    # Sort by adjusted score
    plot_df = plot_df.sort_values('Adjusted Score', ascending=True)
    
    # Create figure with space for components
    fig_width = 16 if show_components else 12
    fig, ax = plt.subplots(figsize=(fig_width, max(8, len(plot_df) * 0.6)), dpi=100)
    
    y = np.arange(len(plot_df))
    
    # Normalize scores for color mapping
    score_min = plot_df['Adjusted Score'].min()
    score_max = plot_df['Adjusted Score'].max()
    score_range = score_max - score_min if score_max != score_min else 1
    normalized = (plot_df['Adjusted Score'] - score_min) / score_range
    
    # Create horizontal bars with gradient colors
    colors = [EFFECTIVENESS_CMAP(n) for n in normalized]
    bars = ax.barh(y, plot_df['Adjusted Score'], color=colors, edgecolor='white', linewidth=1.5)
    
    # Add score labels on bars
    for i, bar in enumerate(bars):
        row = plot_df.iloc[i]
        # Score value
        score_text = f'{row["Adjusted Score"]:.1f}'
        text_x = bar.get_width() + 1
        ax.text(text_x, bar.get_y() + bar.get_height()/2, score_text,
                va='center', ha='left', fontsize=11, fontweight='bold')
        
        # Confidence indicator
        if show_confidence and row['Usage'] < 30:
            ax.text(text_x + 5, bar.get_y() + bar.get_height()/2, 
                    f'(n={row["Usage"]})', va='center', ha='left', 
                    fontsize=9, color='#888888', style='italic')
    
    # Component breakdown on right side
    if show_components:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y)
        
        component_labels = []
        for _, row in plot_df.iterrows():
            label = f"W:{row['Whiff Rate']:.0f}  C:{row['Chase Rate']:.0f}  WC:{row['Weak Contact Rate']:.0f}"
            component_labels.append(label)
        
        ax2.set_yticklabels(component_labels, fontsize=9, color='#555555')
        ax2.tick_params(axis='y', length=0)
    
    # Styling
    title_suffix = _get_title_suffix(batter_hand)
    ax.set_title(f'{pitcher_name} - Sequence Effectiveness Ranking{title_suffix}\n',
                 fontsize=16, fontweight='bold', pad=10)
    
    # Subtitle with weight info
    weight_info = (f'Score = {score_config.whiff_weight:.0%} Whiff + '
                   f'{score_config.chase_weight:.0%} Chase + '
                   f'{score_config.weak_contact_weight:.0%} Weak Contact')
    ax.text(0.5, 1.02, weight_info, transform=ax.transAxes, ha='center',
            fontsize=10, color='#666666', style='italic')
    
    ax.set_xlabel('Composite Effectiveness Score', fontsize=12, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Sequence'], fontsize=11)
    ax.set_xlim(0, plot_df['Adjusted Score'].max() * 1.25)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add rank numbers
    for i in range(len(plot_df)):
        rank = len(plot_df) - i
        ax.text(-2, i, f'#{rank}', va='center', ha='right', 
                fontsize=10, fontweight='bold', color='#444444')
    
    # Legend for confidence
    if show_confidence:
        legend_text = '* Lower sample sizes (n<30) shown in italics'
        ax.text(0.02, -0.06, legend_text, transform=ax.transAxes,
                fontsize=9, color='#888888', style='italic')
    
    plt.tight_layout()
    return fig


# =============================================================================
# CHART TYPE 3: HEATMAP MATRIX
# =============================================================================

def _create_heatmap_matrix(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    batter_hand: Optional[str],
    top_n: int,
    score_config: CompositeScoreConfig,
    metric: str = 'Overall Score',
    **kwargs
) -> plt.Figure:
    """
    Pitch-to-pitch transition matrix showing effectiveness
    Rows = "From" pitch, Columns = "To" pitch
    """
    # Parse sequences to get from/to pitches (respect top_n limit)
    transitions = []
    for _, row in sequence_df.head(top_n).iterrows():
        parts = row['Sequence'].split(' → ')
        if len(parts) == 2:
            transitions.append({
                'from': parts[0],
                'to': parts[1],
                'value': row[metric],
                'usage': row['Usage']
            })
    
    trans_df = pd.DataFrame(transitions)
    
    if len(trans_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No 2-pitch sequences found', ha='center', va='center')
        return fig
    
    # Create pivot table
    pivot = trans_df.pivot_table(
        index='from', 
        columns='to', 
        values='value', 
        aggfunc='mean'
    )
    
    # Also create usage matrix for annotations
    usage_pivot = trans_df.pivot_table(
        index='from',
        columns='to', 
        values='usage',
        aggfunc='sum'
    )
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    
    # Create heatmap
    mask = pivot.isna()
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap=EFFECTIVENESS_CMAP,
        mask=mask,
        ax=ax,
        cbar_kws={'label': metric},
        linewidths=2,
        linecolor='white',
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    
    # Add usage counts below each value
    for i, from_pitch in enumerate(pivot.index):
        for j, to_pitch in enumerate(pivot.columns):
            if not pd.isna(pivot.loc[from_pitch, to_pitch]):
                usage = usage_pivot.loc[from_pitch, to_pitch]
                ax.text(j + 0.5, i + 0.75, f'n={int(usage)}',
                       ha='center', va='center', fontsize=8, 
                       color='#666666', style='italic')
    
    # Styling
    title_suffix = _get_title_suffix(batter_hand)
    ax.set_title(f'{pitcher_name} - Pitch Sequence Matrix{title_suffix}\n{metric}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Second Pitch (To)', fontsize=12, fontweight='bold')
    ax.set_ylabel('First Pitch (From)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


# =============================================================================
# CHART TYPE 4: SCATTER BUBBLE
# =============================================================================

def _create_scatter_bubble(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    batter_hand: Optional[str],
    top_n: int,
    score_config: CompositeScoreConfig,
    x_metric: str = 'Chase Rate',
    y_metric: str = 'Whiff Rate',
    **kwargs
) -> plt.Figure:
    """
    Scatter plot with:
    - X: One metric (default: Chase Rate)
    - Y: Another metric (default: Whiff Rate)
    - Bubble size: Sample size
    - Color: Weak Contact Rate (or third metric)
    """
    plot_df = sequence_df.head(min(top_n, 15)).copy()  # Limit for readability
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    
    # Normalize sizes (min 100, max 1000)
    size_min, size_max = plot_df['Usage'].min(), plot_df['Usage'].max()
    if size_max == size_min:
        sizes = [400] * len(plot_df)
    else:
        sizes = 100 + 900 * (plot_df['Usage'] - size_min) / (size_max - size_min)
    
    # Create scatter
    scatter = ax.scatter(
        plot_df[x_metric],
        plot_df[y_metric],
        s=sizes,
        c=plot_df['Weak Contact Rate'],
        cmap=EFFECTIVENESS_CMAP,
        alpha=0.7,
        edgecolors='white',
        linewidth=2
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Weak Contact Rate (%)', fontsize=11)
    
    # Label each point
    for i, row in plot_df.iterrows():
        # Shorten sequence name for label
        seq_parts = row['Sequence'].split(' → ')
        short_name = ' → '.join([p[:3] for p in seq_parts])
        
        ax.annotate(
            short_name,
            (row[x_metric], row[y_metric]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )
    
    # Add quadrant lines (at median values)
    x_med = plot_df[x_metric].median()
    y_med = plot_df[y_metric].median()
    ax.axvline(x_med, color='#cccccc', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y_med, color='#cccccc', linestyle='--', linewidth=1, alpha=0.7)
    
    # Quadrant labels
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[1] * 0.95, 
            '★ BEST', ha='right', va='top', fontsize=10, 
            fontweight='bold', color='#06A77D', alpha=0.8)
    
    # Styling
    title_suffix = _get_title_suffix(batter_hand)
    ax.set_title(f'{pitcher_name} - Sequence Performance Map{title_suffix}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'{x_metric} (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{y_metric} (%)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Size legend
    size_legend = ax.legend(
        *scatter.legend_elements(prop="sizes", num=4, alpha=0.6),
        loc="lower right",
        title="Sample Size"
    )
    ax.add_artist(size_legend)
    
    plt.tight_layout()
    return fig


# =============================================================================
# CHART TYPE 5: LOLLIPOP CHART
# =============================================================================

def _create_lollipop(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    batter_hand: Optional[str],
    top_n: int,
    score_config: CompositeScoreConfig,
    metric: str = 'Overall Score',
    **kwargs
) -> plt.Figure:
    """
    Clean lollipop chart - dots connected by stems
    Great for clear value comparison
    """
    plot_df = sequence_df.head(top_n).sort_values(metric, ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.5)), dpi=100)
    
    y = np.arange(len(plot_df))
    values = plot_df[metric].values
    
    # Normalize for coloring
    v_min, v_max = values.min(), values.max()
    v_range = v_max - v_min if v_max != v_min else 1
    normalized = (values - v_min) / v_range
    colors = [EFFECTIVENESS_CMAP(n) for n in normalized]
    
    # Draw stems
    for i, (val, color) in enumerate(zip(values, colors)):
        ax.hlines(y=i, xmin=0, xmax=val, color=color, linewidth=2, alpha=0.7)
    
    # Draw dots
    ax.scatter(values, y, s=150, c=colors, edgecolors='white', linewidth=2, zorder=5)
    
    # Value labels
    for i, (val, row) in enumerate(zip(values, plot_df.itertuples())):
        ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
        ax.text(val + 6, i, f'(n={row.Usage})', va='center', fontsize=9, 
                color='#888888', style='italic')
    
    # Styling
    title_suffix = _get_title_suffix(batter_hand)
    ax.set_title(f'{pitcher_name} - Sequence {metric} Ranking{title_suffix}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(metric, fontsize=12, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Sequence'], fontsize=11)
    ax.set_xlim(0, values.max() * 1.3)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove y-axis line
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


# =============================================================================
# CHART TYPE 6: SMALL MULTIPLES
# =============================================================================

def _create_small_multiples(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    batter_hand: Optional[str],
    top_n: int,
    score_config: CompositeScoreConfig,
    **kwargs
) -> plt.Figure:
    """
    Faceted view - separate chart for each metric
    Each panel sorts sequences by its respective metric for easy comparison
    """
    plot_df = sequence_df.head(top_n).copy()
    
    metrics = ['Whiff Rate', 'Chase Rate', 'Weak Contact Rate']
    colors = [METRIC_COLORS['whiff'], METRIC_COLORS['chase'], METRIC_COLORS['weak_contact']]
    
    # Don't share y-axis since each panel has its own sort order
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=100, sharey=False)
    
    for ax, metric, color in zip(axes, metrics, colors):
        # Sort by this metric for this panel
        sorted_df = plot_df.sort_values(metric, ascending=True).reset_index(drop=True)
        values = sorted_df[metric].values
        
        # Create y positions for this panel's sorted data
        y = np.arange(len(sorted_df))
        
        # Horizontal bars
        bars = ax.barh(y, values, color=color, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Value labels
        for i, (val, usage) in enumerate(zip(values, sorted_df['Usage'])):
            ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=10)
            ax.text(val + 6, i, f'n={usage}', va='center', fontsize=8, 
                    color='#888888', style='italic')
        
        # Styling
        ax.set_title(metric, fontsize=14, fontweight='bold', color=color)
        ax.set_xlabel('Rate (%)', fontsize=11)
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_df['Sequence'], fontsize=10)
        ax.set_xlim(0, max(values) * 1.35)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Overall title
    title_suffix = _get_title_suffix(batter_hand)
    fig.suptitle(f'{pitcher_name} - Sequence Metrics Breakdown{title_suffix}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


# =============================================================================
# INTERACTIVE HTML TABLE
# =============================================================================

def create_interactive_table(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    batter_hand: Optional[str] = None,
    output_path: Optional[str] = None,
    score_config: Optional[CompositeScoreConfig] = None
) -> str:
    """
    Create an interactive sortable HTML table
    
    Returns:
    --------
    html_content : str
        HTML string of the table
    """
    if score_config is None:
        score_config = CompositeScoreConfig()
    
    title_suffix = _get_title_suffix(batter_hand)
    
    # Recalculate adjusted scores
    adjusted_scores = []
    for _, row in sequence_df.iterrows():
        _, adjusted = score_config.calculate_score(
            row['Whiff Rate'], row['Chase Rate'], 
            row['Weak Contact Rate'], row['Usage']
        )
        adjusted_scores.append(round(adjusted, 1))
    
    df = sequence_df.copy()
    df['Adjusted Score'] = adjusted_scores
    df = df.sort_values('Adjusted Score', ascending=False)
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>{pitcher_name} - Pitch Sequences{title_suffix}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #7B2D8E;
            padding-bottom: 10px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: linear-gradient(135deg, #7B2D8E 0%, #5a1f6a 100%);
            color: white;
            padding: 15px 10px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            position: relative;
        }}
        th:hover {{
            background: linear-gradient(135deg, #8e3da3 0%, #6b2a7d 100%);
        }}
        th::after {{
            content: ' ⇅';
            opacity: 0.5;
            font-size: 12px;
        }}
        th.sort-asc::after {{ content: ' ▲'; opacity: 1; }}
        th.sort-desc::after {{ content: ' ▼'; opacity: 1; }}
        td {{
            padding: 12px 10px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f4ff;
        }}
        .metric-whiff {{ color: #E63946; font-weight: bold; }}
        .metric-chase {{ color: #D4A106; font-weight: bold; }}
        .metric-weak {{ color: #0891A8; font-weight: bold; }}
        .metric-score {{ color: #7B2D8E; font-weight: bold; font-size: 1.1em; }}
        .usage {{ color: #888; font-style: italic; }}
        .rank {{
            background: #7B2D8E;
            color: white;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
        }}
        .weights {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
    </style>
</head>
<body>
    <h1>{pitcher_name} - Pitch Sequence Analysis{title_suffix}</h1>
    <div class="weights">
        <strong>Composite Score Weights:</strong> 
        Whiff {score_config.whiff_weight:.0%} + 
        Chase {score_config.chase_weight:.0%} + 
        Weak Contact {score_config.weak_contact_weight:.0%}
    </div>
    <table id="sequenceTable">
        <thead>
            <tr>
                <th data-type="number">Rank</th>
                <th data-type="string">Sequence</th>
                <th data-type="number">Usage</th>
                <th data-type="number">Whiff %</th>
                <th data-type="number">Chase %</th>
                <th data-type="number">Weak Contact %</th>
                <th data-type="number">Score</th>
            </tr>
        </thead>
        <tbody>
'''
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        # Escape sequence text to prevent XSS if data contains special characters
        escaped_sequence = html.escape(str(row['Sequence']))
        html_content += f'''            <tr>
                <td><span class="rank">{i}</span></td>
                <td>{escaped_sequence}</td>
                <td class="usage">n={row['Usage']}</td>
                <td class="metric-whiff">{row['Whiff Rate']:.1f}</td>
                <td class="metric-chase">{row['Chase Rate']:.1f}</td>
                <td class="metric-weak">{row['Weak Contact Rate']:.1f}</td>
                <td class="metric-score">{row['Adjusted Score']:.1f}</td>
            </tr>
'''
    
    html_content += '''        </tbody>
    </table>
    
    <script>
        // Simple table sorting
        document.querySelectorAll('th').forEach(th => {
            th.addEventListener('click', () => {
                const table = th.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const idx = Array.from(th.parentNode.children).indexOf(th);
                const type = th.dataset.type;
                const asc = !th.classList.contains('sort-asc');
                
                // Clear other sorts
                th.parentNode.querySelectorAll('th').forEach(h => {
                    h.classList.remove('sort-asc', 'sort-desc');
                });
                th.classList.add(asc ? 'sort-asc' : 'sort-desc');
                
                rows.sort((a, b) => {
                    let aVal, bVal;
                    
                    if (type === 'number') {
                        // Extract numeric content for number columns
                        aVal = parseFloat(a.children[idx].textContent.replace(/[^\\d.-]/g, '')) || 0;
                        bVal = parseFloat(b.children[idx].textContent.replace(/[^\\d.-]/g, '')) || 0;
                    } else {
                        // Use full text content for string columns
                        aVal = a.children[idx].textContent.trim().toLowerCase();
                        bVal = b.children[idx].textContent.trim().toLowerCase();
                    }
                    
                    if (aVal < bVal) return asc ? -1 : 1;
                    if (aVal > bVal) return asc ? 1 : -1;
                    return 0;
                });
                
                rows.forEach(row => tbody.appendChild(row));
                
                // Update ranks
                tbody.querySelectorAll('tr').forEach((row, i) => {
                    row.querySelector('.rank').textContent = i + 1;
                });
            });
        });
    </script>
</body>
</html>'''
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Saved interactive table: {output_path}")
    
    return html_content


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_chart_types() -> Dict[str, str]:
    """Return available chart types with descriptions"""
    return {
        'grouped_bar': 'Traditional grouped bar chart showing all metrics side by side',
        'composite_score': 'Horizontal bar ranked by weighted composite effectiveness score',
        'heatmap_matrix': 'Pitch-to-pitch transition matrix showing effectiveness',
        'scatter_bubble': 'Multi-dimensional scatter plot (x/y metrics, size=usage, color=third metric)',
        'lollipop': 'Clean dot-and-stem chart for clear value comparison',
        'small_multiples': 'Faceted view with separate chart for each metric',
    }


def create_all_visualizations(
    sequence_df: pd.DataFrame,
    pitcher_name: str,
    output_dir: str,
    batter_hand: Optional[str] = None,
    top_n: int = 10
) -> Dict[str, str]:
    """
    Generate all visualization types and save to directory
    
    Returns:
    --------
    manifest : dict
        Mapping of chart type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hand_suffix = f'_vs_{batter_hand}HH' if batter_hand else ''
    manifest = {}
    
    chart_types = get_available_chart_types().keys()
    
    for chart_type in chart_types:
        try:
            filename = f'sequence_{chart_type}{hand_suffix}.png'
            filepath = str(output_path / filename)
            
            create_sequence_visualization(
                sequence_df=sequence_df,
                pitcher_name=pitcher_name,
                chart_type=chart_type,
                batter_hand=batter_hand,
                top_n=top_n,
                output_path=filepath
            )
            manifest[chart_type] = filepath
            
        except Exception as e:
            print(f"⚠️  Error creating {chart_type}: {e}")
    
    # Also create interactive table
    table_path = str(output_path / f'sequence_table{hand_suffix}.html')
    create_interactive_table(
        sequence_df=sequence_df,
        pitcher_name=pitcher_name,
        batter_hand=batter_hand,
        output_path=table_path
    )
    manifest['interactive_table'] = table_path
    
    print(f"\n✓ Generated {len(manifest)} visualizations in {output_dir}")
    return manifest


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PITCH SEQUENCE VISUALIZATION MODULE")
    print("="*80)
    
    print("\nAvailable Chart Types:")
    print("-"*60)
    for chart_type, desc in get_available_chart_types().items():
        print(f"  • {chart_type:20s} - {desc}")
    
    print("\n" + "-"*60)
    print("USAGE EXAMPLE:")
    print("-"*60)
    print('''
from pitch_viz import analyze_pitch_sequences
from sequence_visualizations import (
    create_sequence_visualization,
    create_interactive_table,
    CompositeScoreConfig
)
import pandas as pd

# Load data
df = pd.read_csv('data/skubal_statcast_2024.csv')

# Analyze sequences
seq_df = analyze_pitch_sequences(df, "Tarik Skubal", batter_hand='R')

# Create composite score chart (recommended)
fig = create_sequence_visualization(
    sequence_df=seq_df,
    pitcher_name="Tarik Skubal",
    chart_type="composite_score",
    batter_hand="R",
    output_path="skubal_composite_score.png"
)

# Customize scoring weights
custom_config = CompositeScoreConfig(
    whiff_weight=0.5,    # Prioritize whiffs
    chase_weight=0.3,
    weak_contact_weight=0.2
)

fig = create_sequence_visualization(
    sequence_df=seq_df,
    pitcher_name="Tarik Skubal", 
    chart_type="composite_score",
    score_config=custom_config,
    output_path="skubal_custom_score.png"
)

# Create interactive HTML table
create_interactive_table(
    sequence_df=seq_df,
    pitcher_name="Tarik Skubal",
    batter_hand="R",
    output_path="skubal_table.html"
)
''')
    
    print("="*80)

