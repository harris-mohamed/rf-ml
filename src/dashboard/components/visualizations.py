"""Plotly visualization functions for RF-ML dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

import sys
from pathlib import Path

# Add parent directory to path for imports
dashboard_dir = Path(__file__).parent.parent
if str(dashboard_dir) not in sys.path:
    sys.path.insert(0, str(dashboard_dir))

from utils.helpers import get_color_palette, NUM_TRANSPONDERS


def plot_cluster_pie_chart(cluster_stats: dict) -> go.Figure:
    """
    Create a pie chart showing cluster distribution.

    Args:
        cluster_stats: Dictionary from clustering.get_cluster_statistics()

    Returns:
        Plotly Figure object
    """
    if not cluster_stats or 'cluster_counts' not in cluster_stats:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    labels = [f"Cluster {label}" for label in sorted(cluster_stats['cluster_counts'].keys())]
    values = [cluster_stats['cluster_counts'][label] for label in sorted(cluster_stats['cluster_counts'].keys())]
    percentages = [cluster_stats['cluster_percentages'][label] for label in sorted(cluster_stats['cluster_counts'].keys())]

    # Get colors
    colors = get_color_palette(len(labels))

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text=f"Cluster Distribution ({cluster_stats['total_samples']} snapshots)",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        showlegend=True,
        height=500,
        margin=dict(t=80, b=40, l=40, r=40)
    )

    return fig


def plot_waterfall(
    time_array: np.ndarray,
    frequency_array: np.ndarray,
    psd_array: np.ndarray,
    show_transponder_lines: bool = True,
    title: str = "Spectrum Waterfall"
) -> go.Figure:
    """
    Create an interactive waterfall plot.

    Args:
        time_array: Time points in minutes (1D array)
        frequency_array: Frequency points in Hz (1D array)
        psd_array: PSD values in dBm/Hz (2D array: time x frequency)
        show_transponder_lines: Whether to show transponder boundaries
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Convert to GHz for display
    freq_ghz = frequency_array / 1e9
    time_hours = time_array / 60

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        x=freq_ghz,
        y=time_hours,
        z=psd_array,
        colorscale='Viridis',
        colorbar=dict(title=dict(text="PSD<br>(dBm/Hz)", side='right')),
        hovertemplate='Freq: %{x:.3f} GHz<br>Time: %{y:.2f} hrs<br>PSD: %{z:.1f} dBm/Hz<extra></extra>'
    ))

    # Add transponder boundaries based on actual data range
    if show_transponder_lines:
        # Calculate transponder boundaries from actual frequency range
        freq_start = frequency_array[0]
        freq_end = frequency_array[-1]
        freq_span = freq_end - freq_start
        transponder_bw = freq_span / NUM_TRANSPONDERS

        for i in range(NUM_TRANSPONDERS + 1):
            edge_freq = (freq_start + i * transponder_bw) / 1e9
            fig.add_vline(
                x=edge_freq,
                line=dict(color='white', width=1, dash='dash'),
                opacity=0.5
            )

        # Add transponder labels
        for i in range(NUM_TRANSPONDERS):
            center_freq = (freq_start + (i + 0.5) * transponder_bw) / 1e9
            fig.add_annotation(
                x=center_freq,
                y=time_hours[-1],
                text=f"T{i}",
                showarrow=False,
                yshift=10,
                font=dict(color='white', size=10),
                bgcolor='rgba(0,0,0,0.5)'
            )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
        xaxis=dict(title=dict(text='Frequency (GHz)', font=dict(size=14))),
        yaxis=dict(title=dict(text='Time (hours)', font=dict(size=14))),
        height=600,
        margin=dict(t=80, b=60, l=60, r=100)
    )

    return fig


def plot_cluster_traces(
    frequencies: np.ndarray,
    min_trace: np.ndarray,
    max_trace: np.ndarray,
    avg_trace: np.ndarray,
    cluster_label: int,
    color: Optional[str] = None
) -> go.Figure:
    """
    Plot min, max, and average power spectrum traces for a cluster.

    Args:
        frequencies: Frequency array in Hz
        min_trace: Minimum power spectrum
        max_trace: Maximum power spectrum
        avg_trace: Average power spectrum
        cluster_label: Cluster ID for title
        color: Color to use for traces (optional)

    Returns:
        Plotly Figure object
    """
    if len(frequencies) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for this cluster",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    freq_ghz = frequencies / 1e9

    # Choose color
    if color is None:
        colors = get_color_palette(1)
        color = colors[0]

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=freq_ghz,
        y=min_trace,
        mode='lines',
        name='Minimum',
        line=dict(color=color, width=2, dash='dash'),
        opacity=0.6,
        hovertemplate='Freq: %{x:.3f} GHz<br>Min PSD: %{y:.1f} dBm/Hz<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=freq_ghz,
        y=max_trace,
        mode='lines',
        name='Maximum',
        line=dict(color=color, width=2, dash='dot'),
        opacity=0.6,
        hovertemplate='Freq: %{x:.3f} GHz<br>Max PSD: %{y:.1f} dBm/Hz<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=freq_ghz,
        y=avg_trace,
        mode='lines',
        name='Average',
        line=dict(color=color, width=3),
        hovertemplate='Freq: %{x:.3f} GHz<br>Avg PSD: %{y:.1f} dBm/Hz<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=f"Cluster {cluster_label} - Spectrum Traces",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(title=dict(text='Frequency (GHz)', font=dict(size=14))),
        yaxis=dict(title=dict(text='PSD (dBm/Hz)', font=dict(size=14))),
        height=400,
        margin=dict(t=80, b=60, l=60, r=40),
        showlegend=True,
        legend=dict(x=1, y=1, xanchor='right', yanchor='top'),
        hovermode='x unified'
    )

    return fig


def plot_cluster_timeline(
    df: pd.DataFrame,
    cluster_label: int,
    color: Optional[str] = None
) -> go.Figure:
    """
    Plot binary timeline showing when a cluster is active.

    Args:
        df: DataFrame with 'cluster' column
        cluster_label: Cluster ID to plot
        color: Color to use for markers (optional)

    Returns:
        Plotly Figure object
    """
    if 'cluster' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Get time array
    time_array = np.array([
        (t - df.index[0]).total_seconds() / 60
        for t in df.index
    ])
    time_hours = time_array / 60

    # Get cluster activity
    is_active = (df['cluster'] == cluster_label).values

    # Choose color
    if color is None:
        colors = get_color_palette(1)
        color = colors[0]

    fig = go.Figure()

    # Plot as scatter (dots when active)
    active_times = time_hours[is_active]
    active_values = np.ones_like(active_times)

    fig.add_trace(go.Scatter(
        x=active_times,
        y=active_values,
        mode='markers',
        name=f'Cluster {cluster_label} Active',
        marker=dict(
            color=color,
            size=8,
            symbol='circle'
        ),
        hovertemplate='Time: %{x:.2f} hrs<br>Status: Active<extra></extra>'
    ))

    # Add reference line at y=1
    fig.add_hline(
        y=1,
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.3
    )

    fig.update_layout(
        title=dict(
            text=f"Cluster {cluster_label} - Activity Timeline",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(text='Time (hours)', font=dict(size=14)),
            range=[0, max(time_hours) * 1.05] if len(time_hours) > 0 else [0, 1]
        ),
        yaxis=dict(
            title=dict(text='Activity', font=dict(size=14)),
            range=[0.5, 1.5],
            tickvals=[1],
            ticktext=['Active'],
            showgrid=False
        ),
        height=300,
        margin=dict(t=80, b=60, l=60, r=40),
        showlegend=False
    )

    return fig


def plot_all_clusters_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Plot timeline showing all cluster activities.

    Args:
        df: DataFrame with 'cluster' column

    Returns:
        Plotly Figure object
    """
    if 'cluster' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Get time array
    time_array = np.array([
        (t - df.index[0]).total_seconds() / 60
        for t in df.index
    ])
    time_hours = time_array / 60

    # Get unique clusters
    unique_clusters = sorted(df['cluster'].unique())
    colors = get_color_palette(len(unique_clusters))

    fig = go.Figure()

    # Plot each cluster
    for idx, cluster_label in enumerate(unique_clusters):
        is_active = (df['cluster'] == cluster_label).values
        active_times = time_hours[is_active]
        active_values = np.ones_like(active_times) * idx

        fig.add_trace(go.Scatter(
            x=active_times,
            y=active_values,
            mode='markers',
            name=f'Cluster {cluster_label}',
            marker=dict(
                color=colors[idx],
                size=6,
                symbol='circle'
            ),
            hovertemplate=f'Cluster {cluster_label}<br>Time: %{{x:.2f}} hrs<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text="All Clusters - Activity Timeline",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(text='Time (hours)', font=dict(size=14)),
            range=[0, max(time_hours) * 1.05] if len(time_hours) > 0 else [0, 1]
        ),
        yaxis=dict(
            title=dict(text='Cluster', font=dict(size=14)),
            tickvals=list(range(len(unique_clusters))),
            ticktext=[f'C{label}' for label in unique_clusters],
        ),
        height=400,
        margin=dict(t=80, b=60, l=60, r=40),
        showlegend=True,
        legend=dict(x=1, y=1, xanchor='left', yanchor='top')
    )

    return fig
