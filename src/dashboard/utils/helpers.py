"""Utility functions and constants for the RF-ML dashboard."""

import numpy as np
from datetime import datetime
from typing import Union

# Ku-band downlink frequency range (GHz)
KU_BAND_START = 10.7e9  # Hz
KU_BAND_END = 12.75e9   # Hz
KU_BAND_BW = KU_BAND_END - KU_BAND_START

# Transponder configuration
NUM_TRANSPONDERS = 6
TRANSPONDER_BW = KU_BAND_BW / NUM_TRANSPONDERS  # ~341.67 MHz each


def get_transponder_index(freq_hz: float) -> int:
    """
    Calculate the transponder index for a given frequency.

    Args:
        freq_hz: Frequency in Hz

    Returns:
        Transponder index (0-5), or -1 if out of range
    """
    if freq_hz < KU_BAND_START or freq_hz > KU_BAND_END:
        return -1

    offset = freq_hz - KU_BAND_START
    index = int(offset / TRANSPONDER_BW)

    # Clamp to valid range
    return min(max(index, 0), NUM_TRANSPONDERS - 1)


def get_transponder_range(transponder_idx: int) -> tuple:
    """
    Get the frequency range for a transponder.

    Args:
        transponder_idx: Transponder index (0-5)

    Returns:
        Tuple of (start_freq_hz, end_freq_hz)
    """
    start_freq = KU_BAND_START + (transponder_idx * TRANSPONDER_BW)
    end_freq = start_freq + TRANSPONDER_BW
    return (start_freq, end_freq)


def format_frequency(freq_hz: float, precision: int = 2) -> str:
    """
    Format frequency in human-readable form.

    Args:
        freq_hz: Frequency in Hz
        precision: Number of decimal places

    Returns:
        Formatted string (e.g., "12.25 GHz")
    """
    if freq_hz >= 1e9:
        return f"{freq_hz/1e9:.{precision}f} GHz"
    elif freq_hz >= 1e6:
        return f"{freq_hz/1e6:.{precision}f} MHz"
    elif freq_hz >= 1e3:
        return f"{freq_hz/1e3:.{precision}f} kHz"
    else:
        return f"{freq_hz:.{precision}f} Hz"


def format_time(timestamp: Union[datetime, str, float]) -> str:
    """
    Format timestamp in human-readable form.

    Args:
        timestamp: Datetime object, ISO string, or Unix timestamp

    Returns:
        Formatted string (e.g., "2025-01-15 14:30:00")
    """
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            return timestamp
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = timestamp

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def get_color_palette(n_colors: int) -> list:
    """
    Generate a color palette for cluster visualization.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of hex color strings
    """
    # Use a perceptually distinct color palette
    base_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]

    if n_colors <= len(base_colors):
        return base_colors[:n_colors]

    # If we need more colors, generate them using HSV
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
        colors.append(hex_color)

    return colors
