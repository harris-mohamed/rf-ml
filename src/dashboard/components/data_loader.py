"""Data loading utilities for the RF-ML dashboard."""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

from satellite_downlink_simulator.simulation import SpectrumRecord


def list_available_json_files(data_dir: str = "dev_output") -> List[str]:
    """
    List all JSON files in the data directory.

    Args:
        data_dir: Directory to search for JSON files

    Returns:
        List of JSON file paths (relative to project root)
    """
    # Get project root (assuming we're in src/dashboard/components/)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    data_path = project_root / data_dir

    if not data_path.exists():
        return []

    # Find all JSON files
    json_files = list(data_path.glob("*.json"))

    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Return relative paths as strings
    return [str(f.relative_to(project_root)) for f in json_files]


def load_spectrum_data(json_filepath: str) -> pd.DataFrame:
    """
    Load spectrum data from a JSON file and convert to DataFrame.

    Args:
        json_filepath: Path to JSON file (relative to project root or absolute)

    Returns:
        DataFrame with columns:
            - timestamp (index)
            - cf_hz: Center frequency in Hz
            - bw_hz: Bandwidth in Hz
            - rbw_hz: Resolution bandwidth in Hz
            - vbw_hz: Video bandwidth in Hz
            - frequencies: Array of frequency points (Hz)
            - powers: Array of power values (dBm/Hz)
    """
    # Convert to absolute path if relative
    if not os.path.isabs(json_filepath):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        json_filepath = str(project_root / json_filepath)

    # Load records from JSON
    records = SpectrumRecord.from_file(json_filepath)

    # Convert to dataframe
    data_rows = []
    for record in records:
        freq_start = record.cf_hz - (record.bw_hz / 2)
        freq_end = record.cf_hz + (record.bw_hz / 2)
        frequencies = np.linspace(freq_start, freq_end, record.psd_shape[0])
        powers = record.get_psd()

        data_rows.append({
            'timestamp': record.timestamp,
            'cf_hz': record.cf_hz,
            'bw_hz': record.bw_hz,
            'rbw_hz': record.rbw_hz,
            'vbw_hz': record.vbw_hz,
            'frequencies': frequencies,
            'powers': powers
        })

    df = pd.DataFrame(data_rows)
    df = df.set_index('timestamp')

    return df


def get_file_info(json_filepath: str) -> dict:
    """
    Get metadata about a JSON file without fully loading it.

    Args:
        json_filepath: Path to JSON file

    Returns:
        Dictionary with file metadata
    """
    # Convert to absolute path if relative
    if not os.path.isabs(json_filepath):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        json_filepath = str(project_root / json_filepath)

    path = Path(json_filepath)

    if not path.exists():
        return {
            'filename': path.name,
            'exists': False,
            'size_mb': 0,
            'modified': None,
            'num_records': 0
        }

    # Get file size
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Get modification time
    modified = datetime.fromtimestamp(path.stat().st_mtime)

    # Try to get number of records (without full load)
    try:
        records = SpectrumRecord.from_file(str(path))
        num_records = len(records)
    except:
        num_records = -1

    return {
        'filename': path.name,
        'exists': True,
        'size_mb': size_mb,
        'modified': modified,
        'num_records': num_records
    }


def build_waterfall_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build time, frequency, and PSD arrays for waterfall plotting.

    Args:
        df: DataFrame from load_spectrum_data()

    Returns:
        Tuple of (time_array, frequency_array, psd_array)
        - time_array: Time points in minutes from start (1D array)
        - frequency_array: Frequency points in Hz (1D array)
        - psd_array: PSD values in dBm/Hz (2D array: time x frequency)
    """
    num_snapshots = len(df)

    # Get frequency array from first row (should be same for all)
    frequency_array = df.iloc[0]['frequencies']
    num_freq_bins = len(frequency_array)

    # Build time array (minutes from start)
    time_array = np.array([
        (t - df.index[0]).total_seconds() / 60
        for t in df.index
    ])

    # Build PSD array (time x frequency)
    psd_array = np.zeros((num_snapshots, num_freq_bins))
    for i, (timestamp, row) in enumerate(df.iterrows()):
        psd_array[i, :] = row['powers']

    return time_array, frequency_array, psd_array


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for a loaded dataset.

    Args:
        df: DataFrame from load_spectrum_data()

    Returns:
        Dictionary with summary statistics
    """
    time_array, frequency_array, psd_array = build_waterfall_arrays(df)

    return {
        'num_snapshots': len(df),
        'num_freq_bins': len(frequency_array),
        'time_start': df.index[0],
        'time_end': df.index[-1],
        'duration_minutes': time_array[-1] - time_array[0],
        'freq_start_hz': frequency_array[0],
        'freq_end_hz': frequency_array[-1],
        'freq_span_mhz': (frequency_array[-1] - frequency_array[0]) / 1e6,
        'psd_min_dbm': np.min(psd_array),
        'psd_max_dbm': np.max(psd_array),
        'psd_mean_dbm': np.mean(psd_array),
        'rbw_khz': df.iloc[0]['rbw_hz'] / 1e3,
        'vbw_khz': df.iloc[0]['vbw_hz'] / 1e3,
    }
