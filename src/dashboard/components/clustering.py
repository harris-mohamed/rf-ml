"""BIRCH clustering for RF spectrum data."""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy.signal import find_peaks
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract features from spectrum data for clustering.

    Features extracted (8 total):
        1. mean_power: Average power across spectrum
        2. max_power: Maximum power value
        3. std_power: Standard deviation of power
        4. median_power: Median power value
        5. num_peaks: Number of significant peaks detected
        6. spectral_centroid: Frequency-weighted power centroid
        7. power_above_threshold: Number of bins above (mean + std)
        8. low_freq_ratio: Fraction of energy in low vs high frequencies

    Args:
        df: DataFrame from data_loader.load_spectrum_data()

    Returns:
        Feature array of shape (num_samples, 8)
    """
    features_list = []

    for idx, row in df.iterrows():
        frequencies = row['frequencies']
        powers = row['powers']

        # Basic statistics
        mean_power = np.mean(powers)
        max_power = np.max(powers)
        std_power = np.std(powers)
        median_power = np.median(powers)

        # Peak detection
        peaks, _ = find_peaks(powers, prominence=5)
        num_peaks = len(peaks)

        # Spectral centroid (frequency-weighted average)
        total_power = np.sum(powers)
        if total_power > 0:
            spectral_centroid = np.sum(frequencies * powers) / total_power
        else:
            spectral_centroid = np.mean(frequencies)

        # Power above threshold (occupied bandwidth indicator)
        threshold = mean_power + std_power
        power_above_threshold = np.sum(powers > threshold)

        # Energy concentration (low vs high frequency)
        mid_freq = (frequencies[0] + frequencies[-1]) / 2
        low_freq_mask = frequencies < mid_freq
        high_freq_mask = frequencies >= mid_freq

        low_freq_energy = np.sum(powers[low_freq_mask])
        high_freq_energy = np.sum(powers[high_freq_mask])
        total_energy = low_freq_energy + high_freq_energy

        if total_energy > 0:
            low_freq_ratio = low_freq_energy / total_energy
        else:
            low_freq_ratio = 0.5

        features_list.append([
            mean_power,
            max_power,
            std_power,
            median_power,
            num_peaks,
            spectral_centroid,
            power_above_threshold,
            low_freq_ratio
        ])

    return np.array(features_list)


def perform_clustering(
    features: np.ndarray,
    threshold: float = 0.5,
    branching_factor: int = 50
) -> Tuple[np.ndarray, Birch, StandardScaler]:
    """
    Perform BIRCH clustering on extracted features.

    Uses pure BIRCH clustering where the number of clusters is determined
    automatically based on the threshold parameter.

    Args:
        features: Feature array from extract_features()
        threshold: Radius of subcluster obtained by merging (main parameter controlling cluster count)
        branching_factor: Maximum number of CF subclusters in each node

    Returns:
        Tuple of (cluster_labels, birch_model, scaler)
        - cluster_labels: Cluster assignment for each sample (1D array)
        - birch_model: Fitted BIRCH model
        - scaler: Fitted StandardScaler for normalization
    """
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Apply BIRCH clustering (pure BIRCH without fixed n_clusters)
    birch = Birch(
        n_clusters=None,
        threshold=threshold,
        branching_factor=branching_factor
    )
    cluster_labels = birch.fit_predict(features_normalized)

    return cluster_labels, birch, scaler


def add_cluster_labels(
    df: pd.DataFrame,
    cluster_labels: np.ndarray
) -> pd.DataFrame:
    """
    Add cluster labels to the dataframe.

    Args:
        df: DataFrame from data_loader.load_spectrum_data()
        cluster_labels: Cluster labels from perform_clustering()

    Returns:
        DataFrame with added 'cluster' column
    """
    df_copy = df.copy()
    df_copy['cluster'] = cluster_labels
    return df_copy


def get_cluster_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about cluster distribution.

    Args:
        df: DataFrame with 'cluster' column

    Returns:
        Dictionary with cluster statistics
    """
    if 'cluster' not in df.columns:
        return {}

    cluster_counts = df['cluster'].value_counts().sort_index()
    total_samples = len(df)

    stats = {
        'num_clusters': len(cluster_counts),
        'total_samples': total_samples,
        'cluster_counts': cluster_counts.to_dict(),
        'cluster_percentages': {
            label: 100 * count / total_samples
            for label, count in cluster_counts.items()
        }
    }

    return stats


def get_cluster_traces(
    df: pd.DataFrame,
    cluster_label: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get min, max, and average spectrum traces for a specific cluster.

    Args:
        df: DataFrame with 'cluster' column
        cluster_label: Cluster ID to analyze

    Returns:
        Tuple of (frequencies, min_trace, max_trace, avg_trace)
        - frequencies: Frequency array (Hz)
        - min_trace: Minimum power spectrum for this cluster
        - max_trace: Maximum power spectrum for this cluster
        - avg_trace: Average power spectrum for this cluster
    """
    # Filter to cluster
    cluster_mask = df['cluster'] == cluster_label
    cluster_data = df[cluster_mask]

    if len(cluster_data) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Get frequency array (should be same for all rows)
    frequencies = cluster_data.iloc[0]['frequencies']
    num_freq_bins = len(frequencies)

    # Collect all power spectra for this cluster
    all_spectra = np.zeros((len(cluster_data), num_freq_bins))
    for i, (idx, row) in enumerate(cluster_data.iterrows()):
        all_spectra[i, :] = row['powers']

    # Compute min, max, avg across time dimension
    min_trace = np.min(all_spectra, axis=0)
    max_trace = np.max(all_spectra, axis=0)
    avg_trace = np.mean(all_spectra, axis=0)

    return frequencies, min_trace, max_trace, avg_trace


def get_cluster_timeline(df: pd.DataFrame, cluster_label: int) -> np.ndarray:
    """
    Get binary timeline showing when a cluster is active.

    Args:
        df: DataFrame with 'cluster' column
        cluster_label: Cluster ID to analyze

    Returns:
        Boolean array indicating cluster presence at each time point
    """
    if 'cluster' not in df.columns:
        return np.array([])

    return (df['cluster'] == cluster_label).values


FEATURE_NAMES = [
    'Mean Power',
    'Max Power',
    'Std Power',
    'Median Power',
    'Num Peaks',
    'Spectral Centroid',
    'Power Above Threshold',
    'Low Freq Ratio'
]
