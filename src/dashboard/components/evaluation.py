"""Evaluation metrics and utilities for clustering results."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples
)
import time


class ClusteringEvaluator:
    """Evaluate clustering quality using multiple metrics."""

    def __init__(self):
        """Initialize clustering evaluator."""
        self.metrics = {}
        self.computation_time = None

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metric_list: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluate clustering using specified metrics.

        Args:
            features: Feature array (n_samples, n_features)
            labels: Cluster labels
            metric_list: List of metrics to compute
                        ('silhouette', 'davies_bouldin', 'calinski_harabasz', 'inertia')
                        If None, computes all available metrics

        Returns:
            Dict of metric name to value
        """
        start_time = time.time()

        if metric_list is None:
            metric_list = ['silhouette', 'davies_bouldin', 'calinski_harabasz']

        results = {}

        # Filter out noise points (label -1 from DBSCAN)
        mask = labels >= 0
        n_valid = np.sum(mask)
        n_outliers = np.sum(~mask)

        if n_valid < 2:
            results['error'] = 'insufficient_samples'
            results['n_outliers'] = n_outliers
            return results

        features_filtered = features[mask]
        labels_filtered = labels[mask]

        n_clusters = len(np.unique(labels_filtered))

        # Add basic statistics
        results['n_clusters'] = n_clusters
        results['n_samples'] = len(labels)
        results['n_outliers'] = n_outliers
        results['outlier_percentage'] = 100 * n_outliers / len(labels)

        if n_clusters < 2:
            results['error'] = 'single_cluster'
            return results

        # Compute requested metrics
        if 'silhouette' in metric_list:
            try:
                results['silhouette_score'] = silhouette_score(
                    features_filtered,
                    labels_filtered
                )
            except Exception as e:
                results['silhouette_score'] = float('nan')
                results['silhouette_error'] = str(e)

        if 'davies_bouldin' in metric_list:
            try:
                results['davies_bouldin_index'] = davies_bouldin_score(
                    features_filtered,
                    labels_filtered
                )
            except Exception as e:
                results['davies_bouldin_index'] = float('nan')
                results['davies_bouldin_error'] = str(e)

        if 'calinski_harabasz' in metric_list:
            try:
                results['calinski_harabasz_score'] = calinski_harabasz_score(
                    features_filtered,
                    labels_filtered
                )
            except Exception as e:
                results['calinski_harabasz_score'] = float('nan')
                results['calinski_harabasz_error'] = str(e)

        if 'inertia' in metric_list:
            # Compute within-cluster sum of squares
            try:
                inertia = self._compute_inertia(features_filtered, labels_filtered)
                results['inertia'] = inertia
            except Exception as e:
                results['inertia'] = float('nan')
                results['inertia_error'] = str(e)

        self.computation_time = time.time() - start_time
        results['computation_time'] = self.computation_time

        self.metrics = results
        return results

    def _compute_inertia(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute within-cluster sum of squares (inertia).

        Args:
            features: Feature array
            labels: Cluster labels

        Returns:
            Inertia value
        """
        inertia = 0.0
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            cluster_points = features[mask]

            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                squared_distances = np.sum((cluster_points - center) ** 2, axis=1)
                inertia += np.sum(squared_distances)

        return inertia

    def get_silhouette_per_cluster(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, float]:
        """
        Get average silhouette score for each cluster.

        Args:
            features: Feature array
            labels: Cluster labels

        Returns:
            Dict mapping cluster label to average silhouette score
        """
        # Filter out noise points
        mask = labels >= 0
        if np.sum(mask) < 2:
            return {}

        features_filtered = features[mask]
        labels_filtered = labels[mask]

        n_clusters = len(np.unique(labels_filtered))
        if n_clusters < 2:
            return {}

        try:
            # Compute silhouette score for each sample
            silhouette_vals = silhouette_samples(features_filtered, labels_filtered)

            # Average by cluster
            results = {}
            for label in np.unique(labels_filtered):
                mask_cluster = labels_filtered == label
                results[int(label)] = float(np.mean(silhouette_vals[mask_cluster]))

            return results

        except Exception:
            return {}

    def interpret_metrics(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Provide human-readable interpretations of metric values.

        Args:
            metrics: Dict of metric values from evaluate()

        Returns:
            Dict of metric name to interpretation string
        """
        interpretations = {}

        if 'silhouette_score' in metrics:
            score = metrics['silhouette_score']
            if np.isnan(score):
                interpretations['silhouette'] = "Could not compute"
            elif score > 0.7:
                interpretations['silhouette'] = "Excellent separation"
            elif score > 0.5:
                interpretations['silhouette'] = "Good separation"
            elif score > 0.3:
                interpretations['silhouette'] = "Moderate separation"
            elif score > 0.0:
                interpretations['silhouette'] = "Weak separation"
            else:
                interpretations['silhouette'] = "Poor separation (overlapping clusters)"

        if 'davies_bouldin_index' in metrics:
            score = metrics['davies_bouldin_index']
            if np.isnan(score):
                interpretations['davies_bouldin'] = "Could not compute"
            elif score < 0.5:
                interpretations['davies_bouldin'] = "Excellent clustering"
            elif score < 1.0:
                interpretations['davies_bouldin'] = "Good clustering"
            elif score < 1.5:
                interpretations['davies_bouldin'] = "Moderate clustering"
            else:
                interpretations['davies_bouldin'] = "Poor clustering (clusters overlap)"

        if 'calinski_harabasz_score' in metrics:
            score = metrics['calinski_harabasz_score']
            if np.isnan(score):
                interpretations['calinski_harabasz'] = "Could not compute"
            elif score > 1000:
                interpretations['calinski_harabasz'] = "Excellent clustering"
            elif score > 500:
                interpretations['calinski_harabasz'] = "Good clustering"
            elif score > 100:
                interpretations['calinski_harabasz'] = "Moderate clustering"
            else:
                interpretations['calinski_harabasz'] = "Weak clustering"

        return interpretations


def compare_clustering_results(
    results_list: list[Dict[str, Any]],
    names: list[str]
) -> pd.DataFrame:
    """
    Compare multiple clustering results side-by-side.

    Args:
        results_list: List of evaluation result dictionaries
        names: List of names for each result (e.g., method names)

    Returns:
        DataFrame with comparison table
    """
    comparison_data = []

    for name, results in zip(names, results_list):
        row = {'Method': name}

        # Add all numeric metrics
        for key, value in results.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Set Method as index
    if 'Method' in df.columns:
        df = df.set_index('Method')

    return df


def rank_clustering_results(
    comparison_df: pd.DataFrame,
    preferences: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Rank clustering results based on metrics.

    Args:
        comparison_df: DataFrame from compare_clustering_results()
        preferences: Dict mapping metric name to preference ('higher' or 'lower')
                    If None, uses default preferences:
                    - silhouette_score: higher is better
                    - davies_bouldin_index: lower is better
                    - calinski_harabasz_score: higher is better
                    - inertia: lower is better

    Returns:
        DataFrame with added rank columns
    """
    if preferences is None:
        preferences = {
            'silhouette_score': 'higher',
            'davies_bouldin_index': 'lower',
            'calinski_harabasz_score': 'higher',
            'inertia': 'lower'
        }

    df_ranked = comparison_df.copy()

    # Add rank columns
    for metric, preference in preferences.items():
        if metric in df_ranked.columns:
            ascending = (preference == 'lower')
            rank_col = f'{metric}_rank'
            df_ranked[rank_col] = df_ranked[metric].rank(ascending=ascending)

    # Compute overall rank (average of individual ranks)
    rank_cols = [col for col in df_ranked.columns if col.endswith('_rank')]
    if rank_cols:
        df_ranked['overall_rank'] = df_ranked[rank_cols].mean(axis=1)
        df_ranked = df_ranked.sort_values('overall_rank')

    return df_ranked


def get_cluster_distribution(labels: np.ndarray) -> Dict[int, int]:
    """
    Get distribution of samples across clusters.

    Args:
        labels: Cluster labels

    Returns:
        Dict mapping cluster label to count
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def compute_cluster_sizes_stats(labels: np.ndarray) -> Dict[str, Any]:
    """
    Get statistics about cluster sizes.

    Args:
        labels: Cluster labels

    Returns:
        Dict with cluster size statistics
    """
    distribution = get_cluster_distribution(labels)

    # Remove outliers (-1) if present
    sizes = [count for label, count in distribution.items() if label >= 0]

    if not sizes:
        return {'error': 'no_valid_clusters'}

    return {
        'n_clusters': len(sizes),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'mean_size': np.mean(sizes),
        'median_size': np.median(sizes),
        'std_size': np.std(sizes),
        'size_imbalance': max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')
    }
