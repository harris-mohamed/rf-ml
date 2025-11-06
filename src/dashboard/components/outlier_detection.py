"""Outlier detection methods for RF spectrum data."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class OutlierDetector(ABC):
    """Base class for outlier detection methods."""

    @abstractmethod
    def fit_predict(
        self,
        data: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Detect outliers in data.

        Args:
            data: Input data (features or time series)
            **kwargs: Method-specific parameters

        Returns:
            Tuple of (outlier_labels, outlier_scores, fitted_model)
            - outlier_labels: 1 for inliers, -1 for outliers
            - outlier_scores: Outlier scores (higher = more outlier-like)
            - fitted_model: Fitted outlier detection model
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of outlier detection method."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get description of outlier detection method."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this method."""
        pass

    @abstractmethod
    def supports_large_datasets(self) -> bool:
        """Whether this method is suitable for large datasets (>10k samples)."""
        pass


class IsolationForestDetector(OutlierDetector):
    """Isolation Forest outlier detection (fast, scalable)."""

    def fit_predict(
        self,
        data: np.ndarray,
        contamination: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, IsolationForest]:
        """
        Detect outliers using Isolation Forest.

        Args:
            data: Feature array (n_samples, n_features)
            contamination: Expected proportion of outliers (0.0 to 0.5)
            random_state: Random seed
            **kwargs: Additional parameters for IsolationForest

        Returns:
            Tuple of (outlier_labels, outlier_scores, model)
        """
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            **kwargs
        )

        # Fit and predict
        outlier_labels = model.fit_predict(data)

        # Get anomaly scores (more negative = more outlier-like)
        # Convert to positive scores (higher = more outlier-like)
        outlier_scores = -model.score_samples(data)

        return outlier_labels, outlier_scores, model

    def get_name(self) -> str:
        return "Isolation Forest"

    def get_description(self) -> str:
        return "Fast tree-based outlier detection (recommended for large datasets)"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'contamination': 0.1,
            'random_state': 42
        }

    def supports_large_datasets(self) -> bool:
        return True  # O(n log n) complexity


class LOFDetector(OutlierDetector):
    """Local Outlier Factor detection (density-based)."""

    def fit_predict(
        self,
        data: np.ndarray,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, LocalOutlierFactor]:
        """
        Detect outliers using Local Outlier Factor.

        Args:
            data: Feature array (n_samples, n_features)
            n_neighbors: Number of neighbors to use
            contamination: Expected proportion of outliers
            **kwargs: Additional parameters for LOF

        Returns:
            Tuple of (outlier_labels, outlier_scores, model)
        """
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            **kwargs
        )

        # Fit and predict
        outlier_labels = model.fit_predict(data)

        # Get negative outlier factor scores
        # Convert to positive scores (higher = more outlier-like)
        outlier_scores = -model.negative_outlier_factor_

        return outlier_labels, outlier_scores, model

    def get_name(self) -> str:
        return "Local Outlier Factor"

    def get_description(self) -> str:
        return "Density-based outlier detection (good for clustered data)"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_neighbors': 20,
            'contamination': 0.1
        }

    def supports_large_datasets(self) -> bool:
        return False  # O(n²) complexity


class StatisticalOutlierDetector(OutlierDetector):
    """Statistical outlier detection using z-score or IQR."""

    def fit_predict(
        self,
        data: np.ndarray,
        method: str = 'zscore',
        threshold: float = 3.0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Detect outliers using statistical methods.

        Args:
            data: Feature array (n_samples, n_features)
            method: 'zscore' or 'iqr'
            threshold: Threshold for outlier detection
                       For zscore: points with |z| > threshold are outliers
                       For IQR: points > Q3 + threshold*IQR or < Q1 - threshold*IQR
            **kwargs: Additional parameters (ignored)

        Returns:
            Tuple of (outlier_labels, outlier_scores, stats_dict)
        """
        n_samples, n_features = data.shape
        outlier_scores = np.zeros(n_samples)

        if method == 'zscore':
            # Z-score method
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            # Avoid division by zero
            std[std == 0] = 1.0

            # Compute z-scores for each feature
            z_scores = np.abs((data - mean) / std)

            # Max z-score across features as outlier score
            outlier_scores = np.max(z_scores, axis=1)

            # Label outliers
            outlier_labels = np.where(outlier_scores > threshold, -1, 1)

            stats = {
                'method': 'zscore',
                'threshold': threshold,
                'mean': mean,
                'std': std
            }

        elif method == 'iqr':
            # IQR method
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            iqr = q75 - q25

            # Avoid division by zero
            iqr[iqr == 0] = 1.0

            # Compute distance from IQR bounds
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr

            # Distance beyond bounds for each feature
            lower_dist = np.maximum(0, lower_bound - data)
            upper_dist = np.maximum(0, data - upper_bound)

            # Max distance across features as outlier score
            outlier_scores = np.max(lower_dist + upper_dist, axis=1)

            # Label outliers
            outlier_labels = np.where(outlier_scores > 0, -1, 1)

            stats = {
                'method': 'iqr',
                'threshold': threshold,
                'q25': q25,
                'q75': q75,
                'iqr': iqr
            }

        else:
            raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'")

        return outlier_labels, outlier_scores, stats

    def get_name(self) -> str:
        return "Statistical"

    def get_description(self) -> str:
        return "Simple statistical outlier detection (z-score or IQR method)"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'method': 'zscore',
            'threshold': 3.0
        }

    def supports_large_datasets(self) -> bool:
        return True  # O(n) complexity


class ClusterBasedOutlierDetector(OutlierDetector):
    """Cluster-based outlier detection using distance from cluster centers."""

    def fit_predict(
        self,
        data: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        threshold: float = 2.0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Detect outliers based on distance from cluster centers.

        Args:
            data: Feature array (n_samples, n_features)
            cluster_labels: Cluster assignments from clustering
            cluster_centers: Cluster center coordinates
            threshold: Threshold in standard deviations from cluster center
            **kwargs: Additional parameters (ignored)

        Returns:
            Tuple of (outlier_labels, outlier_scores, stats_dict)
        """
        n_samples = len(data)
        outlier_scores = np.zeros(n_samples)

        # Compute distance from each point to its cluster center
        for label in np.unique(cluster_labels):
            if label == -1:
                # Skip noise points from DBSCAN
                continue

            mask = cluster_labels == label
            center = cluster_centers[label]

            # Compute Euclidean distance
            distances = np.linalg.norm(data[mask] - center, axis=1)

            # Store as outlier scores
            outlier_scores[mask] = distances

        # Compute threshold based on mean and std of distances
        mean_dist = np.mean(outlier_scores)
        std_dist = np.std(outlier_scores)

        threshold_value = mean_dist + threshold * std_dist

        # Label outliers
        outlier_labels = np.where(outlier_scores > threshold_value, -1, 1)

        stats = {
            'method': 'cluster_based',
            'threshold': threshold,
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'threshold_value': threshold_value
        }

        return outlier_labels, outlier_scores, stats

    def get_name(self) -> str:
        return "Cluster-Based"

    def get_description(self) -> str:
        return "Detect outliers based on distance from cluster centers"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'threshold': 2.0
        }

    def supports_large_datasets(self) -> bool:
        return True  # O(n) complexity


# Factory function
def get_outlier_detector(method: str) -> OutlierDetector:
    """
    Get outlier detector by name.

    Args:
        method: Name of outlier detection method
                ('isolation_forest', 'lof', 'statistical', 'cluster_based')

    Returns:
        OutlierDetector instance
    """
    methods = {
        'isolation_forest': IsolationForestDetector,
        'lof': LOFDetector,
        'statistical': StatisticalOutlierDetector,
        'cluster_based': ClusterBasedOutlierDetector
    }

    method_lower = method.lower()
    if method_lower not in methods:
        raise ValueError(
            f"Unknown outlier detection method: {method}. "
            f"Available methods: {list(methods.keys())}"
        )

    return methods[method_lower]()


def get_available_detectors() -> Dict[str, str]:
    """
    Get dict of available outlier detection methods.

    Returns:
        Dict mapping method name to description
    """
    methods = ['isolation_forest', 'lof', 'statistical', 'cluster_based']
    return {
        method: get_outlier_detector(method).get_description()
        for method in methods
    }


def get_outlier_statistics(
    outlier_labels: np.ndarray,
    outlier_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Get statistics about detected outliers.

    Args:
        outlier_labels: 1 for inliers, -1 for outliers
        outlier_scores: Outlier scores

    Returns:
        Dict with outlier statistics
    """
    n_total = len(outlier_labels)
    n_outliers = np.sum(outlier_labels == -1)
    n_inliers = np.sum(outlier_labels == 1)

    outlier_mask = outlier_labels == -1
    inlier_mask = outlier_labels == 1

    stats = {
        'n_total': n_total,
        'n_outliers': n_outliers,
        'n_inliers': n_inliers,
        'outlier_percentage': 100 * n_outliers / n_total if n_total > 0 else 0,
        'mean_outlier_score': np.mean(outlier_scores[outlier_mask]) if n_outliers > 0 else 0,
        'mean_inlier_score': np.mean(outlier_scores[inlier_mask]) if n_inliers > 0 else 0,
        'max_outlier_score': np.max(outlier_scores) if n_total > 0 else 0,
        'min_outlier_score': np.min(outlier_scores) if n_total > 0 else 0,
    }

    return stats
