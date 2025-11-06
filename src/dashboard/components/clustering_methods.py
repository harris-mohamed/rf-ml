"""Clustering methods for RF spectrum data."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings


class ClusteringMethod(ABC):
    """Base class for clustering methods."""

    @abstractmethod
    def fit_predict(
        self,
        data: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, Any]:
        """
        Fit clustering model and predict cluster labels.

        Args:
            data: Input data (features or time series)
            **kwargs: Algorithm-specific parameters

        Returns:
            Tuple of (cluster_labels, fitted_model)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of clustering method."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get description of clustering method."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this method."""
        pass

    @abstractmethod
    def supports_large_datasets(self) -> bool:
        """Whether this method is suitable for large datasets (>10k samples)."""
        pass


class BIRCHClustering(ClusteringMethod):
    """BIRCH clustering (current baseline method)."""

    def fit_predict(
        self,
        data: np.ndarray,
        threshold: float = 0.5,
        branching_factor: int = 50,
        normalize: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, Tuple[Birch, Optional[StandardScaler]]]:
        """
        Perform BIRCH clustering on features.

        Args:
            data: Feature array (n_samples, n_features)
            threshold: Radius of subcluster
            branching_factor: Max CF subclusters per node
            normalize: Whether to apply StandardScaler
            **kwargs: Additional parameters (ignored)

        Returns:
            Tuple of (cluster_labels, (birch_model, scaler))
        """
        # Normalize features
        scaler = None
        if normalize:
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(data)
        else:
            data_normalized = data

        # Apply BIRCH clustering
        birch = Birch(
            n_clusters=None,
            threshold=threshold,
            branching_factor=branching_factor
        )
        cluster_labels = birch.fit_predict(data_normalized)

        return cluster_labels, (birch, scaler)

    def get_name(self) -> str:
        return "BIRCH"

    def get_description(self) -> str:
        return "Memory-efficient hierarchical clustering (current baseline)"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'threshold': 0.5,
            'branching_factor': 50,
            'normalize': True
        }

    def supports_large_datasets(self) -> bool:
        return True


class KMeansClustering(ClusteringMethod):
    """Standard k-means clustering."""

    def fit_predict(
        self,
        data: np.ndarray,
        n_clusters: int = 5,
        init: str = 'k-means++',
        normalize: bool = True,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[np.ndarray, Tuple[KMeans, Optional[StandardScaler]]]:
        """
        Perform k-means clustering on features.

        Args:
            data: Feature array (n_samples, n_features)
            n_clusters: Number of clusters
            init: Initialization method ('k-means++', 'random')
            normalize: Whether to apply StandardScaler
            random_state: Random seed
            **kwargs: Additional parameters passed to KMeans

        Returns:
            Tuple of (cluster_labels, (kmeans_model, scaler))
        """
        # Normalize features
        scaler = None
        if normalize:
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(data)
        else:
            data_normalized = data

        # Apply k-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            random_state=random_state,
            **kwargs
        )
        cluster_labels = kmeans.fit_predict(data_normalized)

        return cluster_labels, (kmeans, scaler)

    def get_name(self) -> str:
        return "K-Means"

    def get_description(self) -> str:
        return "Standard k-means clustering with k-means++ initialization"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': 5,
            'init': 'k-means++',
            'normalize': True,
            'random_state': 42
        }

    def supports_large_datasets(self) -> bool:
        return True


class MiniBatchKMeansClustering(ClusteringMethod):
    """Mini-batch k-means clustering for very large datasets."""

    def fit_predict(
        self,
        data: np.ndarray,
        n_clusters: int = 5,
        batch_size: int = 1000,
        normalize: bool = True,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[np.ndarray, Tuple[MiniBatchKMeans, Optional[StandardScaler]]]:
        """
        Perform mini-batch k-means clustering on features.

        Args:
            data: Feature array (n_samples, n_features)
            n_clusters: Number of clusters
            batch_size: Size of mini-batches
            normalize: Whether to apply StandardScaler
            random_state: Random seed
            **kwargs: Additional parameters

        Returns:
            Tuple of (cluster_labels, (mb_kmeans_model, scaler))
        """
        # Normalize features
        scaler = None
        if normalize:
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(data)
        else:
            data_normalized = data

        # Apply mini-batch k-means
        mb_kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            **kwargs
        )
        cluster_labels = mb_kmeans.fit_predict(data_normalized)

        return cluster_labels, (mb_kmeans, scaler)

    def get_name(self) -> str:
        return "Mini-Batch K-Means"

    def get_description(self) -> str:
        return "Scalable k-means variant using mini-batches (for very large datasets)"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': 5,
            'batch_size': 1000,
            'normalize': True,
            'random_state': 42
        }

    def supports_large_datasets(self) -> bool:
        return True


class KShapeClustering(ClusteringMethod):
    """k-Shape clustering for time series data."""

    def fit_predict(
        self,
        data: np.ndarray,
        n_clusters: int = 5,
        max_iter: int = 100,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[np.ndarray, KShape]:
        """
        Perform k-Shape clustering on time series.

        Args:
            data: Time series array (n_samples, n_timepoints) or DataFrame with 'powers'
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            random_state: Random seed
            **kwargs: Additional parameters

        Returns:
            Tuple of (cluster_labels, kshape_model)

        Note:
            k-Shape uses shape-based distance (normalized cross-correlation).
            Fast and suitable for large datasets. Data should be raw time series,
            not extracted features.
        """
        # Convert to tslearn format if needed
        if isinstance(data, pd.DataFrame):
            # Assume DataFrame has 'powers' column with arrays
            ts_data = to_time_series_dataset([row['powers'] for _, row in data.iterrows()])
        elif len(data.shape) == 2:
            # Already in correct format (n_samples, n_timepoints)
            ts_data = to_time_series_dataset(data)
        else:
            raise ValueError("Data must be 2D array or DataFrame with 'powers' column")

        # Apply k-Shape
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            kshape = KShape(
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=random_state,
                **kwargs
            )
            cluster_labels = kshape.fit_predict(ts_data)

        return cluster_labels, kshape

    def get_name(self) -> str:
        return "k-Shape"

    def get_description(self) -> str:
        return "Time series clustering using shape-based distance (fast, recommended for RF spectra)"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': 5,
            'max_iter': 100,
            'random_state': 42
        }

    def supports_large_datasets(self) -> bool:
        return True  # O(n log n) complexity


class DBSCANClustering(ClusteringMethod):
    """DBSCAN clustering for outlier detection."""

    def fit_predict(
        self,
        data: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
        normalize: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, Tuple[DBSCAN, Optional[StandardScaler]]]:
        """
        Perform DBSCAN clustering on features.

        Args:
            data: Feature array (n_samples, n_features)
            eps: Maximum distance between samples in same neighborhood
            min_samples: Minimum samples in neighborhood to form core point
            normalize: Whether to apply StandardScaler
            **kwargs: Additional parameters

        Returns:
            Tuple of (cluster_labels, (dbscan_model, scaler))

        Note:
            Label -1 indicates outliers/noise points.
        """
        # Normalize features
        scaler = None
        if normalize:
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(data)
        else:
            data_normalized = data

        # Apply DBSCAN
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            **kwargs
        )
        cluster_labels = dbscan.fit_predict(data_normalized)

        return cluster_labels, (dbscan, scaler)

    def get_name(self) -> str:
        return "DBSCAN"

    def get_description(self) -> str:
        return "Density-based clustering with outlier detection (label -1 = outliers)"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'eps': 0.5,
            'min_samples': 5,
            'normalize': True
        }

    def supports_large_datasets(self) -> bool:
        return False  # O(n²) in worst case


# Factory function
def get_clustering_method(method: str) -> ClusteringMethod:
    """
    Get clustering method by name.

    Args:
        method: Name of clustering method
                ('birch', 'kmeans', 'minibatch_kmeans', 'kshape', 'dbscan')

    Returns:
        ClusteringMethod instance
    """
    methods = {
        'birch': BIRCHClustering,
        'kmeans': KMeansClustering,
        'minibatch_kmeans': MiniBatchKMeansClustering,
        'kshape': KShapeClustering,
        'dbscan': DBSCANClustering
    }

    method_lower = method.lower()
    if method_lower not in methods:
        raise ValueError(
            f"Unknown clustering method: {method}. "
            f"Available methods: {list(methods.keys())}"
        )

    return methods[method_lower]()


def get_available_methods() -> Dict[str, str]:
    """
    Get dict of available clustering methods.

    Returns:
        Dict mapping method name to description
    """
    methods = ['birch', 'kmeans', 'minibatch_kmeans', 'kshape', 'dbscan']
    return {
        method: get_clustering_method(method).get_description()
        for method in methods
    }


def evaluate_clustering(
    features: np.ndarray,
    labels: np.ndarray,
    metric: str = 'all'
) -> Dict[str, float]:
    """
    Evaluate clustering quality using unsupervised metrics.

    Args:
        features: Feature array (n_samples, n_features)
        labels: Cluster labels
        metric: Which metric to compute ('silhouette', 'davies_bouldin',
                'calinski_harabasz', or 'all')

    Returns:
        Dict of metric name to value

    Notes:
        - Silhouette score: [-1, 1], higher is better
        - Davies-Bouldin index: [0, inf), lower is better
        - Calinski-Harabasz score: [0, inf), higher is better
    """
    results = {}

    # Filter out noise points (label -1 from DBSCAN)
    mask = labels >= 0
    if np.sum(mask) < 2:
        # Not enough valid samples
        return {'error': 'insufficient_samples'}

    features_filtered = features[mask]
    labels_filtered = labels[mask]

    n_clusters = len(np.unique(labels_filtered))
    if n_clusters < 2:
        # Need at least 2 clusters for metrics
        return {'error': 'single_cluster'}

    # Compute metrics
    if metric in ['silhouette', 'all']:
        try:
            results['silhouette_score'] = silhouette_score(
                features_filtered,
                labels_filtered
            )
        except Exception as e:
            results['silhouette_score'] = float('nan')

    if metric in ['davies_bouldin', 'all']:
        try:
            results['davies_bouldin_index'] = davies_bouldin_score(
                features_filtered,
                labels_filtered
            )
        except Exception as e:
            results['davies_bouldin_index'] = float('nan')

    if metric in ['calinski_harabasz', 'all']:
        try:
            results['calinski_harabasz_score'] = calinski_harabasz_score(
                features_filtered,
                labels_filtered
            )
        except Exception as e:
            results['calinski_harabasz_score'] = float('nan')

    return results


# Legacy compatibility
def perform_clustering(
    features: np.ndarray,
    threshold: float = 0.5,
    branching_factor: int = 50
) -> Tuple[np.ndarray, Birch, StandardScaler]:
    """
    Perform BIRCH clustering (legacy interface for backward compatibility).

    Args:
        features: Feature array from extract_features()
        threshold: Radius of subcluster
        branching_factor: Maximum CF subclusters per node

    Returns:
        Tuple of (cluster_labels, birch_model, scaler)
    """
    method = BIRCHClustering()
    labels, (birch, scaler) = method.fit_predict(
        features,
        threshold=threshold,
        branching_factor=branching_factor,
        normalize=True
    )
    return labels, birch, scaler
