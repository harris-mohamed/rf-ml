"""Configuration system for clustering pipeline."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime


@dataclass
class ClusteringConfig:
    """Configuration for the entire clustering pipeline."""

    # Normalization
    normalization_method: str = 'energy'  # 'none', 'energy', 'robust', 'minmax', 'zscore'

    # Feature extraction
    include_statistical_features: bool = True
    include_shape_features: bool = True
    include_rf_domain_features: bool = True
    include_model_features: bool = True
    peak_prominence: float = 5.0
    n_fft_coeffs: int = 5

    # Clustering
    clustering_method: str = 'kshape'  # 'birch', 'kmeans', 'minibatch_kmeans', 'kshape', 'dbscan'
    n_clusters: int = 5
    random_state: int = 42

    # Method-specific parameters
    birch_threshold: float = 0.5
    birch_branching_factor: int = 50

    kmeans_init: str = 'k-means++'
    kmeans_max_iter: int = 300

    minibatch_batch_size: int = 1000

    kshape_max_iter: int = 100

    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5

    # Outlier detection
    enable_outlier_detection: bool = False
    outlier_method: str = 'isolation_forest'  # 'isolation_forest', 'lof', 'statistical', 'cluster_based'
    outlier_contamination: float = 0.1
    outlier_threshold: float = 3.0

    # Performance
    normalize_features: bool = True
    use_caching: bool = True
    n_jobs: int = -1  # Number of parallel jobs (-1 = all cores)

    # Metadata
    config_name: str = "default"
    created_at: Optional[str] = None

    def __post_init__(self):
        """Set created_at timestamp if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filepath: str):
        """
        Save config to JSON file.

        Args:
            filepath: Path to save config file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Dictionary of config parameters

        Returns:
            ClusteringConfig instance
        """
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'ClusteringConfig':
        """
        Create config from JSON string.

        Args:
            json_str: JSON string

        Returns:
            ClusteringConfig instance
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def load(cls, filepath: str) -> 'ClusteringConfig':
        """
        Load config from JSON file.

        Args:
            filepath: Path to config file

        Returns:
            ClusteringConfig instance
        """
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate configuration parameters.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate normalization method
        valid_norm = ['none', 'energy', 'robust', 'minmax', 'zscore']
        if self.normalization_method not in valid_norm:
            return False, f"Invalid normalization_method. Must be one of {valid_norm}"

        # Validate clustering method
        valid_cluster = ['birch', 'kmeans', 'minibatch_kmeans', 'kshape', 'dbscan']
        if self.clustering_method not in valid_cluster:
            return False, f"Invalid clustering_method. Must be one of {valid_cluster}"

        # Validate outlier method
        valid_outlier = ['isolation_forest', 'lof', 'statistical', 'cluster_based']
        if self.outlier_method not in valid_outlier:
            return False, f"Invalid outlier_method. Must be one of {valid_outlier}"

        # Validate numeric parameters
        if self.n_clusters < 2:
            return False, "n_clusters must be >= 2"

        if self.peak_prominence < 0:
            return False, "peak_prominence must be >= 0"

        if self.n_fft_coeffs < 1:
            return False, "n_fft_coeffs must be >= 1"

        if self.outlier_contamination < 0 or self.outlier_contamination > 0.5:
            return False, "outlier_contamination must be between 0 and 0.5"

        # Validate method-specific parameters
        if self.clustering_method == 'birch':
            if self.birch_threshold <= 0:
                return False, "birch_threshold must be > 0"
            if self.birch_branching_factor < 2:
                return False, "birch_branching_factor must be >= 2"

        if self.clustering_method in ['kmeans', 'kshape']:
            if self.kmeans_init not in ['k-means++', 'random']:
                return False, "kmeans_init must be 'k-means++' or 'random'"

        if self.clustering_method == 'minibatch_kmeans':
            if self.minibatch_batch_size < 1:
                return False, "minibatch_batch_size must be >= 1"

        if self.clustering_method == 'dbscan':
            if self.dbscan_eps <= 0:
                return False, "dbscan_eps must be > 0"
            if self.dbscan_min_samples < 1:
                return False, "dbscan_min_samples must be >= 1"

        return True, None

    def get_feature_config(self):
        """Get feature extraction configuration."""
        from .feature_extraction import FeatureConfig
        return FeatureConfig(
            include_statistical=self.include_statistical_features,
            include_shape=self.include_shape_features,
            include_rf_domain=self.include_rf_domain_features,
            include_model=self.include_model_features
        )

    def get_clustering_params(self) -> Dict[str, Any]:
        """Get parameters for selected clustering method."""
        if self.clustering_method == 'birch':
            return {
                'threshold': self.birch_threshold,
                'branching_factor': self.birch_branching_factor,
                'normalize': self.normalize_features
            }
        elif self.clustering_method == 'kmeans':
            return {
                'n_clusters': self.n_clusters,
                'init': self.kmeans_init,
                'max_iter': self.kmeans_max_iter,
                'normalize': self.normalize_features,
                'random_state': self.random_state
            }
        elif self.clustering_method == 'minibatch_kmeans':
            return {
                'n_clusters': self.n_clusters,
                'batch_size': self.minibatch_batch_size,
                'normalize': self.normalize_features,
                'random_state': self.random_state
            }
        elif self.clustering_method == 'kshape':
            return {
                'n_clusters': self.n_clusters,
                'max_iter': self.kshape_max_iter,
                'random_state': self.random_state
            }
        elif self.clustering_method == 'dbscan':
            return {
                'eps': self.dbscan_eps,
                'min_samples': self.dbscan_min_samples,
                'normalize': self.normalize_features
            }
        else:
            return {}

    def get_outlier_params(self) -> Dict[str, Any]:
        """Get parameters for selected outlier detection method."""
        if self.outlier_method == 'isolation_forest':
            return {
                'contamination': self.outlier_contamination,
                'random_state': self.random_state
            }
        elif self.outlier_method == 'lof':
            return {
                'contamination': self.outlier_contamination
            }
        elif self.outlier_method == 'statistical':
            return {
                'threshold': self.outlier_threshold
            }
        elif self.outlier_method == 'cluster_based':
            return {
                'threshold': self.outlier_threshold
            }
        else:
            return {}


# Preset configurations
PRESET_CONFIGS = {
    "default": ClusteringConfig(
        config_name="default",
        normalization_method='energy',
        clustering_method='kshape',
        n_clusters=5
    ),
    "fast": ClusteringConfig(
        config_name="fast",
        normalization_method='minmax',
        clustering_method='minibatch_kmeans',
        include_model_features=False,  # Skip slower features
        n_clusters=5
    ),
    "comprehensive": ClusteringConfig(
        config_name="comprehensive",
        normalization_method='robust',
        clustering_method='kshape',
        include_statistical_features=True,
        include_shape_features=True,
        include_rf_domain_features=True,
        include_model_features=True,
        n_clusters=5,
        enable_outlier_detection=True,
        outlier_method='isolation_forest'
    ),
    "outlier_focused": ClusteringConfig(
        config_name="outlier_focused",
        normalization_method='robust',
        clustering_method='dbscan',
        dbscan_eps=0.5,
        dbscan_min_samples=5,
        enable_outlier_detection=True,
        outlier_method='isolation_forest'
    ),
    "shape_based": ClusteringConfig(
        config_name="shape_based",
        normalization_method='energy',
        clustering_method='kshape',
        include_statistical_features=False,
        include_shape_features=True,
        include_rf_domain_features=True,
        include_model_features=False,
        n_clusters=5
    )
}


def get_preset_config(preset_name: str) -> ClusteringConfig:
    """
    Get a preset configuration.

    Args:
        preset_name: Name of preset ('default', 'fast', 'comprehensive',
                     'outlier_focused', 'shape_based')

    Returns:
        ClusteringConfig instance
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {list(PRESET_CONFIGS.keys())}"
        )

    return PRESET_CONFIGS[preset_name]


def list_presets() -> Dict[str, str]:
    """
    Get list of available preset configurations.

    Returns:
        Dict mapping preset name to description
    """
    descriptions = {
        "default": "Balanced configuration with k-Shape clustering and energy normalization",
        "fast": "Fast configuration for large datasets using mini-batch k-means",
        "comprehensive": "Complete feature set with outlier detection",
        "outlier_focused": "DBSCAN clustering for interferer/outlier detection",
        "shape_based": "Focus on shape and RF domain features with k-Shape clustering"
    }

    return descriptions
