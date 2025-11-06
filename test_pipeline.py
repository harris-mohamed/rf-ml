"""Quick test of the refactored clustering pipeline."""

import sys
sys.path.insert(0, 'src/dashboard')

from components import data_loader
from components.normalization import get_normalization_strategy
from components.feature_extraction import CompositeFeatureExtractor, FeatureConfig
from components.clustering_methods import get_clustering_method
from components.evaluation import ClusteringEvaluator
from components.config import ClusteringConfig

print("Testing RF-ML Clustering Pipeline")
print("=" * 60)

# Load sample data
print("\n1. Loading data...")
df = data_loader.load_spectrum_data("dev_output/sim_01_20250115-000000.json")
print(f"   ✓ Loaded {len(df)} spectrum snapshots")

# Test normalization
print("\n2. Testing normalization...")
normalizer = get_normalization_strategy('energy')
df_norm, params = normalizer.normalize(df)
print(f"   ✓ Energy normalization complete")

# Test feature extraction
print("\n3. Testing feature extraction...")
config = FeatureConfig(
    include_statistical=True,
    include_shape=True,
    include_rf_domain=True,
    include_model=True
)
extractor = CompositeFeatureExtractor(config)
features, feature_names = extractor.extract_features(df_norm)
print(f"   ✓ Extracted {len(feature_names)} features")
print(f"   ✓ Features shape: {features.shape}")

# Test clustering
print("\n4. Testing k-Shape clustering...")
clustering_method = get_clustering_method('kshape')
cluster_labels, model = clustering_method.fit_predict(df_norm, n_clusters=5)
print(f"   ✓ Clustering complete")
print(f"   ✓ Found {len(set(cluster_labels))} clusters")

# Test evaluation
print("\n5. Testing evaluation...")
evaluator = ClusteringEvaluator()
metrics = evaluator.evaluate(features, cluster_labels)
print(f"   ✓ Evaluation complete")
if 'silhouette_score' in metrics:
    print(f"   ✓ Silhouette score: {metrics['silhouette_score']:.4f}")
if 'davies_bouldin_index' in metrics:
    print(f"   ✓ Davies-Bouldin index: {metrics['davies_bouldin_index']:.4f}")

print("\n" + "=" * 60)
print("All tests passed! ✅")
print("\nYou can now run the dashboard with:")
print("  streamlit run src/dashboard/app.py")
