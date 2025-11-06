"""
RF-ML Clustering Dashboard (Refactored)

Interactive Streamlit dashboard for visualizing and analyzing RF spectrum clustering
with functional data analysis approaches.

Usage:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# Import dashboard components
from components import data_loader, visualizations
from components.normalization import get_normalization_strategy, get_available_normalizations
from components.feature_extraction import CompositeFeatureExtractor, FeatureConfig
from components.clustering_methods import get_clustering_method, get_available_methods, evaluate_clustering
from components.outlier_detection import get_outlier_detector, get_available_detectors, get_outlier_statistics
from components.evaluation import ClusteringEvaluator, compare_clustering_results
from components.config import ClusteringConfig, get_preset_config, list_presets
from components import clustering_legacy  # For backward compatibility

from utils.helpers import format_frequency, format_time, format_duration, get_color_palette


# Page configuration
st.set_page_config(
    page_title="RF-ML Clustering Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def run_clustering_pipeline(df: pd.DataFrame, config: ClusteringConfig):
    """
    Run the complete clustering pipeline.

    Args:
        df: DataFrame with spectrum data
        config: ClusteringConfig with pipeline parameters

    Returns:
        Dict with all pipeline results
    """
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Normalization (10%)
    status_text.text("Step 1/6: Normalizing spectrum data...")
    progress_bar.progress(10)

    normalizer = get_normalization_strategy(config.normalization_method)
    df_normalized, norm_params = normalizer.normalize(df)
    results['df_normalized'] = df_normalized
    results['normalization_params'] = norm_params
    results['normalization_method'] = config.normalization_method

    # Step 2: Feature Extraction (30%)
    status_text.text("Step 2/6: Extracting features...")
    progress_bar.progress(30)

    feature_config = config.get_feature_config()
    extractor = CompositeFeatureExtractor(feature_config)
    features, feature_names = extractor.extract_features(df_normalized)
    results['features'] = features
    results['feature_names'] = feature_names
    results['feature_info'] = extractor.get_feature_info()
    results['n_features'] = extractor.get_total_feature_count()

    # Step 3: Clustering (50%)
    status_text.text(f"Step 3/6: Performing {config.clustering_method.upper()} clustering...")
    progress_bar.progress(50)

    clustering_method = get_clustering_method(config.clustering_method)
    clustering_params = config.get_clustering_params()

    # Special handling for k-Shape (needs raw time series)
    if config.clustering_method == 'kshape':
        cluster_labels, model = clustering_method.fit_predict(df_normalized, **clustering_params)
    else:
        cluster_labels, model = clustering_method.fit_predict(features, **clustering_params)

    results['cluster_labels'] = cluster_labels
    results['clustering_model'] = model
    results['clustering_method'] = config.clustering_method

    # Step 4: Outlier Detection (70%)
    if config.enable_outlier_detection:
        status_text.text(f"Step 4/6: Detecting outliers using {config.outlier_method}...")
        progress_bar.progress(70)

        outlier_detector = get_outlier_detector(config.outlier_method)
        outlier_params = config.get_outlier_params()

        outlier_labels, outlier_scores, outlier_model = outlier_detector.fit_predict(
            features,
            **outlier_params
        )

        results['outlier_labels'] = outlier_labels
        results['outlier_scores'] = outlier_scores
        results['outlier_model'] = outlier_model
        results['outlier_stats'] = get_outlier_statistics(outlier_labels, outlier_scores)
    else:
        results['outlier_labels'] = None
        progress_bar.progress(70)

    # Step 5: Evaluation (85%)
    status_text.text("Step 5/6: Computing evaluation metrics...")
    progress_bar.progress(85)

    evaluator = ClusteringEvaluator()
    eval_metrics = evaluator.evaluate(features, cluster_labels)
    results['evaluation_metrics'] = eval_metrics
    results['evaluator'] = evaluator

    # Step 6: Add labels to dataframe (100%)
    status_text.text("Step 6/6: Finalizing results...")
    progress_bar.progress(100)

    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels

    if config.enable_outlier_detection:
        df_clustered['outlier'] = outlier_labels
        df_clustered['outlier_score'] = outlier_scores

    results['df_clustered'] = df_clustered

    # Compute cluster statistics
    results['cluster_stats'] = clustering_legacy.get_cluster_statistics(df_clustered)

    status_text.text("✅ Pipeline complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

    return results


def main():
    """Main dashboard application."""

    # Title
    st.title("📡 RF Spectrum Clustering Dashboard")
    st.markdown("*Functional Data Analysis Platform*")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Preset selector
        st.subheader("Quick Start")
        preset_descriptions = list_presets()
        preset_names = list(preset_descriptions.keys())

        selected_preset = st.selectbox(
            "Configuration Preset",
            options=preset_names,
            format_func=lambda x: f"{x.title()}",
            help="Choose a preset configuration or customize below"
        )

        # Show preset description
        st.info(f"**{selected_preset.title()}:** {preset_descriptions[selected_preset]}")

        use_preset = st.checkbox("Use preset as-is", value=True)

        st.markdown("---")

        # Data source selection
        st.subheader("1. Data Source")

        # List available JSON files
        available_files = data_loader.list_available_json_files()

        if not available_files:
            st.warning("No JSON files found in dev_output/ directory")
            st.stop()

        # File selector
        selected_file = st.selectbox(
            "Select data file",
            options=available_files,
            format_func=lambda x: x.split('/')[-1]
        )

        # Show file info
        file_info = data_loader.get_file_info(selected_file)
        if file_info['exists']:
            st.info(
                f"**Size:** {file_info['size_mb']:.1f} MB\n\n"
                f"**Records:** {file_info['num_records']}\n\n"
                f"**Modified:** {file_info['modified'].strftime('%Y-%m-%d %H:%M')}"
            )

        st.markdown("---")

        # Configuration
        if use_preset:
            config = get_preset_config(selected_preset)
        else:
            st.subheader("2. Custom Configuration")

            # Normalization
            with st.expander("Normalization", expanded=False):
                norm_options = list(get_available_normalizations().keys())
                normalization_method = st.selectbox(
                    "Normalization Method",
                    options=norm_options,
                    index=norm_options.index('energy')
                )

            # Feature Selection
            with st.expander("Feature Selection", expanded=False):
                include_statistical = st.checkbox("Statistical Features", value=True)
                include_shape = st.checkbox("Shape Features", value=True)
                include_rf_domain = st.checkbox("RF Domain Features", value=True)
                include_model = st.checkbox("Model Features", value=True)

            # Clustering
            with st.expander("Clustering Method", expanded=True):
                cluster_options = list(get_available_methods().keys())
                clustering_method = st.selectbox(
                    "Algorithm",
                    options=cluster_options,
                    index=cluster_options.index('kshape')
                )

                n_clusters = st.slider("Number of Clusters", 2, 20, 5)

                if clustering_method == 'birch':
                    birch_threshold = st.slider("Threshold", 0.1, 2.0, 0.5, 0.1)
                elif clustering_method in ['kmeans', 'kshape']:
                    pass  # Use defaults
                elif clustering_method == 'dbscan':
                    dbscan_eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
                    dbscan_min_samples = st.slider("Min Samples", 2, 20, 5)

            # Build config
            config = ClusteringConfig(
                normalization_method=normalization_method,
                include_statistical_features=include_statistical,
                include_shape_features=include_shape,
                include_rf_domain_features=include_rf_domain,
                include_model_features=include_model,
                clustering_method=clustering_method,
                n_clusters=n_clusters,
                enable_outlier_detection=False  # Will be set below
            )

            # Update method-specific params
            if clustering_method == 'birch' and 'birch_threshold' in locals():
                config.birch_threshold = birch_threshold
            elif clustering_method == 'dbscan' and 'dbscan_eps' in locals():
                config.dbscan_eps = dbscan_eps
                config.dbscan_min_samples = dbscan_min_samples

        st.markdown("---")

        # Outlier Detection (always visible regardless of preset/custom)
        st.subheader("2. Outlier Detection")
        enable_outlier = st.checkbox(
            "Enable Outlier Detection",
            value=config.enable_outlier_detection,
            help="Detect interferers and anomalous spectra"
        )

        if enable_outlier:
            outlier_options = list(get_available_detectors().keys())
            outlier_method = st.selectbox(
                "Detection Method",
                options=outlier_options,
                index=outlier_options.index(config.outlier_method)
            )
            outlier_contamination = st.slider(
                "Expected Outlier Percentage",
                0.01, 0.5, config.outlier_contamination, 0.01,
                help="Fraction of data expected to be outliers (0.01 = 1%)"
            )

            # Update config
            config.enable_outlier_detection = True
            config.outlier_method = outlier_method
            config.outlier_contamination = outlier_contamination
        else:
            config.enable_outlier_detection = False

        st.markdown("---")

        # Run button
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Main content area
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None

    if run_button:
        with st.spinner("Loading data..."):
            try:
                # Load spectrum data
                df = data_loader.load_spectrum_data(selected_file)
                summary = data_loader.get_dataset_summary(df)
                time_array, frequency_array, psd_array = data_loader.build_waterfall_arrays(df)

                st.session_state.df = df
                st.session_state.summary = summary
                st.session_state.time_array = time_array
                st.session_state.frequency_array = frequency_array
                st.session_state.psd_array = psd_array

            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

        # Run clustering pipeline
        try:
            pipeline_results = run_clustering_pipeline(df, config)
            st.session_state.pipeline_results = pipeline_results
            st.session_state.config = config

            st.success("✅ Analysis complete!")

        except Exception as e:
            st.error(f"Error during analysis: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    # Display results
    if st.session_state.pipeline_results is not None:
        results = st.session_state.pipeline_results
        config = st.session_state.config

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Snapshots", st.session_state.summary['num_snapshots'])

        with col2:
            st.metric("Clusters", results['cluster_stats']['num_clusters'])

        with col3:
            st.metric("Features", results['n_features'])

        with col4:
            if results['outlier_labels'] is not None:
                outlier_pct = results['outlier_stats']['outlier_percentage']
                st.metric("Outliers", f"{outlier_pct:.1f}%")
            else:
                st.metric("Outliers", "N/A")

        with col5:
            if 'silhouette_score' in results['evaluation_metrics']:
                sil_score = results['evaluation_metrics']['silhouette_score']
                st.metric("Silhouette", f"{sil_score:.3f}")
            else:
                st.metric("Silhouette", "N/A")

        st.markdown("---")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Cluster Visualization",
            "📈 Method Evaluation",
            "🔬 Feature Analysis",
            "⚠️ Outlier Analysis"
        ])

        with tab1:
            st.subheader("Cluster Distribution")

            # Pie chart
            pie_fig = visualizations.plot_cluster_pie_chart(results['cluster_stats'])
            st.plotly_chart(pie_fig, use_container_width=True)

            st.markdown("---")

            # Waterfall plot
            st.subheader("Spectrum Waterfall")
            waterfall_fig = visualizations.plot_waterfall(
                st.session_state.time_array,
                st.session_state.frequency_array,
                st.session_state.psd_array,
                show_transponder_lines=True,
                title="Spectrum Waterfall (All Data)"
            )
            st.plotly_chart(waterfall_fig, use_container_width=True)

            st.markdown("---")

            # Cluster timeline
            st.subheader("Cluster Timeline")
            timeline_fig = visualizations.plot_all_clusters_timeline(results['df_clustered'])
            st.plotly_chart(timeline_fig, use_container_width=True)

            st.markdown("---")

            # Individual cluster analysis
            st.subheader("Individual Cluster Analysis")

            available_clusters = sorted(results['df_clustered']['cluster'].unique())
            if len(available_clusters) == 0 or (len(available_clusters) == 1 and available_clusters[0] == -1):
                st.warning("No valid clusters found")
            else:
                # Filter out -1 (noise) if present
                valid_clusters = [c for c in available_clusters if c >= 0]

                if valid_clusters:
                    selected_cluster = st.selectbox(
                        "Select cluster",
                        options=valid_clusters,
                        format_func=lambda x: f"Cluster {x}"
                    )

                    cluster_count = results['cluster_stats']['cluster_counts'][selected_cluster]
                    cluster_pct = results['cluster_stats']['cluster_percentages'][selected_cluster]

                    st.info(f"**Cluster {selected_cluster}**: {cluster_count} snapshots ({cluster_pct:.1f}%)")

                    # Traces
                    frequencies, min_trace, max_trace, avg_trace = clustering_legacy.get_cluster_traces(
                        results['df_clustered'],
                        selected_cluster
                    )

                    colors = get_color_palette(len(valid_clusters))
                    cluster_color = colors[valid_clusters.index(selected_cluster)]

                    traces_fig = visualizations.plot_cluster_traces(
                        frequencies, min_trace, max_trace, avg_trace,
                        selected_cluster, color=cluster_color
                    )
                    st.plotly_chart(traces_fig, use_container_width=True)

        with tab2:
            st.subheader("Clustering Evaluation Metrics")

            eval_metrics = results['evaluation_metrics']

            # Display metrics
            if 'error' in eval_metrics:
                st.error(f"Evaluation error: {eval_metrics['error']}")
            else:
                metric_cols = st.columns(3)

                with metric_cols[0]:
                    if 'silhouette_score' in eval_metrics:
                        st.metric("Silhouette Score", f"{eval_metrics['silhouette_score']:.4f}",
                                 help="Range: [-1, 1]. Higher is better. >0.5 = good separation")

                with metric_cols[1]:
                    if 'davies_bouldin_index' in eval_metrics:
                        st.metric("Davies-Bouldin Index", f"{eval_metrics['davies_bouldin_index']:.4f}",
                                 help="Range: [0, ∞). Lower is better. <1.0 = good clustering")

                with metric_cols[2]:
                    if 'calinski_harabasz_score' in eval_metrics:
                        st.metric("Calinski-Harabasz Score", f"{eval_metrics['calinski_harabasz_score']:.1f}",
                                 help="Range: [0, ∞). Higher is better. >500 = good clustering")

                # Interpretations
                st.markdown("---")
                st.subheader("Metric Interpretations")

                interpretations = results['evaluator'].interpret_metrics(eval_metrics)
                for metric, interpretation in interpretations.items():
                    st.write(f"**{metric.replace('_', ' ').title()}:** {interpretation}")

            st.markdown("---")
            st.subheader("Configuration Summary")

            config_summary = {
                "Normalization": config.normalization_method,
                "Clustering Method": config.clustering_method,
                "Number of Clusters": config.n_clusters,
                "Feature Categories": f"{sum([config.include_statistical_features, config.include_shape_features, config.include_rf_domain_features, config.include_model_features])}/4 enabled",
                "Total Features": results['n_features'],
                "Outlier Detection": "Enabled" if config.enable_outlier_detection else "Disabled"
            }

            for key, value in config_summary.items():
                st.write(f"**{key}:** {value}")

        with tab3:
            st.subheader("Feature Analysis")

            # Feature categories
            st.write("**Features by Category:**")
            feature_info = results['feature_info']

            for category, features in feature_info.items():
                with st.expander(f"{category} ({len(features)} features)", expanded=False):
                    st.write(", ".join(features))

            st.markdown("---")

            # Feature statistics
            st.subheader("Feature Statistics")

            features = results['features']
            feature_names = results['feature_names']

            feature_stats = pd.DataFrame({
                'Feature': feature_names,
                'Mean': np.mean(features, axis=0),
                'Std': np.std(features, axis=0),
                'Min': np.min(features, axis=0),
                'Max': np.max(features, axis=0)
            })

            st.dataframe(feature_stats, use_container_width=True)

        with tab4:
            st.subheader("Outlier Analysis")

            if results['outlier_labels'] is None:
                st.info("Outlier detection was not enabled. Enable it in the sidebar configuration to analyze outliers.")
            else:
                outlier_stats = results['outlier_stats']

                # Statistics
                stat_cols = st.columns(4)

                with stat_cols[0]:
                    st.metric("Total Outliers", outlier_stats['n_outliers'])

                with stat_cols[1]:
                    st.metric("Outlier Percentage", f"{outlier_stats['outlier_percentage']:.1f}%")

                with stat_cols[2]:
                    st.metric("Mean Outlier Score", f"{outlier_stats['mean_outlier_score']:.3f}")

                with stat_cols[3]:
                    st.metric("Mean Inlier Score", f"{outlier_stats['mean_inlier_score']:.3f}")

                st.markdown("---")

                # Outlier distribution
                st.subheader("Outlier Score Distribution")

                outlier_scores = results['outlier_scores']
                outlier_labels = results['outlier_labels']

                import plotly.graph_objects as go

                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=outlier_scores[outlier_labels == 1],
                    name='Inliers',
                    opacity=0.7,
                    marker_color='green'
                ))

                fig.add_trace(go.Histogram(
                    x=outlier_scores[outlier_labels == -1],
                    name='Outliers',
                    opacity=0.7,
                    marker_color='red'
                ))

                fig.update_layout(
                    title="Outlier Score Distribution",
                    xaxis_title="Outlier Score",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                # Show outlier samples
                st.subheader("Outlier Samples")

                outlier_mask = results['df_clustered']['outlier'] == -1
                outlier_df = results['df_clustered'][outlier_mask].copy()

                if len(outlier_df) > 0:
                    st.write(f"Showing {min(10, len(outlier_df))} outlier samples (sorted by outlier score):")

                    outlier_df_display = outlier_df.sort_values('outlier_score', ascending=False).head(10)
                    outlier_df_display = outlier_df_display[['outlier_score', 'cluster']].reset_index()

                    st.dataframe(outlier_df_display, use_container_width=True)
                else:
                    st.info("No outliers detected")

    else:
        # Show instructions
        st.info(
            """
            ### 👈 Getting Started

            1. **Select a preset configuration** or customize parameters in the sidebar
            2. **Choose a data file** from available datasets
            3. **Click "Run Analysis"** to begin clustering

            ### New Features

            - **Multiple normalization methods**: Energy, robust, min-max, z-score
            - **Enhanced features**: 30-50 features across statistical, shape, RF domain, and model categories
            - **Time series clustering**: k-Shape algorithm optimized for RF spectra
            - **Outlier detection**: Isolation Forest, LOF, statistical methods
            - **Comprehensive evaluation**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz score

            ### Configuration Presets

            - **Default**: Balanced k-Shape clustering with energy normalization
            - **Fast**: Mini-batch k-means for large datasets
            - **Comprehensive**: Full feature set with outlier detection
            - **Outlier Focused**: DBSCAN for interferer detection
            - **Shape Based**: Focus on shape and RF domain features
            """
        )

        # Show example datasets
        if available_files:
            st.markdown("### 📁 Available Datasets")

            for file in available_files[:5]:
                info = data_loader.get_file_info(file)
                if info['exists']:
                    st.write(
                        f"- **{file.split('/')[-1]}** "
                        f"({info['size_mb']:.1f} MB, {info['num_records']} records)"
                    )


if __name__ == "__main__":
    main()
