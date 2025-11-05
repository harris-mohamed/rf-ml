"""
RF-ML Clustering Dashboard

Interactive Streamlit dashboard for visualizing and analyzing RF spectrum clustering results.

Usage:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd

# Import dashboard components
from components import data_loader, clustering, visualizations
from utils.helpers import format_frequency, format_time, format_duration, get_color_palette


# Page configuration
st.set_page_config(
    page_title="RF-ML Clustering Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main dashboard application."""

    # Title
    st.title("📡 RF Spectrum Clustering Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

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
            format_func=lambda x: x.split('/')[-1]  # Show just filename
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

        # Clustering parameters
        st.subheader("2. Clustering Parameters")

        threshold = st.slider(
            "Threshold",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Radius of subcluster - lower values create more clusters, higher values create fewer clusters"
        )

        st.info("💡 BIRCH automatically determines the number of clusters based on the threshold parameter.")

        # Run clustering button
        run_clustering = st.button("🔄 Run Clustering", type="primary", use_container_width=True)

    # Main content area

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # Load data if file selected and button clicked
    if run_clustering or st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            try:
                # Load spectrum data
                df = data_loader.load_spectrum_data(selected_file)

                # Get summary
                summary = data_loader.get_dataset_summary(df)

                # Build waterfall arrays
                time_array, frequency_array, psd_array = data_loader.build_waterfall_arrays(df)

                st.session_state.data_loaded = True
                st.session_state.df = df
                st.session_state.summary = summary
                st.session_state.time_array = time_array
                st.session_state.frequency_array = frequency_array
                st.session_state.psd_array = psd_array

            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

        # Perform clustering
        with st.spinner("Extracting features and clustering..."):
            try:
                # Extract features
                features = clustering.extract_features(st.session_state.df)

                # Perform clustering
                cluster_labels, birch_model, scaler = clustering.perform_clustering(
                    features,
                    threshold=threshold
                )

                # Add cluster labels to dataframe
                df_clustered = clustering.add_cluster_labels(st.session_state.df, cluster_labels)

                # Get cluster statistics
                cluster_stats = clustering.get_cluster_statistics(df_clustered)

                st.session_state.df_clustered = df_clustered
                st.session_state.cluster_stats = cluster_stats
                st.session_state.features = features
                st.session_state.cluster_labels = cluster_labels

            except Exception as e:
                st.error(f"Error during clustering: {e}")
                st.stop()

        # Display results
        st.success("✅ Data loaded and clustered successfully!")

        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Snapshots", st.session_state.summary['num_snapshots'])

        with col2:
            st.metric("Clusters", st.session_state.cluster_stats['num_clusters'])

        with col3:
            st.metric(
                "Duration",
                format_duration(st.session_state.summary['duration_minutes'] * 60)
            )

        with col4:
            st.metric(
                "Frequency Span",
                f"{st.session_state.summary['freq_span_mhz']:.1f} MHz"
            )

        st.markdown("---")

        # Tabs for different views
        tab1, tab2 = st.tabs(["📊 Cluster Overview", "🔍 Cluster Details"])

        with tab1:
            st.subheader("Cluster Distribution")

            # Pie chart
            pie_fig = visualizations.plot_cluster_pie_chart(st.session_state.cluster_stats)
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

            # All clusters timeline
            st.subheader("Cluster Timeline (All Clusters)")

            all_timeline_fig = visualizations.plot_all_clusters_timeline(
                st.session_state.df_clustered
            )
            st.plotly_chart(all_timeline_fig, use_container_width=True)

        with tab2:
            st.subheader("Analyze Individual Cluster")

            # Cluster selector
            available_clusters = sorted(st.session_state.df_clustered['cluster'].unique())

            selected_cluster = st.selectbox(
                "Select a cluster to analyze",
                options=available_clusters,
                format_func=lambda x: f"Cluster {x}"
            )

            # Get cluster info
            cluster_count = st.session_state.cluster_stats['cluster_counts'][selected_cluster]
            cluster_pct = st.session_state.cluster_stats['cluster_percentages'][selected_cluster]

            st.info(
                f"**Cluster {selected_cluster}**\n\n"
                f"Snapshots: {cluster_count} ({cluster_pct:.1f}%)"
            )

            st.markdown("---")

            # Get cluster traces
            frequencies, min_trace, max_trace, avg_trace = clustering.get_cluster_traces(
                st.session_state.df_clustered,
                selected_cluster
            )

            # Plot traces
            st.subheader("Min / Max / Average Spectrum Traces")

            colors = get_color_palette(len(available_clusters))
            cluster_color = colors[list(available_clusters).index(selected_cluster)]

            traces_fig = visualizations.plot_cluster_traces(
                frequencies,
                min_trace,
                max_trace,
                avg_trace,
                selected_cluster,
                color=cluster_color
            )
            st.plotly_chart(traces_fig, use_container_width=True)

            st.markdown("---")

            # Activity timeline
            st.subheader("Activity Timeline")

            timeline_fig = visualizations.plot_cluster_timeline(
                st.session_state.df_clustered,
                selected_cluster,
                color=cluster_color
            )
            st.plotly_chart(timeline_fig, use_container_width=True)

    else:
        # Show instructions
        st.info(
            """
            ### 👈 Getting Started

            1. Select a data file from the sidebar
            2. Adjust clustering parameters if desired
            3. Click **Run Clustering** to begin analysis

            The dashboard will:
            - Load spectrum data from the selected JSON file
            - Extract features from each spectrum snapshot
            - Perform BIRCH clustering
            - Display interactive visualizations
            """
        )

        # Show example dataset info if files available
        if available_files:
            st.markdown("### 📁 Available Datasets")

            for file in available_files[:5]:  # Show first 5
                info = data_loader.get_file_info(file)
                if info['exists']:
                    st.write(
                        f"- **{file.split('/')[-1]}** "
                        f"({info['size_mb']:.1f} MB, {info['num_records']} records)"
                    )


if __name__ == "__main__":
    main()
