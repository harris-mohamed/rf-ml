# rf-ml
A repository to play around with synthetic RF ML approaches

## Environment Setup

This project uses conda for virtual environment management. Create and activate the environment with:

```bash
conda create -n rf-ml python=3.x
conda activate rf-ml
```

## Directory Structure

```
data/
├── raw/              # Original synthetic RF signals
├── processed/        # Preprocessed/cleaned data
└── splits/           # Train/val/test splits

models/
├── checkpoints/      # Saved model weights
└── configs/          # Model configuration files

notebooks/            # Jupyter notebooks for exploration

src/
├── data_generation/  # Synthetic RF signal generators
├── models/           # Model architectures
├── training/         # Training scripts
└── visualization/    # Plotting and analysis tools

outputs/
├── figures/          # Generated visualizations
└── results/          # Metrics, logs, reports

tests/                # Unit and integration tests
```

## Workflow

1. **Data Generation**: Create synthetic RF signals in `src/data_generation/`, save to `data/raw/`
2. **Preprocessing**: Process and split data, save to `data/processed/` and `data/splits/`
3. **Model Development**: Define architectures in `src/models/`, training logic in `src/training/`
4. **Training**: Train models, save checkpoints to `models/checkpoints/`
5. **Visualization**: Generate plots with `src/visualization/`, save to `outputs/figures/`

## Interactive Dashboard

An interactive Streamlit dashboard is available for visualizing and analyzing RF spectrum clustering results.

### Running the Dashboard

```bash
# Activate virtual environment
source venv/bin/activate  # or: conda activate rf-ml

# Run the dashboard
streamlit run src/dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Dashboard Features

- **Data Loading**: Load existing spectrum JSON files from `dev_output/` directory
- **BIRCH Clustering**: Perform unsupervised clustering with configurable parameters
- **Cluster Overview Tab**:
  - Pie chart showing cluster distribution percentages
  - Interactive waterfall plot of spectrum data
  - Timeline showing all cluster activities
- **Cluster Details Tab**:
  - Select individual clusters for detailed analysis
  - Min/Max/Average spectrum traces for each cluster
  - Activity timeline showing when cluster appears

### Dashboard Structure

```
src/dashboard/
├── app.py                      # Main Streamlit application
├── components/
│   ├── data_loader.py          # Load and process spectrum data
│   ├── clustering.py           # BIRCH clustering logic
│   └── visualizations.py       # Plotly chart generation
└── utils/
    └── helpers.py              # Utility functions
```
