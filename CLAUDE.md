# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is for experimenting with synthetic RF (Radio Frequency) machine learning approaches. It's currently in early development stages.

## Environment Setup

This project uses conda for virtual environment management:

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
