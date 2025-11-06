"""Normalization strategies for RF spectrum data."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class NormalizationStrategy(ABC):
    """Base class for normalization strategies."""

    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize spectrum data.

        Args:
            df: DataFrame with 'powers' and 'frequencies' columns

        Returns:
            Tuple of (normalized_df, normalization_params)
            - normalized_df: DataFrame with normalized 'powers' column
            - normalization_params: Dict of parameters for inverse transform
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of normalization method."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get description of normalization method."""
        pass


class NoNormalization(NormalizationStrategy):
    """No normalization - use raw spectrum data."""

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Return data unchanged."""
        return df.copy(), {}

    def get_name(self) -> str:
        return "None"

    def get_description(self) -> str:
        return "No normalization - use raw spectrum data"


class EnergyNormalization(NormalizationStrategy):
    """Normalize by total energy (L2 norm)."""

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize each spectrum by its total energy.

        Divides by sqrt(sum of squared powers), making all spectra
        unit vectors in power space.
        """
        df_norm = df.copy()
        energies = []

        for idx, row in df_norm.iterrows():
            powers = row['powers']
            energy = np.sqrt(np.sum(powers ** 2))

            if energy > 0:
                df_norm.at[idx, 'powers'] = powers / energy
            else:
                # Keep as-is if energy is zero
                pass

            energies.append(energy)

        params = {
            'energies': np.array(energies),
            'method': 'energy'
        }

        return df_norm, params

    def get_name(self) -> str:
        return "Energy"

    def get_description(self) -> str:
        return "Normalize by total energy (L2 norm) - makes all spectra unit vectors"


class RobustScaling(NormalizationStrategy):
    """Robust scaling using median and IQR (resistant to outliers)."""

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scale each spectrum using median centering and IQR scaling.

        This is resistant to outliers/interferers compared to z-score.
        Transforms to: (x - median) / IQR
        """
        df_norm = df.copy()
        medians = []
        iqrs = []

        for idx, row in df_norm.iterrows():
            powers = row['powers']
            median = np.median(powers)
            q25 = np.percentile(powers, 25)
            q75 = np.percentile(powers, 75)
            iqr = q75 - q25

            if iqr > 0:
                df_norm.at[idx, 'powers'] = (powers - median) / iqr
            else:
                # Just center if IQR is zero
                df_norm.at[idx, 'powers'] = powers - median

            medians.append(median)
            iqrs.append(iqr)

        params = {
            'medians': np.array(medians),
            'iqrs': np.array(iqrs),
            'method': 'robust'
        }

        return df_norm, params

    def get_name(self) -> str:
        return "Robust"

    def get_description(self) -> str:
        return "Median centering and IQR scaling - resistant to outliers"


class MinMaxScaling(NormalizationStrategy):
    """Min-max scaling to [0, 1] range."""

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scale each spectrum to [0, 1] range.

        Transforms to: (x - min) / (max - min)
        """
        df_norm = df.copy()
        mins = []
        maxs = []

        for idx, row in df_norm.iterrows():
            powers = row['powers']
            min_val = np.min(powers)
            max_val = np.max(powers)
            range_val = max_val - min_val

            if range_val > 0:
                df_norm.at[idx, 'powers'] = (powers - min_val) / range_val
            else:
                # All values same - set to 0.5
                df_norm.at[idx, 'powers'] = np.full_like(powers, 0.5)

            mins.append(min_val)
            maxs.append(max_val)

        params = {
            'mins': np.array(mins),
            'maxs': np.array(maxs),
            'method': 'minmax'
        }

        return df_norm, params

    def get_name(self) -> str:
        return "Min-Max"

    def get_description(self) -> str:
        return "Scale to [0, 1] range using min-max normalization"


class ZScoreNormalization(NormalizationStrategy):
    """Z-score normalization (standardization)."""

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Standardize each spectrum using z-score.

        Transforms to: (x - mean) / std
        """
        df_norm = df.copy()
        means = []
        stds = []

        for idx, row in df_norm.iterrows():
            powers = row['powers']
            mean = np.mean(powers)
            std = np.std(powers)

            if std > 0:
                df_norm.at[idx, 'powers'] = (powers - mean) / std
            else:
                # Just center if std is zero
                df_norm.at[idx, 'powers'] = powers - mean

            means.append(mean)
            stds.append(std)

        params = {
            'means': np.array(means),
            'stds': np.array(stds),
            'method': 'zscore'
        }

        return df_norm, params

    def get_name(self) -> str:
        return "Z-Score"

    def get_description(self) -> str:
        return "Standardization using mean and standard deviation"


# Factory function
def get_normalization_strategy(method: str) -> NormalizationStrategy:
    """
    Get normalization strategy by name.

    Args:
        method: Name of normalization method
                ('none', 'energy', 'robust', 'minmax', 'zscore')

    Returns:
        NormalizationStrategy instance
    """
    strategies = {
        'none': NoNormalization,
        'energy': EnergyNormalization,
        'robust': RobustScaling,
        'minmax': MinMaxScaling,
        'zscore': ZScoreNormalization
    }

    method_lower = method.lower()
    if method_lower not in strategies:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Available methods: {list(strategies.keys())}"
        )

    return strategies[method_lower]()


def get_available_normalizations() -> Dict[str, str]:
    """
    Get dict of available normalization methods.

    Returns:
        Dict mapping method name to description
    """
    methods = ['none', 'energy', 'robust', 'minmax', 'zscore']
    return {
        method: get_normalization_strategy(method).get_description()
        for method in methods
    }
