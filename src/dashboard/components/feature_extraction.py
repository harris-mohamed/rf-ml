"""Feature extraction for RF spectrum data."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.signal import find_peaks
from scipy.stats import entropy as scipy_entropy, skew, kurtosis
from scipy.fft import fft
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    include_statistical: bool = True
    include_shape: bool = True
    include_rf_domain: bool = True
    include_model: bool = True


class FeatureExtractor(ABC):
    """Base class for feature extractors."""

    @abstractmethod
    def extract(self, frequencies: np.ndarray, powers: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a single spectrum.

        Args:
            frequencies: Frequency array (Hz)
            powers: Power array (dBm/Hz)

        Returns:
            Dictionary of feature name to value
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces."""
        pass

    @abstractmethod
    def get_category(self) -> str:
        """Get feature category name."""
        pass


class StatisticalFeatureExtractor(FeatureExtractor):
    """Extract statistical features from spectrum."""

    def extract(self, frequencies: np.ndarray, powers: np.ndarray) -> Dict[str, float]:
        """Extract 10 statistical features."""
        features = {}

        # Basic statistics
        features['mean_power'] = np.mean(powers)
        features['max_power'] = np.max(powers)
        features['min_power'] = np.min(powers)
        features['std_power'] = np.std(powers)
        features['median_power'] = np.median(powers)
        features['range_power'] = np.ptp(powers)  # peak-to-peak

        # Higher moments
        features['skewness'] = skew(powers)
        features['kurtosis'] = kurtosis(powers)

        # Quantiles
        features['q25_power'] = np.percentile(powers, 25)
        features['q75_power'] = np.percentile(powers, 75)

        return features

    def get_feature_names(self) -> List[str]:
        return [
            'mean_power', 'max_power', 'min_power', 'std_power', 'median_power',
            'range_power', 'skewness', 'kurtosis', 'q25_power', 'q75_power'
        ]

    def get_category(self) -> str:
        return "Statistical"


class ShapeFeatureExtractor(FeatureExtractor):
    """Extract shape-based features from spectrum."""

    def __init__(self, peak_prominence: float = 5.0):
        """
        Initialize shape feature extractor.

        Args:
            peak_prominence: Prominence threshold for peak detection
        """
        self.peak_prominence = peak_prominence

    def extract(self, frequencies: np.ndarray, powers: np.ndarray) -> Dict[str, float]:
        """Extract 12 shape-based features."""
        features = {}

        # Peak detection
        peaks, properties = find_peaks(powers, prominence=self.peak_prominence)
        features['num_peaks'] = len(peaks)

        if len(peaks) > 0:
            # Peak characteristics
            features['avg_peak_height'] = np.mean(powers[peaks])
            features['max_peak_height'] = np.max(powers[peaks])
            features['avg_peak_prominence'] = np.mean(properties['prominences'])

            # Peak width (full width at half maximum approximation)
            widths = []
            for peak_idx in peaks:
                half_max = powers[peak_idx] / 2
                # Find points where power crosses half maximum
                left_idx = peak_idx
                while left_idx > 0 and powers[left_idx] > half_max:
                    left_idx -= 1
                right_idx = peak_idx
                while right_idx < len(powers) - 1 and powers[right_idx] > half_max:
                    right_idx += 1
                width = frequencies[right_idx] - frequencies[left_idx]
                widths.append(width)

            features['avg_peak_width'] = np.mean(widths) if widths else 0.0
            features['max_peak_width'] = np.max(widths) if widths else 0.0

            # Peak spacing
            if len(peaks) > 1:
                spacings = np.diff(frequencies[peaks])
                features['avg_peak_spacing'] = np.mean(spacings)
                features['std_peak_spacing'] = np.std(spacings)
            else:
                features['avg_peak_spacing'] = 0.0
                features['std_peak_spacing'] = 0.0
        else:
            # No peaks found
            features['avg_peak_height'] = 0.0
            features['max_peak_height'] = 0.0
            features['avg_peak_prominence'] = 0.0
            features['avg_peak_width'] = 0.0
            features['max_peak_width'] = 0.0
            features['avg_peak_spacing'] = 0.0
            features['std_peak_spacing'] = 0.0

        # Area under curve (trapezoidal integration)
        features['auc'] = np.trapz(powers, frequencies)

        # Curve slopes (first derivative statistics)
        slopes = np.diff(powers) / np.diff(frequencies)
        features['avg_abs_slope'] = np.mean(np.abs(slopes))
        features['max_abs_slope'] = np.max(np.abs(slopes))

        # Inflection points (second derivative zero crossings)
        second_derivative = np.diff(slopes)
        sign_changes = np.diff(np.sign(second_derivative))
        features['num_inflection_points'] = np.sum(np.abs(sign_changes) > 0)

        return features

    def get_feature_names(self) -> List[str]:
        return [
            'num_peaks', 'avg_peak_height', 'max_peak_height', 'avg_peak_prominence',
            'avg_peak_width', 'max_peak_width', 'avg_peak_spacing', 'std_peak_spacing',
            'auc', 'avg_abs_slope', 'max_abs_slope', 'num_inflection_points'
        ]

    def get_category(self) -> str:
        return "Shape"


class RFDomainFeatureExtractor(FeatureExtractor):
    """Extract RF domain-specific features."""

    def __init__(self, noise_floor_percentile: float = 10.0):
        """
        Initialize RF domain feature extractor.

        Args:
            noise_floor_percentile: Percentile to use as noise floor estimate
        """
        self.noise_floor_percentile = noise_floor_percentile

    def extract(self, frequencies: np.ndarray, powers: np.ndarray) -> Dict[str, float]:
        """Extract 10 RF domain-specific features."""
        features = {}

        # Estimate noise floor
        noise_floor = np.percentile(powers, self.noise_floor_percentile)

        # Occupied bandwidth (3dB, 10dB, 20dB below max)
        max_power = np.max(powers)
        for db_level in [3, 10, 20]:
            threshold = max_power - db_level
            above_threshold = frequencies[powers >= threshold]
            if len(above_threshold) > 0:
                occupied_bw = above_threshold[-1] - above_threshold[0]
                features[f'occupied_bw_{db_level}db'] = occupied_bw / 1e6  # MHz
            else:
                features[f'occupied_bw_{db_level}db'] = 0.0

        # Peak-to-average power ratio (PAPR)
        avg_power_linear = np.mean(10 ** (powers / 10))  # Convert to linear
        max_power_linear = 10 ** (max_power / 10)
        if avg_power_linear > 0:
            features['papr'] = 10 * np.log10(max_power_linear / avg_power_linear)
        else:
            features['papr'] = 0.0

        # Spectral flatness (Wiener entropy)
        # Ratio of geometric mean to arithmetic mean
        powers_linear = 10 ** (powers / 10)
        powers_linear_pos = powers_linear[powers_linear > 0]
        if len(powers_linear_pos) > 0:
            geometric_mean = np.exp(np.mean(np.log(powers_linear_pos)))
            arithmetic_mean = np.mean(powers_linear_pos)
            if arithmetic_mean > 0:
                features['spectral_flatness'] = geometric_mean / arithmetic_mean
            else:
                features['spectral_flatness'] = 0.0
        else:
            features['spectral_flatness'] = 0.0

        # Spectral centroid (frequency-weighted average)
        total_power = np.sum(powers_linear)
        if total_power > 0:
            features['spectral_centroid'] = np.sum(frequencies * powers_linear) / total_power / 1e9  # GHz
        else:
            features['spectral_centroid'] = np.mean(frequencies) / 1e9

        # Roll-off frequency (frequency below which 90%, 95%, 99% of energy)
        cumulative_power = np.cumsum(powers_linear)
        total_energy = cumulative_power[-1]
        for percent in [90, 95, 99]:
            threshold_energy = (percent / 100) * total_energy
            rolloff_idx = np.searchsorted(cumulative_power, threshold_energy)
            if rolloff_idx < len(frequencies):
                features[f'rolloff_{percent}pct'] = frequencies[rolloff_idx] / 1e9  # GHz
            else:
                features[f'rolloff_{percent}pct'] = frequencies[-1] / 1e9

        # Power above noise floor
        features['power_above_noise'] = np.sum(powers > (noise_floor + 3))  # 3dB above noise

        return features

    def get_feature_names(self) -> List[str]:
        return [
            'occupied_bw_3db', 'occupied_bw_10db', 'occupied_bw_20db',
            'papr', 'spectral_flatness', 'spectral_centroid',
            'rolloff_90pct', 'rolloff_95pct', 'rolloff_99pct',
            'power_above_noise'
        ]

    def get_category(self) -> str:
        return "RF Domain"


class ModelFeatureExtractor(FeatureExtractor):
    """Extract model-based features (entropy, FFT, autocorrelation, derivatives)."""

    def __init__(self, n_fft_coeffs: int = 5):
        """
        Initialize model feature extractor.

        Args:
            n_fft_coeffs: Number of FFT coefficients to include as features
        """
        self.n_fft_coeffs = n_fft_coeffs

    def extract(self, frequencies: np.ndarray, powers: np.ndarray) -> Dict[str, float]:
        """Extract 12+ model-based features."""
        features = {}

        # Sample entropy (probability-based)
        # Normalize powers to probabilities
        powers_linear = 10 ** (powers / 10)
        if np.sum(powers_linear) > 0:
            probs = powers_linear / np.sum(powers_linear)
            features['sample_entropy'] = scipy_entropy(probs)
        else:
            features['sample_entropy'] = 0.0

        # Permutation entropy (order-based complexity)
        # Simplified version: count monotonic runs
        diffs = np.diff(powers)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
        features['permutation_entropy'] = sign_changes / max(len(powers) - 2, 1)

        # FFT coefficients (frequency domain representation)
        fft_coeffs = np.abs(fft(powers))
        for i in range(min(self.n_fft_coeffs, len(fft_coeffs) // 2)):
            features[f'fft_coeff_{i}'] = fft_coeffs[i + 1]  # Skip DC component

        # Autocorrelation at different lags
        powers_centered = powers - np.mean(powers)
        for lag in [1, 5, 10]:
            if lag < len(powers):
                autocorr = np.correlate(
                    powers_centered[:-lag],
                    powers_centered[lag:],
                    mode='valid'
                )[0]
                norm_factor = np.sum(powers_centered ** 2)
                if norm_factor > 0:
                    features[f'autocorr_lag_{lag}'] = autocorr / norm_factor
                else:
                    features[f'autocorr_lag_{lag}'] = 0.0
            else:
                features[f'autocorr_lag_{lag}'] = 0.0

        # First derivative statistics
        first_deriv = np.diff(powers) / np.diff(frequencies)
        features['deriv1_mean'] = np.mean(first_deriv)
        features['deriv1_std'] = np.std(first_deriv)
        features['deriv1_max'] = np.max(np.abs(first_deriv))

        # Second derivative statistics
        second_deriv = np.diff(first_deriv) / np.diff(frequencies[:-1])
        features['deriv2_mean'] = np.mean(second_deriv)
        features['deriv2_std'] = np.std(second_deriv)

        return features

    def get_feature_names(self) -> List[str]:
        names = [
            'sample_entropy', 'permutation_entropy',
        ]
        # Add FFT coefficient names
        for i in range(self.n_fft_coeffs):
            names.append(f'fft_coeff_{i}')
        # Add autocorrelation names
        names.extend(['autocorr_lag_1', 'autocorr_lag_5', 'autocorr_lag_10'])
        # Add derivative names
        names.extend(['deriv1_mean', 'deriv1_std', 'deriv1_max', 'deriv2_mean', 'deriv2_std'])
        return names

    def get_category(self) -> str:
        return "Model"


class CompositeFeatureExtractor:
    """Combines multiple feature extractors based on configuration."""

    def __init__(self, config: FeatureConfig = None):
        """
        Initialize composite feature extractor.

        Args:
            config: FeatureConfig specifying which feature categories to include
        """
        if config is None:
            config = FeatureConfig()

        self.config = config
        self.extractors = []

        if config.include_statistical:
            self.extractors.append(StatisticalFeatureExtractor())

        if config.include_shape:
            self.extractors.append(ShapeFeatureExtractor())

        if config.include_rf_domain:
            self.extractors.append(RFDomainFeatureExtractor())

        if config.include_model:
            self.extractors.append(ModelFeatureExtractor())

    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from all spectra in dataframe.

        Args:
            df: DataFrame with 'frequencies' and 'powers' columns

        Returns:
            Tuple of (feature_array, feature_names)
            - feature_array: Array of shape (n_samples, n_features)
            - feature_names: List of feature names
        """
        all_features = []
        feature_names = []

        # Get feature names from extractors
        for extractor in self.extractors:
            feature_names.extend(extractor.get_feature_names())

        # Extract features for each spectrum
        for idx, row in df.iterrows():
            frequencies = row['frequencies']
            powers = row['powers']

            spectrum_features = []
            for extractor in self.extractors:
                feature_dict = extractor.extract(frequencies, powers)
                # Maintain order from get_feature_names()
                for name in extractor.get_feature_names():
                    spectrum_features.append(feature_dict[name])

            all_features.append(spectrum_features)

        feature_array = np.array(all_features)
        return feature_array, feature_names

    def get_feature_info(self) -> Dict[str, List[str]]:
        """
        Get feature information organized by category.

        Returns:
            Dict mapping category name to list of feature names
        """
        info = {}
        for extractor in self.extractors:
            category = extractor.get_category()
            names = extractor.get_feature_names()
            info[category] = names
        return info

    def get_total_feature_count(self) -> int:
        """Get total number of features."""
        return sum(len(e.get_feature_names()) for e in self.extractors)


# Legacy compatibility function (matches old interface)
def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract features using legacy statistical feature set (8 features).

    This function maintains backward compatibility with the original
    clustering.py implementation.

    Args:
        df: DataFrame from data_loader.load_spectrum_data()

    Returns:
        Feature array of shape (num_samples, 8)
    """
    features_list = []

    for idx, row in df.iterrows():
        frequencies = row['frequencies']
        powers = row['powers']

        # Basic statistics
        mean_power = np.mean(powers)
        max_power = np.max(powers)
        std_power = np.std(powers)
        median_power = np.median(powers)

        # Peak detection
        peaks, _ = find_peaks(powers, prominence=5)
        num_peaks = len(peaks)

        # Spectral centroid
        total_power = np.sum(powers)
        if total_power > 0:
            spectral_centroid = np.sum(frequencies * powers) / total_power
        else:
            spectral_centroid = np.mean(frequencies)

        # Power above threshold
        threshold = mean_power + std_power
        power_above_threshold = np.sum(powers > threshold)

        # Low frequency ratio
        mid_freq = (frequencies[0] + frequencies[-1]) / 2
        low_freq_mask = frequencies < mid_freq
        high_freq_mask = frequencies >= mid_freq

        low_freq_energy = np.sum(powers[low_freq_mask])
        high_freq_energy = np.sum(powers[high_freq_mask])
        total_energy = low_freq_energy + high_freq_energy

        if total_energy > 0:
            low_freq_ratio = low_freq_energy / total_energy
        else:
            low_freq_ratio = 0.5

        features_list.append([
            mean_power,
            max_power,
            std_power,
            median_power,
            num_peaks,
            spectral_centroid,
            power_above_threshold,
            low_freq_ratio
        ])

    return np.array(features_list)


# Legacy feature names
LEGACY_FEATURE_NAMES = [
    'Mean Power',
    'Max Power',
    'Std Power',
    'Median Power',
    'Num Peaks',
    'Spectral Centroid',
    'Power Above Threshold',
    'Low Freq Ratio'
]
