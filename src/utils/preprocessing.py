from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
from typing import Tuple


class ECGPreprocessor:
    def __init__(self, sampling_rate: int = 100, target_length: int = 1000):
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.scaler = StandardScaler()

    def bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        order: int = 4,
    ) -> np.ndarray:
        if lowcut <= 0 or highcut <= 0:
            raise ValueError("lowcut and highcut must be positive values.")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive value.")
        if lowcut >= highcut:
            raise ValueError("lowcut must be less than highcut.")

        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        if not (0 < low < 1):
            raise ValueError(f"lowcut frequency {lowcut} is out of the Nyquist range.")
        if not (0 < high < 1):
            raise ValueError(
                f"highcut frequency {highcut} is out of the Nyquist range."
            )

        b, a = butter(order, [low, high], btype="band")
        filtered_data = np.array([filtfilt(b, a, lead, axis=0) for lead in data])
        return filtered_data

    def notch_filter(
        self,
        data: np.ndarray,
        notch_freq: float = 50.0,
        quality_factor: float = 30.0,
    ) -> np.ndarray:
        a, b = iirnotch(notch_freq, quality_factor, fs=self.sampling_rate)
        filtered_data = np.array([filtfilt(b, a, lead, axis=0) for lead in data])
        return filtered_data

    def normalize(self, data: np.ndarray) -> np.ndarray:
        original_shape = data.shape
        n_records, n_time, n_leads = original_shape
        data_reshaped = data.reshape(-1, n_leads)
        scaled_data_reshaped = self.scaler.fit_transform(data_reshaped)
        return scaled_data_reshaped.reshape(original_shape)

    def preprocess(
        self,
        data: np.ndarray,
        apply_filters: bool = True,
    ) -> np.ndarray:
        preproccessed_data = data.copy()

        if apply_filters:
            preproccessed_data = self.bandpass_filter(preproccessed_data)
            preproccessed_data = self.notch_filter(preproccessed_data)

        preproccessed_data = self.normalize(preproccessed_data)

        return preproccessed_data
