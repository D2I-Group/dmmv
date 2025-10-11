import numpy as np
import torch


def find_periods(x: np.ndarray, top_k: int = 2, min_period: int = 2) -> np.ndarray:
    """Find periodic patterns in time series data using FFT analysis

    Args:
        x: Input time series data
        top_k: Number of main frequencies to identify
        min_period: Minimum period threshold, periods below this will be filtered out

    Returns:
        np.ndarray: Array of top k periods found in the data that are >= min_period
    """
    # FFT analysis
    freqs = np.fft.rfftfreq(len(x))
    fft_vals = np.fft.rfft(x)
    amplitudes = np.abs(fft_vals)
    amplitudes[0] = 0  # Ignore DC component

    # Calculate periods and filter out those below threshold
    periods = (1 / freqs[1:]).astype(int)  # Skip first freq (DC)
    valid_mask = periods >= min_period
    valid_amplitudes = amplitudes[1:][valid_mask]
    valid_periods = periods[valid_mask]

    # Find top k peak frequencies among valid periods
    if len(valid_periods) == 0:
        return np.array([])

    top_k = min(top_k, len(valid_periods))
    top_indices = np.argsort(valid_amplitudes)[-top_k:]
    top_periods = valid_periods[top_indices]

    return top_periods


def find_periods_multi_variable(x: torch.Tensor, top_k: int = 2) -> torch.Tensor:
    """Find periodic patterns in multivariate time series data using FFT analysis

    Args:
        x: Input multivariate time series data of shape [num_variables, sequence_length]
        top_k: Number of main frequencies to identify per variable

    Returns:
        torch.Tensor: Single period value that best represents all variables
    """
    device = x.device
    x_np = x.cpu().numpy()
    seq_len = x_np.shape[1]

    all_periods = np.stack(
        [find_periods(x_np[i], top_k=top_k) for i in range(x_np.shape[0])]
    )
    # all_periods shape: [num_vars, top_k]
    flat_periods = all_periods.flatten()  # [num_vars * top_k]

    # Filter out periods equal to sequence length
    valid_periods = flat_periods[flat_periods != seq_len]
    if len(valid_periods) == 0:
        return torch.tensor(
            seq_len // 2, device=device
        )  # Fallback to half sequence length

    unique_periods, counts = np.unique(valid_periods, return_counts=True)
    max_count = counts.max()
    most_common_periods = unique_periods[counts == max_count]
    best_period = int(np.median(most_common_periods))

    return torch.tensor(best_period, device=device)
