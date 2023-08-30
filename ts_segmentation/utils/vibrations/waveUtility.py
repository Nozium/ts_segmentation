import numpy as np
from scipy import signal

__all__ = [
    "series2wave",
    "lowpass",
    "bandpass",
    "zscore",
    "partial_zscore",
    "mutual_information",
]


def series2wave(series, offset=0, Hz=200, start=0):
    """
    Convert a time series to a wave.

    Args:
        series (list): The time series to be converted.
        offset (float): The offset value to be subtracted from the series. Default is 0.
        Hz (int): The sampling frequency of the series. Default is 200.
        start (float): The start time of the wave. Default is 0.

    Returns:
        list: A list containing two arrays, the time array and the amplitude array.
    """
    ampl = np.array(series) - offset
    time = np.arange(start, start + len(ampl) / Hz, 1 / Hz)  # time[sec]
    return [time, ampl]


def lowpass(input_wave, cutoff, numtaps=255, fs=200):
    """
    Apply a low-pass filter to a wave.

    Args:
        input_wave (list): The wave to be filtered.
        cutoff (float): The cutoff frequency of the filter.
        numtaps (int): The number of filter coefficients. Default is 255.
        fs (int): The sampling frequency of the wave. Default is 200.

    Returns:
        list: The filtered wave.
    """
    nyq = fs / 2.0  # ナイキスト周波数
    # フィルタの設計
    cutoff = cutoff / nyq  # ナイキスト周波数が1になるように正規化
    numtaps = 255  # フィルタ係数（タップ）の数（要奇数）
    b = signal.firwin(numtaps, cutoff, pass_zero=True)  # Band-pass
    output_wave = signal.lfilter(b, 1, input_wave)
    return output_wave


def bandpass(input_wave, fe1, fe2, numtaps=255, fs=200):
    """
    Apply a band-pass filter to a wave.

    Args:
        input_wave (list): The wave to be filtered.
        fe1 (float): The lower cutoff frequency of the filter.
        fe2 (float): The upper cutoff frequency of the filter.
        numtaps (int): The number of filter coefficients. Default is 255.
        fs (int): The sampling frequency of the wave. Default is 200.

    Returns:
        list: The filtered wave.
    """
    nyq = fs / 2.0  # ナイキスト周波数
    # フィルタの設計
    fe1, fe2 = fe1 / nyq, fe2 / nyq  # ナイキスト周波数が1になるように正規化
    numtaps = 255  # フィルタ係数（タップ）の数（要奇数）
    b = signal.firwin(numtaps, [fe1, fe2], pass_zero=False)  # Band-pass
    output_wave = signal.lfilter(b, 1, input_wave)
    return output_wave


def zscore(x, axis=None):
    """
    Calculate the z-score of an array.

    Args:
        x (array): The array to be standardized.
        axis (int): The axis along which to calculate the mean and standard deviation. Default is None.

    Returns:
        array: The standardized array.
    """
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x - xmean) / xstd
    return zscore


def partial_zscore(x, partial_start, partial_end, axis=None):
    """
    Calculate the z-score of an array using a partial mean and standard deviation.

    Args:
        x (array): The array to be standardized.
        partial_start (int): The starting index of the partial mean and standard deviation.
        partial_end (int): The ending index of the partial mean and standard deviation.
        axis (int): The axis along which to calculate the mean and standard deviation. Default is None.

    Returns:
        array: The standardized array.
    """
    xmean = np.mean(x[partial_start:partial_end], axis=axis, keepdims=True)
    xstd = np.std(x[partial_start:partial_end], axis=axis, keepdims=True)
    zscore = (x - xmean) / xstd
    return zscore


def mutual_information(X, Y, bins=10):
    """
    Calculate the mutual information between two arrays.

    Args:
        X (array): The first array.
        Y (array): The second array.
        bins (int): The number of bins to use for the histogram. Default is 10.

    Returns:
        tuple: A tuple containing the mutual information value, the joint probability distribution, and the product of the marginal probability distributions.
    """
    # 同時確率分布p(x,y)の計算
    p_xy, xedges, yedges = np.histogram2d(X, Y, bins=bins, density=True)

    # p(x)p(y)の計算
    p_x, _ = np.histogram(X, bins=xedges, density=True)
    p_y, _ = np.histogram(Y, bins=yedges, density=True)
    p_x_y = p_x[:, np.newaxis] * p_y

    # dx と dy
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    # 積分の要素
    elem = p_xy * np.ma.log(p_xy / p_x_y)

    # print(elem,dx,dy)
    return np.sum(elem * dx * dy), p_xy, p_x_y
