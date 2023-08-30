import time

import numpy as np
import statsmodels.api as sm
from pydantic import BaseModel


class isUserInLabel(BaseModel):
    is_user_in: bool


def detect_by_acf(
    x: np.ndarray,
    nlags: int = 5,
    ths_count: float = 0.1,
    ths_user_detect: float = 0.55,
) -> tuple[bool, float]:
    """自己相関係数を利用したユーザ検出

    Args:
        x (np.ndarray): 時系列データ
        nlags (int, optional): 自己相関係数のラグ数. Defaults to 5.
        ths_count (float, optional): 自己相関係数の閾値. Defaults to 0.1.
        ths_user_detect (float, optional): ユーザ検出の閾値. Defaults to 0.55.

    Returns:
        tuple[bool, float]: ユーザ検出の結果, ユーザ検出の結果の正規化値
    """
    acf_value: np.ndarray = sm.tsa.stattools.acf(x, nlags=nlags)
    acf_count = np.sum([1 if float(a) > ths_count else 0 for a in acf_value])
    norm_acf_count = acf_count / nlags
    is_user_in = norm_acf_count > ths_user_detect
    return is_user_in, norm_acf_count


def detect_by_pacf(
    x: np.ndarray,
    nlags: int = 5,
    ths_count: float = 0.1,
    ths_user_detect: float = 0.55,
):
    """偏自己相関係数を利用したユーザ検出

    Args:
        x (np.ndarray): 時系列データ
        nlags (int, optional): 自己相関係数のラグ数. Defaults to 5.
        ths_count (float, optional): 自己相関係数の閾値. Defaults to 0.1.
        ths_user_detect (float, optional): ユーザ検出の閾値. Defaults to 0.55.

    Returns:
        tuple[bool, float]: ユーザ検出の結果, ユーザ検出の結果の正規化値
    """
    pacf_value: np.ndarray = sm.tsa.stattools.pacf(x, nlags=nlags)
    pacf_count = np.sum([1 if float(a) > ths_count else 0 for a in pacf_value])
    norm_pacf_count = pacf_count / nlags
    is_user_in = norm_pacf_count > ths_user_detect
    return is_user_in, norm_pacf_count


def detect_by_std(
    x: np.ndarray,
    ths_std: float = 30.0,
):
    """標準偏差を利用したユーザ検出

    Args:
        x (np.ndarray): 時系列データ
        ths_std (float, optional): 標準偏差の閾値. Defaults to 30.0.

    Returns:
        tuple[bool, float]: ユーザ検出の結果, ユーザ検出の結果の正規化値
    """
    std_value = np.std(x)
    is_user_in = std_value > ths_std
    return is_user_in, std_value


def detect_by_anormaly_retio(
    x: np.ndarray,
    ths_normaly_range: float = 0.80,  # 20%までは誤差許容としmaxを評価
    normaly_value_max: float = 1.0,
    normaly_max_value_std: float = 0.001,
    ths_anormaly_ratio: float = 0.1,
):
    """異常値の割合を利用したユーザ検出

    Args:
        x (np.ndarray): scale化ずみの時系列データ
        ths_normaly_range (float, optional): 正常値の範囲の閾値. Defaults to 0.80.
        normaly_value_max (float, optional): 正常値の最大値. Defaults to 1.0.
        normaly_max_value_std (float, optional): 正常値の標準偏差. Defaults to 0.001.
        ths_anormaly_ratio (float, optional): 異常値の割合の閾値. Defaults to 0.1.

    Returns:
        tuple[bool, float]: ユーザ検出の結果, ユーザ検出の結果の正規化値
    """
    # 正常値の最大値を計算
    ths_max_value = (
        normaly_value_max * ths_normaly_range  # 基準となるMAX の値
        + normaly_value_max * normaly_max_value_std  # MAXの値に対してstdとして許容される誤差量
    )

    # 異常値の割合を計算
    # 0中心化処理をされている正弦波のため、 absをとり比較する
    anormaly_ratio = np.sum(
        [1 if abs(float(a)) > ths_max_value else 0 for a in x]
    ) / len(x)

    is_user_in = anormaly_ratio > ths_anormaly_ratio
    return is_user_in, anormaly_ratio


def detect_by_static_filter(
    x: np.ndarray,
    fs: int = 500,
    ths_static_filter: float = 0.1,
):
    """
    Detects whether a user is present in the input signal using a static filter.

    Args:
        x (np.ndarray): The input signal.
        fs (int, optional): The sampling frequency of the input signal. Defaults to 500.
        ths_static_filter (float, optional): The threshold for the static filter. Defaults to 0.1.

    Returns:
        Tuple[bool, float]: A tuple containing a boolean indicating whether a user is present in the input signal,
        and the square value of the heart beat wave.
    """
    from ts_segmentation.utils.vibrations.waveUtility import (
        extract_pp_wave,
        square_of_heart_beat,
    )

    heart_beat_wave = extract_pp_wave(x, fs=fs)
    square_value = square_of_heart_beat(heart_beat_wave)
    is_user_in = square_value > ths_static_filter
    return is_user_in, square_value


def detect_by_fft_ratio(
    x: np.ndarray,
    fs: int = 500,
    peak_order: int = 1,
    heart_freq_range: tuple[float, float] = (0.8, 4.0),
    ths_fft_amp: float = 0.1,
    ths_fft_ratio: float = 0.1,
):
    # signalに対してfftを実行
    fft_x = np.fft.fft(x)
    fft_x_abs = np.abs(fft_x)
    # fft amp の計算
    fft_x_abs_amp = fft_x_abs / len(fft_x_abs) * 2

    # fft amp に含まれるすべての周波数ピークの取得
    from scipy import signal

    fft_x_abs_amp_peaks = signal.argrelmax(fft_x_abs_amp, order=peak_order)

    # fft peak id を freq に変換しampでソート
    freq_peaks = [
        {"freq": p / len(fft_x_abs_amp) * fs, "value": fft_x_abs_amp[p]}
        for p in fft_x_abs_amp_peaks[0]
    ]
    freq_peaks = sorted(freq_peaks, key=lambda x: x["value"], reverse=True)

    # 心拍数の周波数帯域のピークのみを抽出
    heart_freq_peaks = [
        p["value"]
        for p in freq_peaks
        if heart_freq_range[0] <= p["freq"] <= heart_freq_range[1]
    ]
    if len(heart_freq_peaks) > 0:
        # 検出された心拍周波数帯域のピークが閾値を超えている割合を計算
        heart_freq_peaks_ratio = sum(
            [int(p > ths_fft_amp) for p in heart_freq_peaks]
        ) / len(heart_freq_peaks)
        is_user_in = heart_freq_peaks_ratio > ths_fft_ratio
    else:
        heart_freq_peaks_ratio = 0.0
        is_user_in = False
    return is_user_in, heart_freq_peaks_ratio


def batch_detect_test(x: np.ndarray):
    # apply all detect_by_* functions
    # return list dict results
    # [{
    #   "name":"function_name",
    #   "is_user_in":bool,
    #   "value":float,
    #   "time":float
    # }]
    results = []
    start_time = time.time()
    acf_user_in, acf_value = detect_by_acf(x)
    end_time = time.time()
    results.append(
        {
            "name": "detect_by_acf",
            "is_user_in": acf_user_in,
            "value": acf_value,
            "time": end_time - start_time,
        }
    )

    start_time = time.time()
    pacf_user_in, pacf_value = detect_by_pacf(x)
    end_time = time.time()
    results.append(
        {
            "name": "detect_by_pacf",
            "is_user_in": pacf_user_in,
            "value": pacf_value,
            "time": end_time - start_time,
        }
    )

    start_time = time.time()
    std_user_in, std_value = detect_by_std(x)
    end_time = time.time()
    results.append(
        {
            "name": "detect_by_std",
            "is_user_in": std_user_in,
            "value": std_value,
            "time": end_time - start_time,
        }
    )

    start_time = time.time()
    anormaly_user_in, anormaly_value = detect_by_anormaly_retio(x)
    end_time = time.time()
    results.append(
        {
            "name": "detect_by_anormaly_retio",
            "is_user_in": anormaly_user_in,
            "value": anormaly_value,
            "time": end_time - start_time,
        }
    )

    start_time = time.time()
    static_user_in, static_value = detect_by_static_filter(x)
    end_time = time.time()
    results.append(
        {
            "name": "detect_by_static_filter",
            "is_user_in": static_user_in,
            "value": static_value,
            "time": end_time - start_time,
        }
    )

    start_time = time.time()
    fft_user_in, fft_value = detect_by_fft_ratio(x)
    end_time = time.time()
    results.append(
        {
            "name": "detect_by_fft_ratio",
            "is_user_in": fft_user_in,
            "value": fft_value,
            "time": end_time - start_time,
        }
    )

    return results


if __name__ == "__main__":
    # test
    noise = np.random.normal(0, 1, 500)

    test_freqs = [0.5, 1, 3, 5, 10, 30]

    for freq in test_freqs:
        # 正弦波を生成
        random_sin_data = np.sin(np.linspace(0, freq * 2 * np.pi, 500))
        random_sin_data = (random_sin_data + noise) / 2

        results = batch_detect_test(random_sin_data)
        print(f"{freq}Hz sin wave")
        for r in results:
            print("   ", r)

    # multi sin wave
    multi_sin_data = np.sin(np.linspace(0, 0.5 * 2 * np.pi, 500)) + np.sin(
        np.linspace(0, 4 * 2 * np.pi, 500)
    )
    multi_sin_data = (multi_sin_data + noise) / 2
    results = batch_detect_test(multi_sin_data)
    print(f"multi sin wave 0.5Hz + 4Hz")
    for r in results:
        print("   ", r)
