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
    filters: list[dict[str, str | float]] = [
        {
            "name": "bress",
            "high": 0.5,
            "low": 0.05,
        },
        {
            "name": "heart_beat",
            "high": 0.5,
            "low": 0.05,
        },
    ],
    ths_static_filter: float = 0.1,
):
    pass
