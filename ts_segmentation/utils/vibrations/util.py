import datetime

import numpy as np


def csv_data_timestamps2millisec(t: np.float64) -> str:
    date_t = datetime.datetime.fromtimestamp(t)
    date_t = date_t.replace(microsecond=int((t - int(t)) * 1000000))
    return date_t.isoformat(timespec="milliseconds")
