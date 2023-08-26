from abc import ABC
from typing import Any, Dict

import pandas as pd
from ts_segmentation.utils.label import Label


class LabelDataSet:
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> Label:
        data = pd.read_csv(self._filepath)
        return Label(**data.to_dict())

    def _save(self, data: Label) -> None:
        df = pd.DataFrame([data.dict()])
        df.to_csv(self._filepath, index=False)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)
