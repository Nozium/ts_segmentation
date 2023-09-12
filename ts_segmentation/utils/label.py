from datetime import datetime
from enum import Enum
from typing import List

from pydantic import UUID4, BaseModel


def convert_keys_to_underscore(data):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            new_key = key.replace(" ", "_")
            new_data[new_key] = convert_keys_to_underscore(value)
        return new_data
    elif isinstance(data, list):
        return [convert_keys_to_underscore(item) for item in data]
    else:
        return data


class DataType(str, Enum):
    Range = "Range"
    Classification = "Classification"
    Landmark = "Landmark"
    Segmentation = "Segmentation"


class DetailAttributes(BaseModel):
    pitch_black: bool
    angle_of_view_anomaly: bool
    no_person: bool


class DetailData(BaseModel):
    _type: DataType


class Detail(BaseModel):
    label: str
    annotation_id: UUID4
    data: DetailData
    attributes: DetailAttributes


class TaskPhase(str, Enum):
    planning = "planning"
    development = "development"
    acceptance = "acceptance"
    testing = "testing"
    deployment = "deployment"
    maintenance = "maintenance"
    review = "review"
    completed = "completed"


class TaskStatus(str, Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    on_hold = "on_hold"
    complete = "complete"
    under_review = "under_review"
    rejected = "rejected"
    accepted = "accepted"
    testing = "testing"
    deployed = "deployed"


class Label(BaseModel):
    project_id: UUID4
    annotation_format_version: str
    task_id: str
    task_phase: TaskPhase
    task_phase_stage: int
    task_status: TaskStatus
    input_data_id: UUID4
    input_data_name: str
    details: List[Detail]
    updated_datetime: datetime

    def validate_datetime(cls, value):
        if isinstance(value, datetime):
            return value

        # 確認: valueがstr型である
        if not isinstance(value, str):
            raise TypeError(f"Expected str, got {type(value)}")
        # str型の場合、datetimeに変換
        return datetime.fromisoformat(value)


def load_csv_labels(path: str):
    # csv形式で保存されているlabelの検出
    import os

    import numpy as np
    import pandas as pd

    df = pd.read_csv(path)

    # filenameを取得
    basenames = os.path.basename(path).split("_")
    user_info = {
        "tms_date": basenames[0],
        "hotel": basenames[1],
        "capsule": basenames[2],
        "sleep_id": "" if len(basenames) < 4 else basenames[3],
    }

    # TMSDateの判定のために最初のinvalid_video
    invalid_video = df["invalid_video"].values
    # "no_data"ではない最初のindexを取得
    first_index = np.where(invalid_video != "no_data")[0][0]
    df.iloc[first_index]
    first_timestamp: datetime = datetime.strptime(
        df["timestamp"].values[first_index], "%Y-%m-%d %H:%M:%S"
    )
    # 10:00:00 > first_timestamp > 00:00:00 であればTMSDateが一日進んでいる
    user_info["lodging_date"] = first_timestamp.strftime("%Y%m%d")

    return df, user_info


if __name__ == "__main__":
    # JSONデータのサンプル
    import json

    sample_json_path = "examples/sample_labels/sample_labels.json"
    with open(sample_json_path, "r") as f:
        json_data = json.load(f)
    converted_data = convert_keys_to_underscore(json_data)
    # データをpydanticモデルにパース
    parsed_data = Label(**converted_data)
    print(parsed_data)
