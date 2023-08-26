import datetime
import glob
import hashlib
import json
import os
from pathlib import PurePosixPath
from typing import Any, Dict, List, Tuple, Union

import cv2
import librosa
import numpy as np
import pandas as pd
from pydantic import BaseModel

from ts_segmentation.utils.images.util import equalize_histogram_color, rotate_image
from ts_segmentation.utils.vibrations.util import csv_data_timestamps2millisec

MIC_RAW_FS = 44100
CSV_FS = 500
DATA_PACK_SIZE = 10  # seconds


class UserData(BaseModel):
    user_data_dir: str
    timestamp: datetime.datetime
    has_m4v: bool
    has_wav: bool
    labels: Dict[str, Any]
    scores: List[float]

    def get_filepath(self, file_type="csv") -> str:
        basename = f"{self.timestamp.strftime('%Y-%m-%d_%H%M%S.%f')}.{file_type}"
        filepath = os.path.abspath(os.path.join(self.user_data_dir, basename))
        assert os.path.exists(filepath), f"{filepath} is not exists."
        return filepath

    def get_hash(self):
        return hash_directory(self.user_data_dir)


def load_user_df(filepath: str) -> pd.DataFrame:
    # , start_timestamp: str = "", end_timestamp=""): #this option not implemented
    user_data_df: pd.DataFrame = pd.read_csv(filepath)
    # timestampごとにappendするためにinplace safeでロード
    user_data_df.sort_values(by="timestamp", inplace=True)
    user_data_df["timestamp"] = user_data_df["timestamp"].apply(
        datetime.datetime.strptime, args=("%Y-%m-%d_%H%M%S.%f",)
    )

    # labels, scoresがstr のままになっているのでjson.loadsで変換
    user_data_df["labels"] = user_data_df["labels"].apply(json.loads)
    user_data_df["scores"] = user_data_df["scores"].apply(json.loads)
    return user_data_df


def hash_directory(dir_name: str) -> str:
    """Hash a directory name using SHA256."""
    hash_object = hashlib.sha256(dir_name.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def load_user_data_list(user_data_dir: str) -> pd.DataFrame:
    """preprocess user data dir which include m4v / csv / wav files.
    Each file title is timestamp.

    Args:
        user_data_dir (str): user data directory path

    Returns:
        data(pd.DataFrame): user data structre its contain below columns.
            [user_data_dir:str , timestamp:str , has_m4v:bool, has_wav:bool ,
                labels:dict, scores:list]
    """
    datas = glob.glob(os.path.join(user_data_dir, "*.csv"))
    user_datas = []
    for csv_data in datas:
        timestamp = (".").join(os.path.basename(csv_data).split(".")[:-1])
        m4v_path = os.path.join(user_data_dir, f"{timestamp}.m4v")
        wav_path = os.path.join(user_data_dir, f"{timestamp}.wav")
        if not os.path.exists(m4v_path):
            m4v_path = None
        if not os.path.exists(wav_path):
            wav_path = None
        user_datas.append(
            {
                "user_data_dir": user_data_dir,
                "timestamp": timestamp,
                "has_m4v": m4v_path is not None,
                "has_wav": wav_path is not None,
                "labels": {},
                "scores": [],
            }
        )
    return pd.DataFrame(user_datas)


def merge_dataset(
    base_data: pd.DataFrame,
    merge_datas: List[pd.DataFrame],
    concat_key: str = "timestamp",
) -> pd.DataFrame:
    for data_ in merge_datas:
        base_data = pd.merge(base_data, data_, on=concat_key, how="outer")
    # sorte base data with concat_key
    base_data.sort_values(by=concat_key, inplace=True)
    base_data.reset_index(drop=True, inplace=True)
    return base_data


class CapsuleMicDataset:
    def __init__(self, filepath: str, version=None):
        self._protocol = None
        self._filepath = PurePosixPath(filepath)
        self._version = version
        self._base_data = None
        self._user_hash = None
        self._get_user_hash()

    def _get_user_hash(self) -> str:
        user_data_df: pd.DataFrame = load_user_df(self._filepath)
        for tuple_user_data in user_data_df.itertuples():
            user_data = UserData(**tuple_user_data._asdict())
            self._user_hash = user_data.get_hash()
            break
        return self._user_hash

    def _load_wav(
        self,
        filepath: str,
        downsamples=1,
        fs=MIC_RAW_FS,
        output_dir: str = "./temp/02_intermediate",
        use_hash: bool = True,
        frame_id: int = 0,
        clip_seconds: int = DATA_PACK_SIZE,  # s * clip_ratio
    ) -> Tuple[pd.DataFrame, int]:
        # csv(1000Hz)にたいして 441000Hzでの測定のため、csv 側の基準に合わせて整形して
        # np.array(wav)をcellに保存する
        _file_timestamp = ("").join(os.path.basename(filepath).split(".")[:-1])
        file_timestamp = datetime.datetime.strptime(
            _file_timestamp, "%Y-%m-%d_%H%M%S%f"
        )
        if use_hash:
            hashed_dir_name = self._get_user_hash()
            output_dir = os.path.join(output_dir, hashed_dir_name)
            os.makedirs(output_dir, exist_ok=True)
        # load data as raw data
        sound_x, _ = librosa.load(filepath, sr=fs)
        if len(sound_x) == 0:
            return []
        sound_x = librosa.resample(sound_x, orig_sr=fs, target_sr=fs // downsamples)
        # check clip size for 1 s
        clip_size = fs // downsamples * clip_seconds
        wav_data = []

        # drop_range後の最初のtimestampを計算
        drop_milisecond_timestamp = file_timestamp.replace(microsecond=0)
        start_timestamp = drop_milisecond_timestamp.replace(
            second=file_timestamp.second // clip_seconds * clip_seconds
        )
        # filetimestamp から最初の切り落とす分のデータ長を計算
        drop_range = file_timestamp.microsecond // 1000000 * (
            fs // downsamples
        ) + file_timestamp.second * (fs // downsamples)

        for i in range(len(sound_x[drop_range:]) // clip_size):
            # タイムスタンプをdata_pack_sizeに合わせて整形 / start_time基準でclip_sizeごとに取得
            timestamp = start_timestamp + datetime.timedelta(seconds=i * clip_seconds)
            ## second単位で揃っているので、microsecondをmilisecに変換する必要はないs
            # timestamp_millisec = timestamp.microsecond // 1000
            # timestamp = timestamp.replace(microsecond=timestamp_millisec * 1000)
            # save wav data as .npy file
            output_file = os.path.join(
                output_dir, f"{timestamp.strftime('%Y-%m-%d_%H%M%S.%f')}.npy"
            )
            np.save(output_file, sound_x[i * clip_size : (i + 1) * clip_size])
            wav_data.append(
                {
                    "timestamp": timestamp.isoformat(timespec="milliseconds"),
                    "wav_id": frame_id,
                    "wav_path": output_file,
                }
            )
            frame_id += 1
        wav_data_df = pd.json_normalize(wav_data)
        return wav_data_df, frame_id

    def _recuesive_load(
        self,
        user_data: UserData,
        wave_frame_id: int = 0,
    ) -> Tuple[pd.DataFrame, int]:
        """この関数を呼び出す前にfileの呼び出しをチェックすること"""
        wav_data, wave_frame_id = self._load_wav(
            filepath=user_data.get_filepath("wav"),
            frame_id=wave_frame_id,
        )
        return wav_data, wave_frame_id

    def _load(
        self,
        files: list = [],
        wave_columns: list = ["timestamp", "wav_id", "wav_path"],
    ):
        user_data_df: pd.DataFrame = load_user_df(self._filepath)
        if files:
            assert max(files) <= len(
                user_data_df
            ), "[Optional] files args are out of range"
            user_data_df = user_data_df.iloc[files, :]

        data_columns = list(dict.fromkeys(wave_columns))
        base_data = pd.DataFrame(columns=data_columns, dtype=object)
        wave_frame_id = 0
        for tuple_user_data in user_data_df.itertuples():
            # check userdatas
            user_data = UserData(**tuple_user_data._asdict())
            if not user_data.has_wav:
                continue
            # load datas
            wave_data, wave_frame_id = self._recuesive_load(
                user_data=user_data,
                wave_frame_id=wave_frame_id,
            )
            wave_data = wave_data.reset_index(drop=True)
            base_data = pd.concat([base_data, wave_data], axis=0, ignore_index=True)
        base_data = base_data.reindex(data_columns, axis=1)
        self._base_data = base_data
        return base_data

    def _save(self, data: pd.DataFrame) -> None:
        # Hash the directory name
        hashed_dir_name = self._get_user_hash()

        # Save the data to a CSV file
        data.to_csv(f"temp/02_intermediate/{hashed_dir_name}/capsule_mic_dataset.csv")

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            version=self._version,
            # other details...
        )


class CapsuleCamDataset:
    def __init__(self, filepath: str, version=None):
        # filepath is user_data_list.csv
        self._protocol = None
        self._filepath = PurePosixPath(filepath)
        self._version = version
        self._base_data = None
        self._user_hash = None
        self._get_user_hash()

    def _get_user_hash(self) -> str:
        user_data_df: pd.DataFrame = load_user_df(self._filepath)
        for tuple_user_data in user_data_df.itertuples():
            user_data = UserData(**tuple_user_data._asdict())
            self._user_hash = user_data.get_hash()
            break
        return self._user_hash

    def _load_m4v(
        self,
        filepath: str,
        output_dir: str = "./temp/02_intermediate",
        use_hash: bool = True,
        data_pack_size: int = DATA_PACK_SIZE,
        max_frame: int = 30 * 60,  # max size for 30 minutes video
        rotate_left_angle: int = 90,
        use_equalize_histogram_color: bool = False,
        frame_id: int = 0,
    ) -> Tuple[pd.DataFrame, int]:
        # メモリのやりくりのためにframedataそのものは直接class attr保存しない.
        # timestamp(ms) / frame_id / frame_tmp_path
        _file_timestamp = ("").join(os.path.basename(filepath).split(".")[:-1])
        file_timestamp = datetime.datetime.strptime(
            _file_timestamp, "%Y-%m-%d_%H%M%S%f"
        )
        if use_hash:
            hashed_dir_name = self._get_user_hash()
            output_dir = os.path.join(output_dir, hashed_dir_name)
            os.makedirs(output_dir, exist_ok=True)

        # ビデオキャプチャオブジェクトを作成します
        vidcap = cv2.VideoCapture(filepath)

        # ビデオのフレームレート（FPS）を取得します
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        # data pack sizeからdownsample sizeを推定
        downsamplesize = data_pack_size * fps
        # drop_range後の最初のtimestampを計算
        drop_milisecond_timestamp = file_timestamp.replace(microsecond=0)
        start_timestamp = drop_milisecond_timestamp.replace(
            second=file_timestamp.second // data_pack_size * data_pack_size
        )
        # filetimestamp から最初の切り落とす分のデータ長を計算
        drop_frames = (
            file_timestamp.microsecond // 1000000 + file_timestamp.second
        ) // fps
        m4v_data = []
        while True:
            # load data
            success, image = vidcap.read()
            if not success:
                break  # フレームがない場合はループを終了します

            # file名として%Y-%m-%d_%H%M%S.%fの形式にする
            timestamp = start_timestamp + datetime.timedelta(
                seconds=(frame_count - drop_frames) / fps  # drop frameをstart timeから補正
            )
            timestamp_millisec = timestamp.microsecond // 1000
            timestamp = timestamp.replace(microsecond=timestamp_millisec * 1000)

            frame_count += 1
            if frame_count < drop_frames:
                # 調整分フレームの切り落とし.
                continue

            if (frame_count - drop_frames) % downsamplesize != 0:
                # downsamplesize 分のn枚分のフレーム目の切り落とし
                continue

            # process frame datas
            # Equalize the image histogram
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if use_equalize_histogram_color:
                image = equalize_histogram_color(image)
            image = rotate_image(image, rotate_left_angle)
            # 出力ファイル名を作成します
            output_file = os.path.join(
                output_dir, f"{timestamp.strftime('%Y-%m-%d_%H%M%S.%f')}.jpg"
            )
            # フレームをJPEGファイルとして保存します
            cv2.imwrite(output_file, image)
            # timestamp(ms) / frame_id / frame_tmp_path
            m4v_data.append(
                {
                    "timestamp": timestamp.isoformat(timespec="milliseconds"),
                    "image_id": frame_id,
                    "image_path": output_file,
                }
            )
            frame_id += 1

        # ビデオキャプチャオブジェクトを解放します
        vidcap.release()
        # 結合ようにデータ特徴量表形式に変換します.
        m4v_data_df = pd.json_normalize(m4v_data)
        return m4v_data_df, frame_id

    def _load_datas(
        self,
        user_data: UserData,
        m4v_frame_id: int = 0,
    ) -> Tuple[pd.DataFrame, int]:
        m4v_data, m4v_frame_id = self._load_m4v(
            filepath=user_data.get_filepath("m4v"),
            frame_id=m4v_frame_id,
        )
        return m4v_data, m4v_frame_id

    def _load(
        self,
        files: list = [],
        m4v_columns: list = ["timestamp", "image_id", "image_path"],
    ) -> pd.DataFrame:
        # , start_timestamp: str = "", end_timestamp=""): #this option not implemented
        user_data_df: pd.DataFrame = load_user_df(self._filepath)
        if files:
            assert max(files) <= len(
                user_data_df
            ), "[Optional] files args are out of range"
            user_data_df = user_data_df.iloc[files, :]

        data_columns = list(dict.fromkeys(m4v_columns))
        base_data = pd.DataFrame(columns=data_columns, dtype=object)
        m4v_frame_id = 0
        for tuple_user_data in user_data_df.itertuples():
            user_data = UserData(**tuple_user_data._asdict())
            m4v_data, m4v_frame_id = self._load_datas(
                user_data=user_data, m4v_frame_id=m4v_frame_id
            )
            m4v_data = m4v_data.reset_index(drop=True)
            base_data = pd.concat([base_data, m4v_data], axis=0, ignore_index=True)

        # sort columns to data_columns
        base_data = base_data.reindex(data_columns, axis=1)
        self._base_data = base_data
        return base_data

    def _save(self, data: pd.DataFrame) -> None:
        # Hash the directory name
        hashed_dir_name = self._get_user_hash()

        # Save the data to a CSV file
        data.to_csv(f"temp/02_intermediate/{hashed_dir_name}/capsule_mic_dataset.csv")

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            version=self._version,
            user_data_hash=self._user_hash,
            # other details...
        )


class CapsuleCSVDataset:
    def __init__(self, filepath: str, version=None):
        # filepath is user_data_list.csv
        self._protocol = None
        self._filepath = PurePosixPath(filepath)
        self._version = version
        self._base_data = None
        self._user_hash = None
        self._get_user_hash()

    def _get_user_hash(self) -> str:
        user_data_df: pd.DataFrame = load_user_df(self._filepath)
        for tuple_user_data in user_data_df.itertuples():
            user_data = UserData(**tuple_user_data._asdict())
            self._user_hash = user_data.get_hash()
            break
        return self._user_hash

    def _load_csv(
        self,
        filepath: str,
        colum_label=["timestamp", "raw", "f1", "f2"],
        data_pack_size: int = DATA_PACK_SIZE,
        fs: int = CSV_FS,
    ):
        # read_csvのタイミングでheaderをcolum_lobelとして読み込む.
        csv_data = pd.read_csv(filepath, header=None, names=colum_label)
        # timestamp microsecond をmillisecondに切り下げ
        csv_data["timestamp"] = csv_data["timestamp"].apply(
            csv_data_timestamps2millisec
        )
        # 10秒ごとにデータをまとめる
        # data_pack_size_10s = 10

        # # 実行時間の測定を開始
        # start_time_10s = time()

        # # リサンプリング (10秒ごと)
        # resampled_data_10s = csv_data.resample(f'{data_pack_size_10s}S').agg(lambda x: x.tolist())

        # # タイムスタンプを所望の形式に変換
        # resampled_data_10s.reset_index(inplace=True)
        # resampled_data_10s['timestamp'] = resampled_data_10s['timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3])

        # # 実行時間の測定を終了
        # execution_time_10s = time() - start_time_10s

        # resampled_data_10s.head(10), execution_time_10s

        # 新しいDataFrameを作成
        new_data = []
        for idx in range(0, len(csv_data), fs * data_pack_size):
            segment = csv_data.iloc[idx : idx + fs * data_pack_size]
            if len(segment) == 0:
                continue
            timestamp = segment.iloc[0]["timestamp"]
            raw = segment["raw"].tolist()
            f1 = segment["f1"].tolist()
            f2 = segment["f2"].tolist()
            new_data.append(
                {
                    "timestamp": timestamp,
                    "raw": raw,
                    "f1": f1,
                    "f2": f2,
                }
            )
        new_df = pd.DataFrame(new_data)
        return new_df

    def _load(
        self,
        files: list = [],
        csv_columns: list = ["timestamp", "raw", "f1", "f2"],
    ) -> pd.DataFrame:
        # , start_timestamp: str = "", end_timestamp=""): #this option not implemented
        user_data_df: pd.DataFrame = load_user_df(self._filepath)
        if files:
            assert max(files) <= len(
                user_data_df
            ), "[Optional] files args are out of range"
            user_data_df = user_data_df.iloc[files, :]

        data_columns = list(dict.fromkeys(csv_columns))
        base_data = pd.DataFrame(columns=data_columns, dtype=object)
        for tuple_user_data in user_data_df.itertuples():
            user_data = UserData(**tuple_user_data._asdict())
            csv_data = self._load_csv(filepath=user_data.get_filepath("csv"))
            csv_data = csv_data.reset_index(drop=True)
            base_data = pd.concat([base_data, csv_data], axis=0, ignore_index=True)
        # sort columns to data_columns
        base_data = base_data.reindex(data_columns, axis=1)
        self._base_data = base_data
        return base_data

    def _save(self, data: pd.DataFrame) -> None:
        # Hash the directory name
        hashed_dir_name = self._get_user_hash()

        # Save the data to a CSV file
        data.to_csv(f"temp/02_intermediate/{hashed_dir_name}/capsule_mic_dataset.csv")

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            version=self._version,
            # other details...
        )


class CapsuleDataset:
    def __int__(self, filepath: str, version=None):
        self._protocol = None
        self._filepath = PurePosixPath(filepath)
        self._version = version
        self._base_data = None
        self._user_hash = None
        self._get_user_hash()

    def _get_user_hash(self) -> str:
        user_data_df: pd.DataFrame = load_user_df(self._filepath)
        for tuple_user_data in user_data_df.itertuples():
            user_data = UserData(**tuple_user_data._asdict())
            self._user_hash = user_data.get_hash()
            break
        return self._user_hash

    def _load(self) -> pd.DataFrame:
        hash = self._get_user_hash()
        # Load the data from each CSV file
        csv_data = pd.read_csv(f"temp/02_intermediate/{hash}/capsule_csv_dataset.csv")
        m4v_data = pd.read_csv(f"temp/02_intermediate/{hash}/capsule_cam_dataset.csv")
        wav_data = pd.read_csv(f"temp/02_intermediate/{hash}/capsule_mic_dataset.csv")

        # Merge the datasets
        merged_data = merge_dataset(csv_data, [m4v_data, wav_data])

        # Save the merged data to the intermediate directory
        # merged_data.to_csv(f"temp/02_intermediate/{hash}/merged_data.csv")

        return merged_data

    def _save(self):
        pass

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            version=self._version,
            # other details...
        )


if __name__ == "__main__":
    # preprocess for data
    user_data_dir = "data/internal_sample_data/H00009/C121/20220630_4032980_01/data/"
    user_data_list = load_user_data_list(user_data_dir)
    user_data_list.to_csv("temp/01_raw/user_data.csv", index_label="id")

    # Sample user data
    sample_user_data_list = pd.read_csv("temp/01_raw/user_data.csv", index_col=0)
    print("user_data", len(sample_user_data_list))
    # files はsample_user_data_listの1/2で作成
    files = [len(sample_user_data_list) // 2]
    print("load_csv")
    csv_dataset = CapsuleCSVDataset(filepath="temp/01_raw/user_data.csv")
    csv_data: pd.DataFrame = csv_dataset._load(files=files)
    print("load_m4v")
    m4v_dataset = CapsuleCamDataset(filepath="temp/01_raw/user_data.csv")
    m4v_data: pd.DataFrame = m4v_dataset._load(files=files)
    print("load_wav")
    wav_dataset = CapsuleMicDataset(filepath="temp/01_raw/user_data.csv")
    wav_data: pd.DataFrame = wav_dataset._load(files=files)
    print("merge_dataset")
    data_ = merge_dataset(csv_data, [m4v_data, wav_data])
    print(data_)
    data_.to_csv("temp/02_intermediate/merged_data.csv")
    # capsuledataset = CapsuleDataset(filepath="temp/01_raw/user_data.csv")
    # capsuledataset._load(files=files)
    # print(capsuledataset._base_data)
