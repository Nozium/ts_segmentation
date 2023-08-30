import argparse
import logging

import pandas as pd

from ts_segmentation.datasets.capsule_datasets import (
    CapsuleCamDataset,
    CapsuleCSVDataset,
    CapsuleMicDataset,
    load_user_data_list,
    merge_dataset,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pipeline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_data_dir", "-d", type=str)
    parser.add_argument("--user_data_list_path", "-l", type=str)
    parser.add_argument(
        "--use_hash_name",
        type=bool,
    )
    return parser.parse_args()


def create_merge_dataset(
    user_data_dir: str,
    user_data_list_path: str,
    use_hash_name: bool = False,
) -> pd.DataFrame:
    # preprocess for data
    # user_data_dir = "data/internal_sample_data/H00009/C121/20220630_4032980_01/data/"
    user_data_list = load_user_data_list(user_data_dir)
    user_data_list.to_csv(user_data_list_path, index_label="id")

    # Sample user data
    sample_user_data_list = pd.read_csv(user_data_list_path, index_col=0)
    logger.info("user_data", len(sample_user_data_list))
    # files はsample_user_data_listの1/2で作成
    files = [len(sample_user_data_list) // 2]
    logger.info("load_csv")
    csv_dataset = CapsuleCSVDataset(filepath=user_data_list_path)
    csv_data: pd.DataFrame = csv_dataset._load(files=files)
    logger.info("load_m4v")
    m4v_dataset = CapsuleCamDataset(filepath=user_data_list_path)
    m4v_data: pd.DataFrame = m4v_dataset._load(files=files)
    logger.info("load_wav")
    wav_dataset = CapsuleMicDataset(filepath=user_data_list_path)
    wav_data: pd.DataFrame = wav_dataset._load(files=files)
    logger.info("merge_dataset")
    data_ = merge_dataset(csv_data, [m4v_data, wav_data])
    logger.info(data_)

    if use_hash_name:
        user_hash = csv_dataset._get_user_hash()
        data_.to_csv(f"temp/02_intermediate/{user_hash}_merged_data.csv")
    else:
        data_.to_csv("temp/02_intermediate/merged_data.csv")
    return data_


if __name__ == "__main__":
    args = pipeline_args()
    create_merge_dataset(
        user_data_dir=args.user_data_dir,
        user_data_list_path=args.user_data_list_path,
        use_hash_name=args.use_hash_name,
    )
