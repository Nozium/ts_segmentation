import numpy as np
import pandas as pd
import torchvision

# import test data
merged_data = pd.read_csv(
    "temp/02_intermediate/e84518986d7245779c480f39037aa2008f0d0aa1c1eb2ca225e29a45dc33f333_merged_data.csv"
)
merged_data.head()


def load_img_link(img_path: str):
    # load jpg image from img_path
    import cv2

    return cv2.imread(img_path)


def detect_human_bbox(image: np.array, model):
    pass
