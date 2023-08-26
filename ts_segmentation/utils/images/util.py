import datetime
import glob
import os
from typing import Optional

import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an image by a given angle.
    Args:
        image (np.ndarray): Image to rotate.
        angle (int): Angle to rotate the image by.
    Returns:
        np.ndarray: Rotated image.
    """
    # Get image dimensions
    (height, width) = image.shape[:2]
    # Get image center
    center = (width / 2, height / 2)

    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Get the size of the new image
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_height = int(height * abs_cos + width * abs_sin)
    new_width = int(height * abs_sin + width * abs_cos)

    # Adjust rotation matrix
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]

    # Perform rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    return rotated_image


def equalize_histogram_color(image: np.ndarray) -> np.ndarray:
    """Equalize the histogram of a color image.

    Args:
        image (np.ndarray): Image to equalize the histogram of.

    Returns:
        np.ndarray: Image with equalized histogram.
    """
    # Convert the image from BGR to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Apply histogram equalization on the Y channel
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    # Convert the image back to BGR color space
    equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return equalized_image


def convert_m4v_to_jpg(
    m4v_path: str,
    output_dir: str,
    downsamplesize: Optional[int] = 10,
    max_frame: Optional[int] = 60,
    rotate_left_angle: Optional[int] = 90,
    use_equalize_histogram_color: Optional[bool] = False,
) -> None:
    """convert m4v data to jpg data with downsampling.

    Args:
        m4v_path (str): m4v file path
        output_dir (str): output directory path
        downsamplesize (int, optional): downsampling size. Defaults to 10.
        max_frame (int, optional): max frame size. Defaults to 60.
        rotate_left_angle (int, optional): rotate angle. Defaults to 90.
        use_equalize_histogram_color (bool, optional): use equalize histogram color.
        Defaults to False.

    Returns:
        None
    """
    _file_timestamp = ("").join(os.path.basename(m4v_path).split(".")[:-1])
    file_timestamp = datetime.datetime.strptime(_file_timestamp, "%Y-%m-%d_%H%M%S%f")

    # ビデオキャプチャオブジェクトを作成します
    vidcap = cv2.VideoCapture(m4v_path)

    # ビデオのフレームレート（FPS）を取得します
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        # file名として%Y-%m-%d_%H%M%S.%fの形式にする
        timestamp = file_timestamp + datetime.timedelta(seconds=frame_count / fps)
        frame_count += 1
        if frame_count % downsamplesize != 0:
            continue
        if (frame_count / downsamplesize) > max_frame:
            break

        success, image = vidcap.read()
        if not success:
            break  # フレームがない場合はループを終了します
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
    # ビデオキャプチャオブジェクトを解放します
    vidcap.release()


def convert_m4vs_to_jpg(m4v_dir: str, output_dir: str) -> None:
    """convert m4v data to jpg data with downsampling.

    Args:
        m4v_dir (str): m4v directory path
        output_dir (str): output directory path

    Returns:
        None
    """
    m4v_paths = glob.glob(os.path.join(m4v_dir, "*.m4v"))
    for m4v_path in m4v_paths:
        convert_m4v_to_jpg(m4v_path, output_dir)
