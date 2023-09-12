import datetime
import os
import time

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def nms_bbox(boxes, scores, threshold=0.1):
    """
    Non-maximum suppression
    """
    # Extract indices of boxes to preserve
    keep_idx = torchvision.ops.nms(boxes, scores, threshold)
    # Keep only preserved boxes
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    return boxes, scores


def visualize_bbox(
    frame, boxes, scores, threshold=0.75, save_file="data/tmp/currant_humancount.jpg"
):
    num_persons = sum(scores > threshold)
    # Display the count
    cv2.putText(
        frame,
        f"Count: {num_persons}, {num_persons.item()}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    # bbox if bbox accuracy is over threshold
    for box, score in zip(boxes, scores):
        if score > threshold:
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                4,
            )
            # show score and class
            cv2.putText(
                frame,
                f"{score.item():.2f}",
                (int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    # Show the image
    cv2.imshow("Person Count", frame)
    os.makedirs("data/tmp", exist_ok=True)
    cv2.imwrite(save_file, frame)


def user_detection(
    frame: np.ndarray,
    target_coco_class={"person": 1, "bed": 65},
    threshold=0.75,
    do_rotate=True,
    model=None,
    with_visualization=False,
):
    if do_rotate:
        # 右側に頭がくるので、左側に90度回転
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        frame = frame
    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    img = transform(frame)

    # Initialize the pre-trained model
    if model is None:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Perform inference
    with torch.no_grad():
        prediction = model([img])

    # Extract results
    boxes, scores = prediction[0]["boxes"], prediction[0]["scores"]
    boxes, scores = nms_bbox(boxes, scores, threshold=threshold)
    # Count number of persons detected with confidence level above a threshold (e.g., 0.5)
    num_persons = sum(scores > threshold)
    if with_visualization:
        visualize_bbox(frame, boxes, scores, threshold=threshold)
    return num_persons.item() > 0


def human_count(threshold=0.75, model=None, with_visualization=False):
    # Initialize the pre-trained model
    if model is None:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to get image from webcam")
        return -1

    # Convert the image to PIL image and apply transformations
    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    img = transform(frame)

    # Perform inference
    with torch.no_grad():
        prediction = model([img])

    # Extract results
    boxes, scores = prediction[0]["boxes"], prediction[0]["scores"]
    boxes, scores = nms_bbox(boxes, scores, threshold=threshold)

    # Count number of persons detected with confidence level above a threshold (e.g., 0.5)
    num_persons = sum(scores > threshold)
    if with_visualization:
        visualize_bbox(frame, boxes, scores, threshold=threshold)
    cap.release()
    cv2.destroyAllWindows()
    return num_persons.item()


def human_counting(threshold=0.5, cap_interval=1):
    # Initialize the pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to get image from webcam")
            break

        # Convert the image to PIL image and apply transformations
        transform = T.Compose([T.ToPILImage(), T.ToTensor()])
        img = transform(frame)

        # Perform inference
        with torch.no_grad():
            prediction = model([img])

        # Extract results
        boxes, scores = prediction[0]["boxes"], prediction[0]["scores"]
        boxes, scores = nms_bbox(boxes, scores, threshold=threshold)

        # Count number of persons detected with confidence level above a threshold (e.g., 0.5)
        num_persons = sum(scores > threshold)

        # save value to csv. timestamp, num_persons
        with open("data/dummydata/currant_humancount.csv", "a") as f:
            f.write(f"{datetime.datetime.now()}, {num_persons}\n")

        # Display the count
        cv2.putText(
            frame,
            f"Count: {num_persons}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Show the image
        cv2.imshow("Person Count", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(cap_interval)

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    human_count(with_visualization=True)

    # load test image
    sample_image = cv2.imread(
        "data/internal_sample_data/H00009/C121/20220630_4032980_01/raw_images/2022-07-01_060025.410599.jpg"
    )
    is_user_in = user_detection(sample_image, with_visualization=True, do_rotate=False)
    print(is_user_in)
