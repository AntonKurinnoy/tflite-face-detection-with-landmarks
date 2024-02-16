import math

import cv2 as cv
import numpy as np
from typing import Tuple, Optional

from slim_model.utils import get_angle


def normalize(
        img: np.ndarray
) -> np.ndarray:
    """
    :param img: source image, RGB with HWC and range [0,255]
    :return: normalized image CHW Tensor for PIPNet
    """
    img = img.astype(np.float32)
    img /= 255.
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225
    img = img.transpose((2, 0, 1))  # HWC->CHW
    return img.astype(np.float32)

def draw_bboxes(img: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    im = img[:, :, :].copy()

    bboxes = bboxes[:, :4]
    bboxes = bboxes.reshape(-1, 4)

    # draw bbox
    for box in bboxes:
        im = cv.rectangle(im, (int(box[0]), int(box[1])),
                           (int(box[2]), int(box[3])), (0, 255, 0), 2)

    return im.astype(np.uint8)

def draw_landmarks(
        img: np.ndarray,
        landmarks: np.ndarray,
        print_all: bool = False,
        landmarks_number: int = 98,
        font: float = 0.25,
        circle: int = 1,
        text: bool = False,
        color: Optional[Tuple[int, int, int]] = (0, 255, 0),
        offset: int = 5,
        thickness: int = 1
) -> np.ndarray:
    im = img.astype(np.uint8).copy()
    if landmarks.ndim == 2:
        landmarks = np.expand_dims(landmarks, axis=0)

    if print_all:
        for i in range(landmarks.shape[0]):
            for j in range(landmarks[i].shape[0]):
                x, y = landmarks[i, j, :].astype(int).tolist()
                cv.circle(im, (x, y), circle, color, -1)
                if text:
                    b = np.random.randint(0, 255)
                    g = np.random.randint(0, 255)
                    r = np.random.randint(0, 255)
                    cv.putText(im, '{}'.format(i), (x, y - offset),
                                cv.FONT_ITALIC, font, (b, g, r), thickness)
    elif landmarks_number == 98:
        # 98
        x, y = landmarks[0, 54, :].astype(int).tolist()  # nose
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 60, :].astype(int).tolist()  # right eye
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 72, :].astype(int).tolist()  # left eye
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 62, :].astype(int).tolist()  # right eye up point
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 66, :].astype(int).tolist()  # right eye down point
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 70, :].astype(int).tolist()  # left eye up point
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 74, :].astype(int).tolist()  # left eye down point
        cv.circle(im, (x, y), circle, color, -1)
    elif landmarks_number == 29:
        # 29
        x, y = landmarks[0, 8, :].astype(int).tolist()  # right eye
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 9, :].astype(int).tolist()  # left eye
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 12, :].astype(int).tolist()  # right eye up point
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 13, :].astype(int).tolist()  # right eye down point
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 14, :].astype(int).tolist()  # left eye up point
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 15, :].astype(int).tolist()  # left eye down point
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 16, :].astype(int).tolist()  # right eye iris
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 17, :].astype(int).tolist()  # left eye iris
        cv.circle(im, (x, y), circle, color, -1)
        x, y = landmarks[0, 20, :].astype(int).tolist()  # nose - up point
        cv.circle(im, (x, y), circle, color, -1)

    return im.astype(np.uint8)


def get_eyes_status(landmarks):
    right_up = landmarks[0, 62, :].astype(int).tolist()  # right eye
    right_down = landmarks[0, 66, :].astype(int).tolist()  # left eye
    left_up = landmarks[0, 70, :].astype(int).tolist()  # nose
    left_down = landmarks[0, 74, :].astype(int).tolist()  # nose
    result = "EYES: "

    distance_1 = calculate_distances(right_up, right_down)
    distance_2 = calculate_distances(left_up, left_down)
    # print("Distance between vectors right_up and right_down: {}".format(distance_1))
    # print("Distance between vectors left_up and left_down: {}".format(distance_2))

    if abs(distance_1) < 5 and abs(distance_2) < 5:
        result = result + "CLOSED"
    else:
        result = result + "OPEN"

    return result


def calculate_distances(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_head_turn_98(landmarks):
    a = landmarks[0, 60, :].astype(int).tolist()  # right eye
    b = landmarks[0, 72, :].astype(int).tolist()  # left eye
    c = landmarks[0, 54, :].astype(int).tolist()  # nose
    result = "HEAD TURN: "

    abc = get_angle(a, b, c)
    bac = get_angle(b, a, c)
    acb = get_angle(a, c, b)
    # print("Angle(abc) between vectors BA and BC: {} degrees".format(abc))
    # print("Angle(bac) between vectors AB and AC: {} degrees".format(bac))
    # print("Angle(acb) between vectors AC and BC: {} degrees".format(acb))

    if 90 <= acb <= 110:
        result = result + "FRONT"
    elif acb > 110:
        result = result + "UP"
    elif acb < 90:
        result = result + "DOWN"

    if abs(abc - bac) < 5:
        result = result + "-FRONT"
    elif abc > bac:
        result = result + "-LEFT"
    else:
        result = result + "-RIGHT"

    return result


def get_head_turn_29(landmarks):
    a = landmarks[0, 8, :].astype(int).tolist()  # right eye
    b = landmarks[0, 9, :].astype(int).tolist()  # left eye
    c = landmarks[0, 20, :].astype(int).tolist()  # nose
    result = ""

    abc = get_angle(a, b, c)
    bac = get_angle(b, a, c)
    acb = get_angle(a, c, b)
    # print("Angle(abc) between vectors BA and BC: {} degrees".format(abc))
    # print("Angle(bac) between vectors AB and AC: {} degrees".format(bac))
    # print("Angle(acb) between vectors AC and BC: {} degrees".format(acb))

    if 75 < acb < 90:
        result = result + "FRONT"
    elif acb > 90:
        result = result + "UP"
    elif acb < 75:
        result = result + "DOWN"

    if abs(abc - bac) < 5:
        result = result + "-FRONT"
    elif abc > bac:
        result = result + "-LEFT"
    else:
        result = result + "-RIGHT"

    return result
