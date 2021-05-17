from enum import Enum

import cv2
import numpy as np
import timeit

orb_detector = cv2.cuda_ORB.create(nfeatures=300)
fast_detector = cv2.cuda_FastFeatureDetector.create()


class Detector(Enum):
    ORB = 1
    FAST = 2


def detect_features(image: np.ndarray, detector: Detector):
    global orb_detector
    global fast_detector
    cuda_mat = cv2.cuda_GpuMat()
    grey_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cuda_mat.upload(grey_img)

    if detector is Detector.ORB:
        kp, ds = orb_detector.detectAndComputeAsync(cuda_mat, None)
    else:
        kp, ds = fast_detector.detectAndComputeAsync(cuda_mat, None)

    return kp, ds


def draw_keypoints_orb(kp, image):
    global orb_detector
    return cv2.drawKeypoints(image, orb_detector.convert(kp), None, color=(0, 255, 0))


def draw_keypoints_fast(kp, image):
    global fast_detector
    return cv2.drawKeypoints(image, fast_detector.convert(kp), None, color=(0, 255, 0))
