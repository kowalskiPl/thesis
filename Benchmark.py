import argparse
import sys
import time
from enum import Enum

import cv2
import imutils
import numpy as np

orb_detector = cv2.cuda_ORB.create(nfeatures=500)
fast_detector = cv2.cuda_FastFeatureDetector.create()


class Detector(Enum):
    ORB = 1
    FAST = 2


detector_type: Detector
cuda_mat = cv2.cuda_GpuMat()

def detect_features(image: np.ndarray, detector: Detector):
    global orb_detector
    global fast_detector
    global cuda_mat
    cuda_mat.upload(image)
    grey_mat = cv2.cuda.cvtColor(cuda_mat, cv2.COLOR_RGB2GRAY)

    if detector is Detector.ORB:
        kp, ds = orb_detector.detectAndComputeAsync(grey_mat, None)
        return kp, ds
    else:
        kp, ds = fast_detector.detectAndComputeAsync(grey_mat, None)
        return kp, ds


def draw_keypoints_orb(kp, image):
    global orb_detector
    return cv2.drawKeypoints(image, orb_detector.convert(kp), None, color=(0, 255, 0))


def draw_keypoints_fast(kp, image):
    global fast_detector
    return cv2.drawKeypoints(image, fast_detector.convert(kp), None, color=(0, 255, 0))


def main():
    global detector_type
    global fast_detector
    global orb_detector
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detect-compute", action="store_true", required=False)
    ap.add_argument("-s", "--stitch", action="store_true", required=False)
    ap.add_argument("-i1", "--image-1", type=str, required=True)
    ap.add_argument("-i2", "--image-2", type=str, required=False)
    ap.add_argument("-o", "--orb", action="store_true", required=False)
    ap.add_argument("-f", "--fast", action="store_true", required=False)
    ap.add_argument("-it", "--iterations", type=int, default=10, required=False)
    ap.add_argument("-dk", "--draw-keypoints", action="store_true", required=False)
    args = vars(ap.parse_args())

    if args["stitch"] is True and args["image_2"] is None:
        sys.exit("When stitching provide two images!")
    if args["orb"] is False and args["fast"] is False:
        sys.exit("Provide one detector type!")
    if args["orb"] is True and args["fast"] is True:
        sys.exit("Provide one detector type!")

    image_1 = cv2.imread("benchmarkImages/" + args["image_1"])
    image_1 = imutils.resize(image_1, width=1920)
    image_2: np.ndarray
    if args["image_2"] is not None:
        image_2 = cv2.imread("benchmarkImages/" + args["image_2"])
        image_2 = imutils.resize(image_2, width=1920)

    if args["orb"] is True:
        detector_type = Detector.ORB
    elif args["fast"] is True:
        detector_type = Detector.FAST

    iterations = args["iterations"]
    if args["detect_compute"] is True:
        kp = None
        ds = None
        start_time = time.perf_counter()
        for i in range(iterations):
            kp, ds = detect_features(image_1, detector_type)
        end_time = time.perf_counter()
        print("Average detection time using: " + detector_type.name + " with " + str(iterations) + " iterations:")
        print((end_time - start_time) / args["iterations"])
        if args["draw_keypoints"] is True:
            if detector_type is Detector.ORB:
                keypoints_frame = cv2.drawKeypoints(image_1, orb_detector.convert(kp), None, color=(255, 0, 50))
                cv2.imwrite(args["image_1"] + "_result.png", keypoints_frame)


if __name__ == '__main__':
    main()
