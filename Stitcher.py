from time import sleep

import cv2
import numpy as np

from FeatureDetector import FeatureDetector
from ThreadedFeatureDetector import ThreadedFeatureDetector


class Stitcher:
    __cachedHomography = None
    __maxMatches = 100  # maximum matches to improve performance
    __minMatchesRequired = 4
    __homographyAlgorithm = cv2.RHO
    __maxCameras = 4

    def __init__(self, imagesToStitch):
        if imagesToStitch > self.__maxCameras:
            exit("This number of images is not supported!")
        self.cameras = imagesToStitch
        self.detector = FeatureDetector()
        self.matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
        if imagesToStitch >= 3:
            self.threadedDetector_1 = ThreadedFeatureDetector()
            self.threadedDetector_1.start_thread()
        if imagesToStitch == 4:
            self.threadedDetector_2 = ThreadedFeatureDetector()
            self.threadedDetector_2.start_thread()

    def __match_keypoints(self, kpA, kpB, dsA, dsB, ratio, reproThresh):
        raw_matches = self.matcher.knnMatch(dsA, dsB, 2)  # knn matching with 2 neighbours
        good_matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:  # Lowe's ratio test to filter some bad matches
                good_matches.append((m[0].trainIdx, m[0].queryIdx))
            if len(good_matches) >= self.__maxMatches:
                break

        if len(good_matches) < self.__minMatchesRequired:
            return None

        # use matcher
        cv_kpA = self.detector.cudaOrb.convert(kpA)
        cv_kpB = self.detector.cudaOrb.convert(kpB)

        # acquire points
        pointsA = np.float32([cv_kpA[i].pt for (_, i) in good_matches])
        pointsB = np.float32([cv_kpB[i].pt for (i, _) in good_matches])

        (H, status) = cv2.findHomography(pointsA, pointsB, self.__homographyAlgorithm, reproThresh)

        return good_matches, H, status

    def stitch(self, images: [np.ndarray]):
        image_count = len(images)

        # algorithm will be different for different amount of images hence if else
        if image_count == 2:
            return self.__stitch_two(images)

        if image_count == 3:
            self.threadedDetector_1.add_image_to_queue(images[3])
            two_images = self.__stitch_two([images[0], images[1]])
            kp3 = None
            ds3 = None
            while kp3 is None and ds3 is None:
                kp3, ds3 = self.threadedDetector_1.get_kp_ds()



        return None

    def __stitch_two(self, images):
        kp1, ds1 = self.detector.detect_features_and_keypoints(images[0])
        kp2, ds2 = self.detector.detect_features_and_keypoints(images[1])
        M = self.__match_keypoints(kp1, kp2, ds1, ds2, 0.8, 5)
        if M is None:
            return None
        good_matches, H, status = M
        result = cv2.warpPerspective(images[0], H, (images[0].shape[1] + images[1].shape[1], images[0].shape[0]))
        result[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]
        return result
