from time import sleep

import cv2
import imutils
import numpy as np

from FeatureDetector import FeatureDetector
from ThreadedFeatureDetector import ThreadedFeatureDetector


class Stitcher:
    __cachedHomography = None
    __maxMatches = 100 # maximum matches to improve performance
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
            kp1, ds1 = self.detector.detect_features_and_keypoints(images[0])
            kp2, ds2 = self.detector.detect_features_and_keypoints(images[1])
            M = self.__match_keypoints(kp1, kp2, ds1, ds2, 0.8, 5)
            if M is None:
                return None
            good_matches, H, status = M
            result = cv2.warpPerspective(images[0], H, (images[0].shape[1] + images[1].shape[1], images[0].shape[0]))
            kp2, ds2 = self.detector.detect_features_and_keypoints(result)
            M = self.__match_keypoints(kp1, kp2, ds1, ds2, 0.8, 5)
            if M is None:
                return None
            good_matches, H, status = M
            result = cv2.warpPerspective(images[0], H, (images[0].shape[1] + images[1].shape[1], images[0].shape[0] + images[1].shape[0]))
            result[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]
            # gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = imutils.grab_contours(cnts)
            # c = max(cnts, key=cv2.contourArea)
            # (x, y, w, h) = cv2.boundingRect(c)
            # result = result[y:y + h, x:x + w]
            return result

        if image_count == 3:
            # queue threaded detection
            self.threadedDetector_1.add_image_to_queue(images[2])

            kp1, ds1 = self.detector.detect_features_and_keypoints(images[0])
            kp2, ds2 = self.detector.detect_features_and_keypoints(images[1])
            M = self.__match_keypoints(kp1, kp2, ds1, ds2, 0.8, 5)
            if M is None:
                return None
            good_matches, H, status = M
            two_images = cv2.warpPerspective(images[0], H, (images[0].shape[1] + images[1].shape[1], images[0].shape[0]))
            two_images[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]

            kp3 = None
            ds3 = None
            while kp3 is None and ds3 is None:
                kp3, ds3 = self.threadedDetector_1.get_kp_ds()
            kp12, ds12 = self.detector.detect_features_and_keypoints(two_images)
            M2 = self.__match_keypoints(kp12, kp3, ds12, ds3, 0.8, 5)
            if M2 is None:
                return None
            good_matches, H, status = M
            result = cv2.warpPerspective(two_images, H, (images[0].shape[1] + images[2].shape[1], two_images.shape[0]))
            result[0:images[2].shape[0], 0:images[2].shape[1]] = images[2]
            return result

        return None

