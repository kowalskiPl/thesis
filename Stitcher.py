import cv2
import numpy as np

from FeatureDetector import FeatureDetector


class Stitcher:
    cachedHomography = None
    __maxMatches = 100 # maximum matches to improve performance
    __minMatchesRequired = 20
    __homographyAlgorithm = cv2.RHO

    def __init__(self, cameras):
        self.cameras = cameras
        self.detectors = []
        self.matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
        # create independent detector for each camera
        for i in range(len(cameras)):
            self.detectors.append(FeatureDetector())

    def match_keypoints(self, kpA, kpB, dsA, dsB, ratio, reproThresh):
        raw_matches = self.matcher.knnMatch(kpA, kpB, 2)  # knn matching with 2 neighbours
        good_matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio: # Lowe's ratio test to filter some bad matches
                good_matches.append((m[0].trainIdx, m[0].queryIdx))
            if len(good_matches) >= self.__maxMatches:
                break

        if len(good_matches) < self.__minMatchesRequired:
            return None

        # grab random matcher to convert
        cv_kpA = self.detectors[0].cudaOrb.convert(kpA)
        cv_kpB = self.detectors[0].cudaOrb.convert(kpB)

        # acquire points
        pointsA = np.float32([cv_kpA[i].pt for (_, i) in good_matches])
        pointsB = np.float32([cv_kpB[i].pt for (i, _) in good_matches])

        (H, status) = cv2.findHomography(pointsA, pointsB, self.__homographyAlgorithm, reproThresh)

        return good_matches, H, status