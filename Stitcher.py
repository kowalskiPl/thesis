import cv2

from FeatureDetector import FeatureDetector


class Stitcher:
    cachedHomography = None
    def __init__(self, cameras):
        self.cameras = cameras
        self.detectors = []
        self.matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
        # create independent detector for each camera
        for i in range(len(cameras)):
            self.detectors.append(FeatureDetector())



    # def _stitch_images(self, images, ratio=0.75):


    def match_keypoints(self):
        self.matcher
