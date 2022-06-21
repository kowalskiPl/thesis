import cv2
import numpy as np
import time


class FeatureDetector:
    __max_features = 1000

    def __init__(self):
        self.cudaMat = cv2.cuda_GpuMat()
        self.cudaOrb = cv2.cuda_ORB.create(nfeatures=FeatureDetector.__max_features)

    def detect_features_and_keypoints(self, image):
        grey_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.cudaMat.upload(grey_img)
        kp, ds = self.cudaOrb.detectAndComputeAsync(self.cudaMat, None)
        self.cudaMat.empty()
        return kp, ds
