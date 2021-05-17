import subprocess
import sys

import cv2
import threading
import numpy as np

from CameraInterface import CameraInterface


class OnboardCamera(CameraInterface):
    __readThread: threading.Thread
    __frame: np.ndarray
    videoCapture: cv2.VideoCapture

    def __init__(self, width, height):
        self.__width = width
        self.__height = height
        self.__lock = threading.Lock()
        self.__read = True

    def open_camera(self) -> None:
        # from https://gist.github.com/jkjung-avt/86b60a7723b97da19f7bfa3cb7d2690e
        gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
        if 'nvcamerasrc' in gst_elements:
            # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
            gst_str = ('nvcamerasrc ! '
                       'video/x-raw(memory:NVMM), '
                       'width=(int)2592, height=(int)1458, '
                       'format=(string)I420, framerate=(fraction)30/1 ! '
                       'nvvidconv ! '
                       'video/x-raw, width=(int){}, height=(int){}, '
                       'format=(string)BGRx ! '
                       'videoconvert ! appsink').format(self.__width, self.__height)
        elif 'nvarguscamerasrc' in gst_elements:
            gst_str = ('nvarguscamerasrc ! '
                       'video/x-raw(memory:NVMM), '
                       'width=(int)1920, height=(int)1080, '
                       'format=(string)NV12, framerate=(fraction)30/1 ! '
                       'nvvidconv flip-method=2 ! '
                       'video/x-raw, width=(int){}, height=(int){}, '
                       'format=(string)BGRx ! '
                       'videoconvert ! appsink').format(self.__width, self.__height)
        else:
            raise RuntimeError('onboard camera source not found!')
        self.videoCapture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def __read_frames(self):
        while self.__read:
            with self.__lock:
                _, self.__frame = self.videoCapture.read()

    def start_frame_capture(self) -> None:
        if not self.videoCapture.isOpened():
            sys.exit('Failed to open onboard camera!')
        self.__read = True
        self.__readThread = threading.Thread(target=self.__read_frames)
        self.__readThread.daemon = True
        self.__readThread.start()

    def get_frame(self) -> np.ndarray:
        with self.__lock:
            return self.__frame

    def stop_capture(self):
        self.__read = False
        self.__readThread.join()
        self.videoCapture.release()
