import threading

import cv2
import numpy as np

from CameraInterface import CameraInterface


class USBCamera(CameraInterface):
    __readThread: threading.Thread
    __frame: np.ndarray
    videoCapture: cv2.VideoCapture

    def __init__(self, width, height, dev):
        self.width = width
        self.height = height
        self.dev = dev
        self.__lock = threading.Lock()
        self.__readFrames = True

    def open_camera(self) -> None:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(self.dev, self.width, self.height)
        self.videoCapture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        print(self.videoCapture.isOpened())

    def __read_frames(self):
        while self.__readFrames:
            with self.__lock:
                _, self.__frame = self.videoCapture.read()

    def start_frame_capture(self) -> None:
        self.__readFrames = True
        self.__readThread = threading.Thread(target=self.__read_frames)
        self.__readThread.daemon = True
        self.__readThread.start()

    def get_frame(self) -> np.ndarray:
        with self.__lock:
            return self.__frame

    def stop_capture(self) -> None:
        self.__readFrames = False
        self.__readThread.join()
        self.videoCapture.release()
