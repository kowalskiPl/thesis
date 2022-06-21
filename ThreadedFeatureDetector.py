import queue
import threading

from FeatureDetector import FeatureDetector


class ThreadedFeatureDetector(FeatureDetector):
    __workerThread: threading.Thread
    __event: threading.Event
    __imageQueue: queue.Queue
    __kp = None
    __ds = None
    __lock: threading.Lock

    def __init__(self):
        super().__init__()
        self.__lock = threading.Lock()
        self.__event = threading.Event()
        self.__imageQueue = queue.Queue()

    def __thread_loop(self, event):
        while not event.isSet():
            if self.__imageQueue.empty():
                continue
            image = self.__imageQueue.get()
            with self.__lock:
                self.__kp, self.__ds = self.detect_features_and_keypoints(image)

    def start_thread(self):
        self.__workerThread = threading.Thread(name='Worker', target=self.__thread_loop, args=(self.__event,))
        self.__workerThread.daemon = True
        self.__workerThread.start()

    def stop_thread(self):
        self.__event.set()

    def add_image_to_queue(self, image):
        self.__imageQueue.put(image)

    def get_kp_ds(self):
        with self.__lock:
            return self.__kp, self.__ds

