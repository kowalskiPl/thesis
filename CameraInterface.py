import numpy as np


class CameraInterface:
    def open_camera(self) -> None:
        """Open video capture for camera"""
        pass

    def start_frame_capture(self) -> None:
        """Start reading frames, most likely asynchronously"""
        pass

    def get_frame(self) -> np.ndarray:
        """Return current frame"""
        pass

    def stop_capture(self) -> None:
        """Stop video capture, release resources"""
        pass
