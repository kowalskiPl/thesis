import imutils
from flask import Flask
from flask import Response
from flask import render_template
import argparse
import time
import cv2

from FeatureDetector import FeatureDetector
from OnboardCamera import OnboardCamera
from Stitcher import Stitcher
from USBCamera import USBCamera

onboard_camera: OnboardCamera  # OnboardCamera(1280, 720)
app = Flask(__name__)
stitcher: Stitcher
cameras: [USBCamera] = []
camera_count: int

@app.route("/")
def index():
    return render_template("index.html")


def generate_frame():
    while True:
        frames = []
        for cam in cameras:
            frames.append(cam.get_frame())
        frame = stitcher.stitch(frames)
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    global onboard_camera
    global cameras
    global stitcher
    global camera_count
    usb_cam_1: USBCamera
    usb_cam_2: USBCamera
    usb_cam_3: USBCamera
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-oc", "--onboard-camera", required=False, help="Tells the program to use onboard camera",
                    action="store_true")
    ap.add_argument("-c1", "--usb-camera-1", required=False, type=int, help="Specifies index of usb camera 1")
    ap.add_argument("-c2", "--usb-camera-2", required=False, type=int, help="Specifies index of usb camera 2")
    ap.add_argument("-c3", "--usb-camera-3", required=False, type=int, help="Specifies index of usb camera 3")
    args = vars(ap.parse_args())

    camera_count = 0
    if args["onboard_camera"]:
        onboard_camera = OnboardCamera(1280, 720)
        onboard_camera.open_camera()
        onboard_camera.start_frame_capture()
        camera_count += 1
        cameras.append(onboard_camera)

    if args["usb_camera_1"] is not None:
        usb_cam_1 = USBCamera(1280, 720, args["usb_camera_1"])
        usb_cam_1.open_camera()
        usb_cam_1.start_frame_capture()
        cameras.append(usb_cam_1)
        camera_count += 1

    if args["usb_camera_2"] is not None:
        usb_cam_2 = USBCamera(1280, 720, args["usb_camera_2"])
        usb_cam_2.open_camera()
        usb_cam_2.start_frame_capture()
        cameras.append(usb_cam_2)
        camera_count += 1

    if args["usb_camera_3"] is not None:
        usb_cam_3 = USBCamera(1280, 720, args["usb_camera_3"])
        usb_cam_3.open_camera()
        usb_cam_3.start_frame_capture()
        cameras.append(usb_cam_3)
        camera_count += 1

    stitcher = Stitcher(camera_count)

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
