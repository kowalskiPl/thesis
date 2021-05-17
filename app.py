from flask import Flask
from flask import Response
from flask import render_template
import argparse
import time
import cv2

from FeatureDetector import FeatureDetector
from OnboardCamera import OnboardCamera

onboard_camera: OnboardCamera  # OnboardCamera(1280, 720)
# camera.open_camera()
app = Flask(__name__)
time.sleep(2.0)
# camera.start_frame_capture()
detector = FeatureDetector()


@app.route("/")
def index():
    return render_template("index.html")


def generate_frame():
    while True:
        frame = onboard_camera.get_frame()
        kp, ds = detector.detect_features_and_keypoints(frame)
        keypoints_frame = cv2.drawKeypoints(frame, detector.cudaOrb.convert(kp), None, color=(0, 255, 0))
        (flag, encodedImage) = cv2.imencode(".jpg", keypoints_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    global onboard_camera
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-oc", "--onboard-camera", required=False, help="Tells the program to use onboard camera",
                    action="store_true")
    ap.add_argument("-c", "--usb-cameras", required=False, type=int, help="Specifies number of usb cameras to use")
    args = vars(ap.parse_args())

    if args["onboard_camera"]:
        onboard_camera = OnboardCamera(1280, 720)
        onboard_camera.open_camera()
        onboard_camera.start_frame_capture()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
