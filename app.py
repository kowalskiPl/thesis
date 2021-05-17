from flask import Flask
from flask import Response
from flask import render_template
import argparse
import time
import cv2

from FeatureDetector import FeatureDetector
from OnboardCamera import OnboardCamera

camera = OnboardCamera(1280, 720)
camera.open_camera()
app = Flask(__name__)
time.sleep(2.0)
camera.start_frame_capture()
detector = FeatureDetector()


@app.route("/")
def index():
    return render_template("index.html")


def generate_frame():
    while True:
        frame = camera.get_frame()
        kp, ds = detector.detect_features_and_keypoints(frame)
        keypoints_frame = cv2.drawKeypoints(frame, detector.cudaOrb.convert(kp), None, color=(0, 255, 0))
        (flag, encodedImage) = cv2.imencode(".jpg", keypoints_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)
    print('started')


if __name__ == '__main__':
    main()
