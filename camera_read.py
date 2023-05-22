import cv2
import mediapipe as mp
import numpy as np
import threading


class CamRead(threading.Thread):
    image1 = 0
    image2 = 0
    image3 = 0

    exit_condition = False

    def __init__(self, cam_id):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.last_image = 0

    def run(self):
        self.camera_start(self.cam_id)

    def get_frame(self):
        if self.last_image == 1:
            return self.image1
        elif self.last_image == 2:
            return self.image2
        elif self.last_image == 3:
            return self.image3
        else:
            return None

    def camera_start(self, cam_id):
        capture = cv2.VideoCapture(cam_id)

        if not capture.isOpened():
            print("Cannot open camera")
            exit(1)
        else:
            print("Camera is open")

            while capture.isOpened():

                if self.exit_condition:
                    break

                if self.last_image == 0 or self.last_image == 3:
                    success, self.image1 = capture.read()
                    self.last_image = 1

                elif self.last_image == 1:
                    success, self.image2 = capture.read()
                    self.last_image = 2

                elif self.last_image == 2:
                    success, self.image3 = capture.read()
                    self.last_image = 3

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use "break" instead of "continue"
                    continue

            capture.release()
