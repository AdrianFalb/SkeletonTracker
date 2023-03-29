import time
import sys

import cv2
import threading

import camera_process
from udp_server import UdpServer
from camera_read import CamRead

if __name__ == '__main__':
    # print("---- OpenCV version is: " + cv2.__version__ + " ----\n")
    # print(f"Name of the script      : {sys.argv[0]}")
    # print(f"Arguments of the script : {sys.argv[1:]}")

    robot_ip_address = sys.argv[1]
    server_udp_port = sys.argv[2]
    # robot_ip_address = "192.168.1.13"
    # server_udp_port = "23432"

    udp_server = UdpServer(int(server_udp_port))
    # print("Port: " + str(sys.argv[2]))
    # print("IP: " + str(sys.argv[1]))

    # ================================================================================ Threads

    readThread = CamRead("http://" + str(robot_ip_address) + ":8000/stream.mjpg")
    # readThread = CamRead(0)

    readThread.start()
    time.sleep(2)

    while 1:
        image = readThread.getFrame()

        if image is not None or image != 0:
            cv2.imshow("Camera", cv2.flip(camera_process.processCameraData(image, udp_server, robot_ip_address), 1))
            if cv2.waitKey(5) & 0xFF == 27:  # Close on ESC
                readThread.exitCondition = True
                break
