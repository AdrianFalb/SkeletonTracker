import numpy as np
import socket


class UdpServer:
    def __init__(self, udp_port):
        self.data_to_send = "COMMAND:STOP"
        self.UDP_IP = "127.0.0.1"
        # self.UDP_IP = "147.175.108.19"
        self.UDP_PORT = udp_port
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def set_data(self, data):
        self.data_to_send = data

    def get_data(self):
        return self.data_to_send

    def send_message(self):
        self.data_socket.sendto(self.data_to_send, (self.UDP_IP, self.UDP_PORT))
