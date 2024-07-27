import serial
import time
import numpy as np

class MatekOpticalFlowSensor:
    FUNC_LIDAR = 7937
    FUNC_FLOW = 7938
    HEADER_SIZE = 8

    def __init__(self, port, baudrate=115200, timeout=1):
        self.serial_port = serial.Serial(port, baudrate, timeout=timeout)
        self.data = {
            'height': [],
            'xm': [],
            'ym': []
        }

    def crc8_dvb_s2(self, crc, a):
        crc ^= a
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0xD5
            else:
                crc = crc << 1
            crc &= 0xFF  # Ensure crc is always 8 bits
        return crc

    def calculate_checksum(self, data):
        crc = 0
        for byte in data:
            crc = self.crc8_dvb_s2(crc, byte)
        return crc

    def parse_lidar_payload(self, payload):
        quality = payload[0]
        range_data = int.from_bytes(payload[1:5], byteorder='little')
        return quality, range_data

    def parse_flow_payload(self, payload):
        # seems like anything less than 255 is ok. Maybe use it to calculate covariance
        quality = payload[0]
        # quality = int.from_bytes(payload[0:1], byteorder='little', signed=False)
        x_velocity = int.from_bytes(payload[1:5], byteorder='little', signed=True)
        y_velocity = int.from_bytes(payload[5:9], byteorder='little', signed=True)

        # print(f"parse_flow_payload payload: {[hex(x) for x in payload]} size: {len(payload)}, payload[1:5]: {payload[1:5]}, payload[5:9]: {payload[5:9]}")
        print(f"parse_flow_payload quality: {quality}, x_velocity: {x_velocity}, y_velocity: {y_velocity}")

        return quality, x_velocity, y_velocity

    def poll_flow(self, num_samples=1):
        self.data = {
            'height': [],
            'xm': [],
            'ym': []
        }

        lidar_count = 0
        flow_count = 0
        start_time = time.time()

        while len(self.data['height']) < num_samples or len(self.data['xm']) < num_samples or len(self.data['ym']) < num_samples:
            while True:
                if self.serial_port.read(1) == b'$':
                    break

            while self.serial_port.in_waiting < self.HEADER_SIZE - 1:
                time.sleep(0.01)

            header = b'$' + self.serial_port.read(self.HEADER_SIZE - 1)
            if header[1:2] != b'X' or header[2:3] not in b'<>!':
                print(f"Invalid header: {header}")
                continue

            flag = header[3]
            func = int.from_bytes(header[4:6], byteorder='little')
            size = int.from_bytes(header[6:8], byteorder='little')

            while self.serial_port.in_waiting < size + 1:
                time.sleep(0.01)

            payload = self.serial_port.read(size)
            msg_crc = self.serial_port.read(1)[0]

            checksum_data = header[3:] + payload
            calculated_checksum = self.calculate_checksum(checksum_data)

            if calculated_checksum != msg_crc:
                print(f"Checksum mismatch: expected {msg_crc}, calculated {calculated_checksum}")
                continue

            if func == self.FUNC_LIDAR and size == 5:
                lidar_count += 1
                quality, range_data = self.parse_lidar_payload(payload)
                if len(self.data['height']) < num_samples:
                    # millimeters to meters
                    self.data['height'].append(range_data / 1000.)
            elif func == self.FUNC_FLOW and size == 9:
                flow_count += 1
                quality, x_velocity, y_velocity = self.parse_flow_payload(payload)
                if quality < 255:
                    if len(self.data['xm']) < num_samples:
                        self.data['xm'].append(x_velocity)
                    if len(self.data['ym']) < num_samples:
                        self.data['ym'].append(y_velocity)
            else:
                print(f"Unknown packet: func={func}, size={size}, payload={payload}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        lidar_frequency = lidar_count / elapsed_time
        flow_frequency = flow_count / elapsed_time

        # print(f"Lidar message frequency: {lidar_frequency} Hz")
        # print(f"Optical flow message frequency: {flow_frequency} Hz")

        return self.data
