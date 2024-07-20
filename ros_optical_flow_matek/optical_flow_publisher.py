# Copyright (c) 2023 Aditya Kamath
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import serial
import time
import sys
import rclpy
import numpy as np
from typing import Optional
from rclpy.lifecycle import Node, Publisher, State, TransitionCallbackReturn
from rclpy.timer import Timer
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, Pose, Twist, Point, Quaternion, Vector3, TransformStamped, Transform
# from pmw3901 import PMW3901, PAA5100, BG_CS_FRONT_BCM, BG_CS_BACK_BCM

# hard-coded values for PAA5100 and PMW3901 (to be verified for PMW3901)
FOV_DEG = 42.0
RES_PIX = 35

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
        quality = payload[0]
        x_velocity = int.from_bytes(payload[1:3], byteorder='little', signed=True)
        y_velocity = int.from_bytes(payload[3:5], byteorder='little', signed=True)
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
                    self.data['height'].append(range_data)
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


class OpticalFlowPublisher(Node):
    def __init__(self, node_name='optical_flow'):
        super().__init__(node_name)
        self._odom_pub: Optional[Publisher] = None
        self._tf_broadcaster: Optional[TransformBroadcaster] = None
        self._timer_odom: Optional[Timer] = None
        self._timer_poll: Optional[Timer] = None

        # declare parameters and default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('timer_period', 0.01),
                ('poll_rate', 10.0),
                ('sensor_timeout', 1.0),
                ('parent_frame', 'odom'),
                ('child_frame', 'base_link'),
                ('x_init', 0.0),
                ('y_init', 0.0),
                ('z_height', 0.025),
                ('board', 'paa5100'),
                ('scaler', 5),
                # ('spi_nr', 0),
                # ('spi_slot', 'front'),
                ('rotation', 0),
                ('publish_tf', True),
                ('port', '/dev/matek'),
                ('baud', 115200),
            ]
        )
        
        self._pos_x = self.get_parameter('x_init').value
        self._pos_y = self.get_parameter('y_init').value
        self._pos_z = self.get_parameter('z_height').value
        self._scaler = self.get_parameter('scaler').value
        self._dt = self.get_parameter('timer_period').value
        self._poll_rate = self.get_parameter('poll_rate').value
        self._baud = self.get_parameter('baud').value
        self._port = self.get_parameter('port').value
        self._sensor = None
        self.last_time = None
        
        self.get_logger().info('Initialized')
        self.get_logger().info(f"port: {self._port}, baud: {self._baud}")
        self.get_logger().info(f"parent_frame: {self.get_parameter('parent_frame').value}, child_frame: {self.get_parameter('child_frame').value}")
        self.get_logger().info(f"x_init: {self._pos_x}, y_init: {self._pos_y}, z_height: {self._pos_z}")
        self.get_logger().info(f"board: {self.get_parameter('board').value}, scaler: {self._scaler}")

    def poll_sensor(self):
        try:
            self._sensor.poll_flow()
            vx, vy = float(self._sensor.data['xm'][0]), float(self._sensor.data['ym'][0])

            if self.last_time is not None:
                # self.get_logger().info('data: ' + repr(self._sensor.data))
                # Convert velocity to position change
                self._pos_x += vx * (self.get_clock().now().to_msg().sec - self.last_time) / 1000.
                self._pos_y += vy * (self.get_clock().now().to_msg().sec - self.last_time) / 1000.

            self.last_time = self.get_clock().now().to_msg().sec
            # self.get_logger().info(f"vx: {vx}, vy: {vy}, _pos_x: {self._pos_x}, _pos_y: {self._pos_y}")
            self.get_logger().info(f"_pos_x: {self._pos_x:.6f}, _pos_y: {self._pos_y:.6f}")
        except (RuntimeError, AttributeError) as e:
            self.get_logger().error('Exception occurred during poll_sensor: ' + str(e))

    def publish_odom(self):
        try:
            # TODO: Average out the samples?
            if len(self._sensor.data['xm']) == 0:
                # No samples yet
                return
            
            if self._odom_pub is not None and self._odom_pub.is_activated:
                dx, dy = 0.0, 0.0
                try:
                    # dx, dy = self._sensor.get_motion(timeout=self.get_parameter('sensor_timeout').value)
                    dx, dy = float(self._sensor.data['xm'][0]), float(self._sensor.data['ym'][0])
                    self._pos_z = float(self._sensor.data['height'][0]) / 1000 # MM
                    # self.get_logger().info(f"dx: {dx}, dy: {dy}, z: {self._pos_z}")
                except (RuntimeError, AttributeError) as e:
                    self.get_logger().error('Exception occurred during publish_odom: ' + str(e))

                fov = np.radians(FOV_DEG)
                cf = self._pos_z*2*np.tan(fov/2)/(RES_PIX*self._scaler)

                dist_x, dist_y = 0.0, 0.0
                if self.get_parameter('board').value == 'paa5100':
                    # Convert data from sensor frame to ROS frame for PAA5100
                    # ROS frame: front/back = +x/-x, left/right = +y/-y
                    # Sensor frame: front/back = -y/+y, left/right = +x/-x
                    dist_x = -1*cf*dy
                    dist_y = cf*dx
                elif self.get_parameter('board').value == 'pmw3901':
                    # ROS and Sensor frames are assumed to align for PMW3901 based on https://docs.px4.io/main/en/sensor/pmw3901.html#mounting-orientation
                    dist_x = cf*dx
                    dist_y = cf*dy
                elif self.get_parameter('board').value == '3901-l0x':
                    # TODO: Double check
                    dist_x = cf*dx
                    dist_y = cf*dy
                    # self.get_logger().info(f"dist_x: {round(dist_x, 4)}, dist_y: {round(dist_y, 4)}")
                
                # self._pos_x += dist_x
                # self._pos_y += dist_y
                # self.get_logger().info(f"_pos_x: {round(self._pos_x, 4)}, _pos_y: {round(self._pos_y, 4)}")

                odom_msg = Odometry(
                    header = Header(
                        stamp = self.get_clock().now().to_msg(),
                        frame_id = self.get_parameter('parent_frame').value
                    ),
                    child_frame_id = self.get_parameter('child_frame').value,
                    pose = PoseWithCovariance(
                        pose = Pose(position = Point(x=self._pos_x, y=self._pos_y, z=self._pos_z))
                    ),
                    twist = TwistWithCovariance(
                        twist = Twist(linear = Vector3(x=dist_x/self._dt, y=dist_y/self._dt, z=0.0))
                    ),
                )
                self._odom_pub.publish(odom_msg)

                if self.get_parameter('publish_tf').value is True:
                    tf_msg = TransformStamped(
                        header = odom_msg.header,
                        child_frame_id = odom_msg.child_frame_id,
                        transform = Transform(translation = Vector3(x=odom_msg.pose.pose.position.x,
                                                                    y=odom_msg.pose.pose.position.y,
                                                                    z=odom_msg.pose.pose.position.z)),
                    )
                    self._tf_broadcaster.sendTransform(tf_msg)
        except Exception as e:
            self.get_logger().error('Exception occurred during publishing: ' + str(e))
            self.get_logger().exception('Exception occurred during publishing')

    # def on_configure(self, state: State) -> TransitionCallbackReturn:
    #     sensor_classes = {'pwm3901': PMW3901, 'paa5100': PAA5100}
    #     SensorClass = sensor_classes.get(self.get_parameter('board').value)

    #     if SensorClass is not None:
    #         spi_slots = {'front': BG_CS_FRONT_BCM, 'back': BG_CS_BACK_BCM}
    #         self._sensor = SensorClass(spi_port=self.get_parameter('spi_nr').value, 
    #                                     spi_cs_gpio=spi_slots.get(self.get_parameter('spi_slot').value))
    #         self._sensor.set_rotation(self.get_parameter('rotation').value)

    #         if self._sensor is not None:
    #             self._odom_pub = self.create_lifecycle_publisher(Odometry, 'odom', qos_profile=qos_profile_sensor_data)
    #             self._tf_broadcaster = TransformBroadcaster(self)
    #             self._timer = self.create_timer(self._dt, self.publish_odom)
            
    #             self.get_logger().info('Configured')
    #             return TransitionCallbackReturn.SUCCESS
    #         else:
    #             self.get_logger().info('Configuration Failure: Invalid SPI Settings')
    #             return TransitionCallbackReturn.FAILURE
    #     else:
    #         self.get_logger().info('Configuration Failure: Invalid Sensor')
    #         return TransitionCallbackReturn.FAILURE

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring')

        try:
            self._sensor = MatekOpticalFlowSensor(port=self._port, baudrate=self._baud, timeout=self.get_parameter('sensor_timeout').value)
        except Exception as e:
            self.get_logger().error('Configuration Exception: Invalid Port or Baud Rate:' + str(e))
            self.get_logger().exception('Exception occurred during sensor initialization')
            return TransitionCallbackReturn.FAILURE

        self._odom_pub = self.create_lifecycle_publisher(Odometry, 'odom', qos_profile=qos_profile_sensor_data)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._timer_odom = self.create_timer(self._dt, self.publish_odom)
        self._timer_poll = self.create_timer(1.0/self._poll_rate, self.poll_sensor)
    
        self.get_logger().info('Configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activated')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.terminate()
        self.get_logger().info('Clean Up Successful')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.terminate()
        self.get_logger().info('Shut Down Successful')
        return TransitionCallbackReturn.SUCCESS
        
    def terminate(self):
        if self._timer_poll is not None:
            self._timer_poll.cancel()
            self.destroy_timer(self._timer_poll)
        if self._timer_odom is not None:
            self._timer_odom.cancel()
            self.destroy_timer(self._timer_odom)
        if self._odom_pub is not None:
            self.destroy_publisher(self._odom_pub)
        if self._tf_broadcaster is not None:
            del self._tf_broadcaster

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.terminate()
        node.destroy_node()

if __name__ == '__main__':
    main()
