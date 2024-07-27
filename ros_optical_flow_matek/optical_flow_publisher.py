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
from .matek_optical_flow_sensor import MatekOpticalFlowSensor

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
                ('poll_rate', 10.0),
                ('sensor_timeout', 1.0),
                ('parent_frame', 'odom'),
                ('child_frame', 'optical_flow_front_link'),
                ('board', '3901-l0x'),
                ('scaler', -800.),
                ('rotation', 0),
                ('publish_tf', True),
                ('port', '/dev/matek'),
                ('baud', 115200),
                ('fov_degrees', 42.0), # Field of view in degrees
                ('image_width_pixels', 35), # Width of image in pixels
            ]
        )

        self._pos_x = 0.
        self._pos_y = 0.
        self._pos_z = 0.
        self._scaler = self.get_parameter('scaler').value
        self._poll_rate = self.get_parameter('poll_rate').value
        self._dt = 1.0 / self._poll_rate
        self._baud = self.get_parameter('baud').value
        self._port = self.get_parameter('port').value
        self._sensor = None
        self.last_time = None
        self.linear_velocity = Vector3()

        self.get_logger().info('Initialized')
        self.get_logger().info(f"port: {self._port}, baud: {self._baud}")
        self.get_logger().info(f"parent_frame: {self.get_parameter('parent_frame').value}, child_frame: {self.get_parameter('child_frame').value}")
        self.get_logger().info(f"board: {self.get_parameter('board').value}, scaler: {self._scaler}")

    def poll_sensor(self):
        current_time = self.get_clock().now()
        try:
            self._sensor.poll_flow()
            delta_time = float((current_time - self.last_time).nanoseconds) / 1e9 if self.last_time is not None else self._dt
            # self.get_logger().info(f"type(delta_time): {type(delta_time)}, delta_time: {delta_time}, current_time: {current_time}, last_time: {self.last_time}")
            self.last_time = current_time

            # Get raw sensor value (average of data for more stable reading)
            # 13.000000
            # x_dps: -0.016250

            xv = np.average(self._sensor.data['xm']) # x-axis velocity
            yv = np.average(self._sensor.data['ym']) # y-axis velocity
            self._pos_z = np.average(self._sensor.data['height']) # Meters off the ground

            self.get_logger().info(f"xv: {xv:.6f}, yv: {yv:.6f}, _pos_z: {self._pos_z:.6f}")

            fov_degrees = self.get_parameter('fov_degrees').value
            fov_radians = np.radians(fov_degrees)
            fov_width_meters = 2.0 * self._pos_z * np.tan(fov_radians / 2.0)
            # self.get_logger().info(f"fov_degrees: {fov_degrees},  fov_radians: {fov_radians}. fov_width_meters: {fov_width_meters:.6f}, self._pos_z: {self._pos_z}")
            scale_factor = fov_width_meters / self.get_parameter('image_width_pixels').value

            # Calculate the distance traveled using self._pos_z in meters over time
            dist_x = xv * scale_factor
            dist_y = yv * scale_factor
            
            # Calculate velocities in meters per second
            x_vel = dist_x / delta_time
            y_vel = dist_y / delta_time
            self.linear_velocity.x = 0.0
            self.linear_velocity.y = 0.0

            self.get_logger().info(f"x_vel: {x_vel}, y_vel: {y_vel}")

            if not np.isnan(x_vel) and not np.isinf(x_vel):
                self.linear_velocity.x = x_vel

            if not np.isnan(y_vel) and not np.isinf(y_vel):
                self.linear_velocity.y = y_vel
            
            self.get_logger().info(f"linear_velocity: {self.linear_velocity}")

            # Update positions
            self._pos_x += dist_x
            self._pos_y += dist_y

            # self.get_logger().info(f"_pos_x: {self._pos_x:.6f}, _pos_y: {self._pos_y:.6f}")
        except (RuntimeError, AttributeError) as e:
            self.get_logger().error('Exception occurred during poll_sensor: ' + str(e))

    def publish_odom(self):
        try:
            if len(self._sensor.data['xm']) == 0:
                return
            
            # TODO: Calculate angular from transform from child to parent (radius = distance of sensor from midpoint of base_link)

            if self._odom_pub is not None and self._odom_pub.is_activated:
                odom_msg = Odometry(
                    header=Header(
                        stamp=self.get_clock().now().to_msg(),
                        frame_id=self.get_parameter('parent_frame').value
                    ),
                    child_frame_id=self.get_parameter('child_frame').value,
                    pose=PoseWithCovariance(
                        pose=Pose(position=Point(x=self._pos_x, y=self._pos_y, z=self._pos_z))
                    ),
                    twist=TwistWithCovariance(
                        twist=Twist(linear=self.linear_velocity, angular=Vector3())
                    ),
                )
                self._odom_pub.publish(odom_msg)

                if self.get_parameter('publish_tf').value is True:
                    tf_msg = TransformStamped(
                        header=odom_msg.header,
                        child_frame_id=odom_msg.child_frame_id,
                        transform=Transform(translation=Vector3(x=odom_msg.pose.pose.position.x,
                                                                y=odom_msg.pose.pose.position.y,
                                                                z=odom_msg.pose.pose.position.z)),
                    )
                    self._tf_broadcaster.sendTransform(tf_msg)
        except Exception as e:
            self.get_logger().error('Exception occurred during publishing: ' + str(e))
            self.get_logger().exception('Exception occurred during publishing')

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
        self._timer_poll = self.create_timer(1.0 / self._poll_rate, self.poll_sensor)

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
