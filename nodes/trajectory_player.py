#!/usr/bin/env python3
"""Plays the planned joint trajectories of the robot and the positioner.

Reads config/trajectory.csv (made by comrob2021_welding.planner), waits
`start_delay` seconds of simulation time and sends one position command per
joint to Gazebo, interpolated at `rate` Hz. With `loop` true the task repeats.
"""
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64

JOINTS = ['positioner_joint_1', 'positioner_joint_2'] + [f'joint_{i}' for i in range(1, 7)]
COLUMNS = [1, 2, 4, 5, 6, 7, 8, 9]


class TrajectoryPlayer(Node):
    def __init__(self):
        super().__init__('trajectory_player')
        share = get_package_share_directory('comrob2021_welding')
        self.declare_parameter('csv', os.path.join(share, 'config', 'trajectory.csv'))
        self.declare_parameter('start_delay', 3.0)
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('loop', True)
        data = np.loadtxt(self.get_parameter('csv').value, delimiter=',', skiprows=1)
        self.t = data[:, 0]
        # unwrap so joints never jump by 2*pi between samples
        self.q = np.unwrap(data[:, COLUMNS], axis=0)
        self.delay = self.get_parameter('start_delay').value
        self.loop = self.get_parameter('loop').value
        self.pubs = [self.create_publisher(Float64, f'/welding/{j}/cmd_pos', 10) for j in JOINTS]
        self.t0 = None
        self.create_timer(1.0 / self.get_parameter('rate').value, self.step)
        self.get_logger().info(f'Trajectory of {self.t[-1]:.0f} s loaded, {len(self.t)} samples')

    def step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        tau = now - self.t0 - self.delay
        T = self.t[-1]
        if self.loop and tau > T + 2.0:          # 2 s pause at the end, then repeat
            self.t0 = now - self.delay
            tau = 0.0
        tau = min(max(tau, 0.0), T)
        for k, pub in enumerate(self.pubs):
            pub.publish(Float64(data=float(np.interp(tau, self.t, self.q[:, k]))))


def main():
    rclpy.init()
    node = TrajectoryPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
