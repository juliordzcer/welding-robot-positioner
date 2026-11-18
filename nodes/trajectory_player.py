#!/usr/bin/env python3
"""Plays the planned joint trajectories of the robot and the positioner.

Reads a trajectory file (made by comrob2021_welding.planner), waits
`start_delay` seconds of simulation time and sends one position command per
joint to Gazebo, interpolated at `rate` Hz. When the task ends it publishes
the results; the service /welding/restart runs the task again. With `loop`
true the task repeats by itself.

Topics:   /welding/status (std_msgs/String, JSON), 10 Hz
Services: /welding/restart (std_srvs/Trigger)
"""
import json
import os
import threading
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

from comrob2021_welding import planner as pl

JOINTS = ['positioner_joint_1', 'positioner_joint_2'] + [f'joint_{i}' for i in range(1, 7)]
COLUMNS = [1, 2, 4, 5, 6, 7, 8, 9]


class TrajectoryPlayer(Node):
    def __init__(self):
        super().__init__('trajectory_player')
        share = get_package_share_directory('comrob2021_welding')
        self.declare_parameter('csv', os.path.join(share, 'config', 'trajectory.csv'))
        self.declare_parameter('start_delay', 3.0)
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('bead', True)
        csv = self.get_parameter('csv').value
        self.mode = 'two_axis' if 'two_axis' in os.path.basename(csv) else 'paper'
        data = np.loadtxt(csv, delimiter=',', skiprows=1)
        self.t = data[:, 0]
        self.q = np.unwrap(data[:, COLUMNS], axis=0)   # no 2*pi jumps between samples
        self.wt, self.wr = data[:, 10], data[:, 11]
        self.gap = data[:, 12] if data.shape[1] > 12 else np.full(len(self.t), np.nan)
        self.results = self.summary()
        self.delay = self.get_parameter('start_delay').value
        self.loop = self.get_parameter('loop').value
        self.pubs = [self.create_publisher(Float64, f'/welding/{j}/cmd_pos', 10) for j in JOINTS]
        self.status_pub = self.create_publisher(String, '/welding/status', 10)
        self.create_service(Trigger, '/welding/restart', self.restart)
        self.t0 = None
        self.tau = 0.0
        self.state = 'waiting'
        self.reported = False
        self.create_timer(1.0 / self.get_parameter('rate').value, self.step)
        self.create_timer(0.1, self.publish_status)
        self.bead = None
        if self.get_parameter('bead').value:
            from comrob2021_welding.bead import Bead
            self.bead = Bead()
            threading.Thread(target=self.draw_loop, daemon=True).start()
        self.get_logger().info(f'Mode {self.mode}: {self.t[-1]:.0f} s task, {len(self.t)} samples')

    def summary(self):
        w = [pl.mixed(a, b) for a, b in zip(self.wt, self.wr)]
        return {'f': round(float(pl.objective(w)), 4),
                'wT_min': round(float(self.wt.min()), 3), 'wT_mean': round(float(self.wt.mean()), 3),
                'wR_min': round(float(self.wr.min()), 3), 'wR_mean': round(float(self.wr.mean()), 3),
                'clearance_min_mm': round(float(1000 * np.nanmin(self.gap)), 0)
                if not np.all(np.isnan(self.gap)) else None,
                'duration_s': float(self.t[-1])}

    def restart(self, _req, res):
        self.t0 = None
        self.reported = False
        res.success = True
        res.message = 'Task restarted'
        self.get_logger().info('Task restarted')
        return res

    def step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        tau = now - self.t0 - self.delay
        T = self.t[-1]
        if self.loop and tau > T + 2.0:          # 2 s pause at the end, then repeat
            self.t0 = now - self.delay
            tau = 0.0
            self.reported = False
        self.state = 'waiting' if tau < 0 else 'running' if tau <= T else 'finished'
        tau = min(max(tau, 0.0), T)
        self.tau = tau
        for k, pub in enumerate(self.pubs):
            pub.publish(Float64(data=float(np.interp(tau, self.t, self.q[:, k]))))
        if self.state == 'finished' and not self.reported:
            self.reported = True
            r = self.results
            self.get_logger().info(
                f"Task finished ({self.mode}): f = {r['f']}, wT* min {r['wT_min']} / mean {r['wT_mean']}, "
                f"wR min {r['wR_min']} / mean {r['wR_mean']}, torch-chuck gap min {r['clearance_min_mm']} mm. "
                "Restart: ros2 service call /welding/restart std_srvs/srv/Trigger")

    def publish_status(self):
        tau = self.tau
        msg = {'mode': self.mode, 'state': self.state, 't': round(tau, 2), 'T': float(self.t[-1]),
               'phi1_deg': round(float(np.degrees(np.interp(tau, self.t, self.q[:, 0]))), 1),
               'phi2_deg': round(float(np.degrees(np.interp(tau, self.t, self.q[:, 1]))), 1),
               'wT': round(float(np.interp(tau, self.t, self.wt)), 3),
               'wR': round(float(np.interp(tau, self.t, self.wr)), 3),
               'gap_mm': round(float(1000 * np.interp(tau, self.t, self.gap)), 0),
               'results': self.results if self.state == 'finished' else None}
        self.status_pub.publish(String(data=json.dumps(msg)))

    def draw_loop(self):
        """Draws the weld bead in its own thread, so Gazebo calls never delay the joints."""
        last = None
        while rclpy.ok():
            tau = self.tau
            if last is not None and tau < last:
                self.bead.clear()
            if tau != last:
                self.bead.update(tau, float(np.interp(tau, self.t, self.q[:, 0])),
                                 float(np.interp(tau, self.t, self.q[:, 1])))
            last = tau
            time.sleep(0.2)


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
