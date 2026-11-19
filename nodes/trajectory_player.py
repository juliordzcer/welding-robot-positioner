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

POSITIONER = ['positioner_joint_1', 'positioner_joint_2']


def segments(data, col, suffix=''):
    """Bead segments (times, points, normals in frame k) of the runs with the arc on."""
    if 'arc' + suffix not in col:
        return []
    arc = data[:, col['arc' + suffix]] > 0.5
    pk = data[:, [col[f'pk{suffix}_{a}'] for a in 'xyz']]
    nk = data[:, [col[f'nk{suffix}_{a}'] for a in 'xyz']]
    edges = np.flatnonzero(np.diff(np.r_[0, arc.astype(int), 0]))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        idx = np.arange(a, b)[::4]                       # one point every 4 samples is enough
        idx = np.r_[idx, b - 1] if idx[-1] != b - 1 else idx
        out.append((data[idx, 0], pk[idx], nk[idx]))
    return out


class TrajectoryPlayer(Node):
    def __init__(self):
        super().__init__('trajectory_player')
        share = get_package_share_directory('comrob2021_welding')
        self.declare_parameter('csv', os.path.join(share, 'config', 'trajectory.csv'))
        self.declare_parameter('start_delay', 3.0)
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('bead', True)
        self.declare_parameter('mode', '')
        csv = self.get_parameter('csv').value
        self.mode = self.get_parameter('mode').value or (
            'two_axis' if 'two_axis' in os.path.basename(csv) else 'paper')
        with open(csv) as f:
            names = f.readline().strip().split(',')
        col = {n: i for i, n in enumerate(names)}
        data = np.loadtxt(csv, delimiter=',', skiprows=1)
        self.col, self.data = col, data
        robot = [n for n in names if n.startswith('joint_') or n.startswith('r2_joint_')]
        self.joints = POSITIONER + robot
        self.t = data[:, 0]
        self.q = np.unwrap(data[:, [col['phi1'], col['phi2']] + [col[n] for n in robot]], axis=0)
        self.wt, self.wr = data[:, col['wT']], data[:, col['wR']]
        self.gap = data[:, col['clearance']] if 'clearance' in col else np.full(len(self.t), np.nan)
        self.welding = data[:, col['arc']] > 0.5 if 'arc' in col else np.ones(len(self.t), bool)
        self.results = self.summary()
        self.delay = self.get_parameter('start_delay').value
        self.loop = self.get_parameter('loop').value
        self.pubs = [self.create_publisher(Float64, f'/welding/{j}/cmd_pos', 10) for j in self.joints]
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
            segs = segments(data, col) + segments(data, col, '2')
            self.bead = Bead(segs or None)
            threading.Thread(target=self.draw_loop, daemon=True).start()
        self.get_logger().info(f'Mode {self.mode}: {self.t[-1]:.0f} s task, {len(self.t)} samples')

    def summary(self):
        """Results over the welding samples (the whole task in the paper modes)."""
        k = self.welding
        if 'arc2' in self.col:              # two robots: robot 2 welds too
            k = k | (self.data[:, self.col['arc2']] > 0.5)
        wt, wr, gap = self.wt[k], self.wr[k], self.gap[k]
        if 'wT2' in self.col:
            k2 = self.data[:, self.col['arc2']] > 0.5
            wt = np.r_[self.wt[self.welding], self.data[k2, self.col['wT2']]]
            wr = np.r_[self.wr[self.welding], self.data[k2, self.col['wR2']]]
        w = [pl.mixed(a, b) for a, b in zip(wt, wr)]
        return {'f': round(float(pl.objective(w)), 4),
                'wT_min': round(float(wt.min()), 3), 'wT_mean': round(float(wt.mean()), 3),
                'wR_min': round(float(wr.min()), 3), 'wR_mean': round(float(wr.mean()), 3),
                'clearance_min_mm': round(float(1000 * np.nanmin(gap)), 0)
                if not np.all(np.isnan(gap)) else None,
                'duration_s': float(self.t[-1]),
                'arc_s': round(float(np.sum(np.diff(self.t)[k[1:]])), 1)}

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
               'seam': self.seam_now(tau),
               'results': self.results if self.state == 'finished' else None}
        self.status_pub.publish(String(data=json.dumps(msg)))

    def seam_now(self, tau):
        """Letters of the seams being welded now, '' outside the industrial modes."""
        if 'seam' not in self.col:
            return ''
        i = min(int(np.searchsorted(self.t, tau)), len(self.t) - 1)
        out = [chr(65 + int(self.data[i, self.col[c]])) for c in ('seam', 'seam2')
               if c in self.col and self.data[i, self.col[c]] >= 0]
        return '+'.join(out) or 'moving'

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
