"""Weld bead and arc glow drawn in Gazebo with visual markers.

The bead is a tube along the part of the seam already welded. It is fixed to
the workpiece, so its points are moved with the positioner at every update.
By default it follows the helical seam of the paper; the industrial
applications give their own seams as segments (time, point, normal in frame k).
"""
import time

import numpy as np
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.marker_pb2 import Marker
from gz.transport13 import Node

from . import planner as pl

SIDES = 6
R_BEAD = 0.006
HOT_TIME = 1.5
CHUNK = 360       # vertices per marker message     # last seconds of bead drawn hot (glowing)


def _color(c, msg, emissive=None):
    msg.ambient.r, msg.ambient.g, msg.ambient.b, msg.ambient.a = c
    msg.diffuse.r, msg.diffuse.g, msg.diffuse.b, msg.diffuse.a = c
    if emissive:
        msg.emissive.r, msg.emissive.g, msg.emissive.b, msg.emissive.a = emissive


def _tube(points, normals):
    """Triangle list of a tube along points; normals give the outward direction."""
    tris = []
    if len(points) < 2:
        return tris
    rings = []
    for i, p in enumerate(points):
        tdir = points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]
        tdir /= max(np.linalg.norm(tdir), 1e-9)
        n = normals[i] - tdir * (normals[i] @ tdir)
        n /= np.linalg.norm(n)
        b = np.cross(tdir, n)
        c = p + n * 0.6 * R_BEAD          # bead sits on the surface
        ang = np.linspace(0, 2 * np.pi, SIDES, endpoint=False)
        rings.append([c + R_BEAD * (np.cos(a) * n + np.sin(a) * b) for a in ang])
    for r0, r1 in zip(rings, rings[1:]):
        for k in range(SIDES):
            a, b, c, d = r0[k], r0[(k + 1) % SIDES], r1[k], r1[(k + 1) % SIDES]
            tris += [a, c, b, b, c, d]
    return tris


class Bead:
    def __init__(self, segments=None):
        """segments: list of (times, points, normals) of each seam in frame k; None = paper seam."""
        self.node = Node()
        # /marker is served by the Gazebo window, /sensors/marker by the camera sensors
        self.services = ['/marker', '/sensors/marker']
        if segments is None:
            ts = np.arange(0.0, pl.T_TASK + 1e-9, 0.2)
            pk = np.array([pl.torch_on_workpiece(t)[:3, 3] for t in ts])
            nk = pk * np.array([1, 1, 0])                # radial direction on the pipe
            nk /= np.linalg.norm(nk, axis=1, keepdims=True)
            segments = [(ts, pk, nk)]
        self.segments = segments
        self.rng = np.random.default_rng(0)
        self.active, self.checked = [], -1e9

    def _send(self, m):
        now = time.monotonic()
        if now - self.checked > 5.0:              # look for the marker services every 5 s
            available = set(self.node.service_list())
            self.active = [s for s in self.services if s in available]
            self.checked = now
        for srv in self.active:
            # one-way service: no reply comes, so do not wait for one
            self.node.request(srv, m, Marker, Empty, 1)

    def _marker(self, mid, mtype, color, emissive=None):
        m = Marker()
        m.ns = 'weld'
        m.id = mid
        m.action = Marker.ADD_MODIFY
        m.type = mtype
        _color(color, m.material, emissive)
        m.material.double_sided = True
        return m

    def clear(self):
        m = Marker()
        m.ns = 'weld'
        m.action = Marker.DELETE_ALL
        self._send(m)

    def update(self, tau, phi1, phi2):
        """tau: task time [s]; phi1, phi2: positioner angles now [rad]."""
        T = pl.T_E_M0 @ pl.positioner(phi1, phi2) @ pl.T_M2_K
        R, p = T[:3, :3], T[:3, 3]
        parts, tips = [], []                     # (points, normals, hot) of each tube; arc tips
        for ts, pk, nk in self.segments:
            if tau <= ts[0]:
                continue
            n = int(np.searchsorted(ts, tau, side='right'))
            P, N = pk[:n], nk[:n]
            if tau < ts[-1] and tau - ts[n - 1] > 1e-3:     # exact point under the torch
                w = (tau - ts[n - 1]) / (ts[n] - ts[n - 1])
                P = np.vstack([P, (1 - w) * pk[n - 1] + w * pk[n]])
                N = np.vstack([N, (1 - w) * nk[n - 1] + w * nk[n]])
            pts, nrm = P @ R.T + p, N @ R.T
            arc = tau < ts[-1]
            step = (ts[-1] - ts[0]) / max(len(ts) - 1, 1)
            hot = max(0, len(pts) - int(HOT_TIME / step)) if arc else len(pts)
            parts.append((pts[:hot + 1], nrm[:hot + 1], False))
            if arc:
                parts.append((pts[hot:], nrm[hot:], True))
                tips.append(pts[-1] + nrm[-1] * 0.004)
        mid = 10
        for P, N, is_hot in parts:
            col, em = ((1.0, 0.45, 0.1, 1), (1.0, 0.4, 0.05, 1)) if is_hot else ((0.32, 0.24, 0.18, 1), None)
            tris = _tube(P, N)
            for k in range(0, len(tris), CHUNK):     # small messages render reliably
                m = self._marker(mid, Marker.TRIANGLE_LIST, col, em)
                m.scale.x = m.scale.y = m.scale.z = 1.0
                m.pose.orientation.w = 1.0
                for v in tris[k:k + CHUNK]:
                    q = m.point.add()
                    q.x, q.y, q.z = v
                self._send(m)
                mid += 1
        for k in range(4):                 # arc glow at the torch tips
            if k < len(tips):
                m = self._marker(3 + k, Marker.SPHERE, (1.0, 0.95, 0.75, 0.9), (1.0, 0.9, 0.6, 1))
                m.pose.position.x, m.pose.position.y, m.pose.position.z = tips[k]
                s = self.rng.uniform(0.012, 0.022)
                m.scale.x = m.scale.y = m.scale.z = s
            else:
                m = Marker()
                m.ns, m.id, m.action = 'weld', 3 + k, Marker.DELETE_MARKER
            self._send(m)
