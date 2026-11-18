"""Motion planning of the Fanuc ARC Mate 100iC and the 2-DOF positioner.

Reproduces Section II-IV of the paper: modified D-H models (Tables I-II),
the helical seam (16)-(23), the positioner and torch time functions
(26)-(28) with Tables III-IV, inverse kinematics and the translational and
rotational manipulabilities (11)-(15).

Usage:  python3 -m comrob2021_welding.planner [output_dir]
"""
import os
import sys

import numpy as np

D2R = np.pi / 180.0

# ---------------------------------------------------------------- geometry
# Table I: modified D-H (Khalil-Kleinfinger) of the robot, metres.
ROBOT_DH = [  # (alpha, d, r)
    (0.0, 0.0, 0.0),
    (90 * D2R, 0.150, 0.0),
    (0.0, 0.600, 0.0),
    (90 * D2R, 0.200, 0.640),
    (-90 * D2R, 0.0, 0.0),
    (90 * D2R, 0.0, 0.0),
]
RM2 = 0.130            # Table II
RHO = -22 * D2R        # torch, eq. (1)
DHX, DHY, DHZ = -0.0498, 0.0, 0.504
EZ = 0.450             # eq. (2)
MU, XM0, YM0, ZM0 = 90 * D2R, 1.200, 0.500, -0.010   # eq. (3)
PHI1 = 10.4 * D2R      # constant tilt of the positioner
RT, HT, T_TASK = 0.100, 0.450, 40.0                  # helical seam
WT_MAX = 0.45          # eq. (15)


def dh(alpha, d, theta, r):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, d],
                     [ca * st, ca * ct, -sa, -r * sa],
                     [sa * st, sa * ct, ca, r * ca],
                     [0, 0, 0, 1.0]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def hom(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


T_E_0 = hom(np.eye(3), [0, 0, EZ])
T_6_H = hom(ry(RHO), [DHX, DHY, DHZ])
T_E_M0 = hom(np.array([[np.cos(MU), 0, np.sin(MU)],
                       [np.sin(MU), 0, -np.cos(MU)],
                       [0, 1, 0]]), [XM0, YM0, ZM0])
T_M2_K = np.eye(4)     # workpiece frame on the platform


def robot_frames(q):
    """Frames 0..6 of the robot w.r.t. Sigma_0."""
    T = np.eye(4)
    out = []
    for (a, d, r), th in zip(ROBOT_DH, q):
        T = T @ dh(a, d, th, r)
        out.append(T.copy())
    return out


def fk_tool(q):
    return T_E_0 @ robot_frames(q)[-1] @ T_6_H


def positioner(phi1, phi2):
    """^m0 T_m2, eq. (4)."""
    return dh(0, 0, phi1, 0) @ dh(-90 * D2R, 0, phi2, RM2)


def jacobian(q):
    """Basic Jacobian of O6 (wrist centre) in Sigma_0."""
    F = robot_frames(q)
    p6 = F[-1][:3, 3]
    J = np.zeros((6, 6))
    for i, T in enumerate(F):
        z, p = T[:3, 2], T[:3, 3]
        J[:3, i] = np.cross(z, p6 - p)
        J[3:, i] = z
    return J


def manipulabilities(q):
    J = jacobian(q)
    wt = abs(np.linalg.det(J[:3, :3])) / WT_MAX
    wr = abs(np.linalg.det(J[3:, 3:]))
    return wt, wr


# ------------------------------------------------------- time functions
def phi2(t):
    """Eq. (26), Table III."""
    rows = [(0.0, 2.5, 180.0, -12.0), (2.5, 35.0, 168.0, -336.0),
            (37.5, 2.5, -168.0, -12.0)]
    t0, T, p0, d = rows[0] if t < 2.5 else rows[1] if t < 37.5 else rows[2]
    s = min(max(t - t0, 0.0), T)
    if t0 == 0.0:
        v = p0 + d * (s / T - np.sin(np.pi * s / T) / np.pi)
    elif t0 == 2.5:
        v = p0 + d * s / T
    else:
        v = p0 + d * (s / T + np.sin(np.pi * s / T) / np.pi)
    return v * D2R


def psi(t):
    """Eq. (28), Table IV (cycloidal segments)."""
    rows = [(0, 4, 40.0, -58.4), (4, 16, -18.4, 16.0),
            (20, 16, -2.4, -12.0), (36, 4, -14.4, -76.6)]
    for t0, T, p0, d in rows:
        if t <= t0 + T or (t0, T) == (36, 4):
            s = min(max(t - t0, 0.0), T)
            return (p0 + d * (s / T - np.sin(2 * np.pi * s / T) / (2 * np.pi))) * D2R


def zyz(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([[ca * cb * cc - sa * sc, -ca * cb * sc - sa * cc, ca * sb],
                     [sa * cb * cc + ca * sc, -sa * cb * sc + ca * cc, sa * sb],
                     [-sb * cc, sb * sc, cb]])


def torch_on_workpiece(t):
    """^k T_h, eqs. (16)-(23)."""
    lam = 2 * np.pi * t / T_TASK
    p = [RT * np.cos(lam), RT * np.sin(lam), HT * t / T_TASK]
    return hom(zyz(np.pi + lam, 90 * D2R, psi(t)), p)


def desired_tool(t, phi1=PHI1):
    """^e T_h through the positioner, eq. (6)."""
    return T_E_M0 @ positioner(phi1, phi2(t)) @ T_M2_K @ torch_on_workpiece(t)


# ------------------------------------------------------------------ IK
def pose_error(T, Td):
    ep = Td[:3, 3] - T[:3, 3]
    Re = Td[:3, :3] @ T[:3, :3].T
    eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    return np.hstack([ep, eo])


def tool_jacobian(q, eps=1e-6):
    T0 = fk_tool(q)
    J = np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        J[:, i] = pose_error(T0, fk_tool(q + dq)) / eps
    return J


def ik(Td, q0, iters=200, tol=1e-9):
    q = np.array(q0, float)
    for _ in range(iters):
        e = pose_error(fk_tool(q), Td)
        if e @ e < tol:
            return q, True
        J = tool_jacobian(q)
        lam = 1e-4
        q = q + J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
    return q, e @ e < 1e-6


def first_posture(phi1=PHI1):
    """IK solution at t = 0 with the best manipulability (smallest joint angles on ties)."""
    rng = np.random.default_rng(0)
    best, best_w = None, -1
    Td = desired_tool(0.0, phi1)
    for _ in range(400):
        q0 = rng.uniform(-np.pi, np.pi, 6)
        q, ok = ik(Td, q0)
        if not ok:
            continue
        q = wrap(q)
        w = min(manipulabilities(q)) - 0.01 * np.linalg.norm(dh_to_urdf(q))
        if w > best_w:
            best, best_w = q, w
    return best


# ------------------------------------------------- DH -> URDF joints
# The ROS-Industrial M-10iA URDF uses Fanuc joint conventions. The signs
# and offsets below map the D-H angles of Table I to its joints; they were
# found by matching the forward kinematics of both models.
SIGN = np.array([1, -1, 1, -1, -1, -1])
OFFS = np.array([0, np.pi / 2, 0, np.pi, 0, np.pi])
# Frame 6 of Table I expressed in the URDF link_6 frame (flange, 0.1 m
# beyond the wrist centre): R_C maps URDF axes to D-H axes.
R_C = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0.0]])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def dh_to_urdf(q):
    return wrap(SIGN * (np.asarray(q) - OFFS))


def torch_in_flange():
    """^link_6 T_h for the URDF (torch tip)."""
    R = R_C.T @ T_6_H[:3, :3]
    p = R_C.T @ T_6_H[:3, 3] - np.array([0.100, 0, 0])
    return R, p


def mixed(wt, wr):
    """Mixed manipulability of one path point, eq. (25)."""
    pair = np.array([wt, wr])
    return pair.mean() - pair.std()


def objective(w):
    """Eq. (24): f = -(mean - std) of the mixed manipulabilities."""
    w = np.asarray(w)
    return -(w.mean() - w.std())


_Q0 = {}


def plan(n=801, phi1_fn=lambda t: PHI1):
    """Joint trajectories for the task. phi1_fn gives the tilt of the positioner."""
    t = np.linspace(0, T_TASK, n)
    if 'q' not in _Q0:
        _Q0['q'] = first_posture()
    q, ok = ik(desired_tool(0.0, phi1_fn(0.0)), _Q0['q'])   # same posture family as the paper
    if not ok:
        raise RuntimeError("IK failed at t=0")
    rows = []
    for ti in t:
        p1 = phi1_fn(ti)
        q, ok = ik(desired_tool(ti, p1), q)
        if not ok:
            raise RuntimeError(f"IK failed at t={ti:.2f}s")
        wt, wr = manipulabilities(q)
        rows.append([ti, p1, phi2(ti), psi(ti), *dh_to_urdf(q), wt, wr])
    return np.array(rows)


# ------------------------------------------- study: both positioner axes
KNOTS = np.linspace(0, T_TASK, 6)
PHI1_LIM = 45 * D2R


def phi1_spline(values):
    """Smooth tilt phi1(t) through the knot values, zero velocity at both ends."""
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(KNOTS, values, bc_type='clamped')
    return lambda t: float(np.clip(cs(t), -PHI1_LIM, PHI1_LIM))


def score(phi1_fn, n=81):
    """Objective (24) of a tilt function, on n samples of the task."""
    try:
        d = plan(n, phi1_fn)
    except RuntimeError:
        return 1.0
    if np.any(np.abs(np.diff(np.unwrap(d[:, 4:10], axis=0), axis=0)) > 0.4):  # change of posture
        return 1.0
    return objective([mixed(a, b) for a, b in zip(d[:, 10], d[:, 11])])


def optimize_two_axis():
    from scipy.optimize import minimize
    x0 = np.full(len(KNOTS), PHI1)
    res = minimize(lambda x: score(phi1_spline(x)), x0, method='Nelder-Mead',
                   options={'maxiter': 400, 'xatol': 1e-3, 'fatol': 1e-5})
    return res.x, res.fun


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    study = len(sys.argv) > 2 and sys.argv[2] == "two_axis"
    os.makedirs(out, exist_ok=True)
    hdr = "t,phi1,phi2,psi,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,wT,wR"
    if study:
        x, f = optimize_two_axis()
        print("phi1 knots [deg]:", np.round(np.degrees(x), 2), " f =", round(f, 4))
        data = plan(phi1_fn=phi1_spline(x))
        name = "trajectory_two_axis.csv"
    else:
        data = plan()
        name = "trajectory.csv"
    np.savetxt(os.path.join(out, name), data, delimiter=",", header=hdr, comments="")
    w = [mixed(a, b) for a, b in zip(data[:, 10], data[:, 11])]
    print(f"{name}: f = {objective(w):.4f}, min wT* = {data[:, 10].min():.3f}, "
          f"min wR = {data[:, 11].min():.3f}")


if __name__ == "__main__":
    main()
