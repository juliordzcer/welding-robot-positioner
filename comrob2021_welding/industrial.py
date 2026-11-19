"""Beyond the paper: an industrial job for the same robot and positioner.

NOT PART OF THE COMRob 2021 PAPER. This module reuses the robot, the
positioner, the inverse kinematics and the manipulability index of the paper
(planner.py) on a common industrial part: a flanged pipe spool with four
seams. It adds three applications:

1. Flat-position welding. For each seam the positioner keeps the joint in the
   flat position (bisector of the joint within FLAT_TOL of vertical, seam axis
   within FLAT_TOL of horizontal), and the tilt, the rotation offset and the
   torch angle are optimized to maximize the index of eq. (24) of the paper.
2. Seam sequencing. The order and direction of the seams are chosen to
   minimize the cycle time, with the axis speeds, retract and approach moves
   and arc times.
3. Robot placement. The robot pedestal is moved on a grid around its place in
   the paper and the place with the best manipulability for the whole job is
   found.

Usage:  python3 -m comrob2021_welding.industrial [output_dir] [--layout]
"""
import itertools
import json
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from . import planner as pl

D2R = np.pi / 180.0

# ------------------------------------------------------------ workpiece
# Flanged pipe spool on a riser held by the chuck. Frame k as in the paper
# (planner.T_M2_K): z = pipe axis, origin ZK above the faceplate.
R_PIPE, L_PIPE = 0.100, 0.450
R_FLANGE, T_FLANGE = 0.170, 0.020
R_RISER = 0.080
ZA, ZB = 0.050, L_PIPE - 0.050               # ends of the longitudinal seams (torch clears the flanges)
SOLIDS = [  # (radius, z_min, z_max) of coaxial cylinders in frame k
    (pl.PLATE_R, -pl.ZK - 0.035, -pl.ZK),
    (pl.CHUCK_R, -pl.ZK, -pl.ZK + pl.CHUCK_H),
    (R_RISER, -pl.ZK, -T_FLANGE),
    (R_FLANGE, -T_FLANGE, 0.0),
    (R_PIPE, 0.0, L_PIPE),
    (R_FLANGE, L_PIPE, L_PIPE + T_FLANGE),
]

# ---------------------------------------------------------- process data
V_WELD = 0.012          # travel speed [m/s] (0.72 m/min, robotic MAG)
TRAVEL_ANGLE = 10 * D2R  # push angle of the torch
FLAT_TOL = 15 * D2R     # flat position tolerance
GAP_MIN = 0.010         # torch / wrist to workpiece, faceplate and chuck [m]
TILT_LIM = 135 * D2R    # tilt range of the positioner
ROT_LIM = 6.28          # rotation range of the positioner [rad]
RETRACT = 0.10          # retract and approach distance along the torch [m]
T_RETRACT = 0.8         # time of one retract or approach move [s]
T_ARC_ON, T_ARC_OFF = 0.5, 0.8   # gas pre-flow and arc start; crater fill
# axis speeds [rad/s]: robot J1..J6 (Fanuc M-10iA data sheet), tilt, rotation
V_MAX = np.array([90, 150, 210, 190, 210, 400, 405, 615]) * D2R
A_MAX = V_MAX / 0.35    # reach full speed in 0.35 s
URDF_LIM = np.array([(-3.14, 3.14), (-1.57, 2.79), (-3.14, 4.61),
                     (-3.31, 3.31), (-3.31, 3.31), (-6.28, 6.28)])
HOME_URDF = np.array([0.0, -0.3, 0.0, 0.0, -1.2, 0.0])   # robot away from the part
HOME_POS = (0.0, 0.0)   # positioner at the load position (pipe vertical)


@dataclass
class Seam:
    name: str
    label: str
    kind: str           # 'circ' (around the pipe) or 'long' (along it)
    z: float = 0.0      # height of a circular seam
    sigma: float = 1.0  # +1: flange below the seam, -1: flange above
    theta: float = 0.0  # angle of a longitudinal seam

    @property
    def length(self):
        return 2 * np.pi * R_PIPE if self.kind == 'circ' else ZB - ZA

    @property
    def duration(self):
        return self.length / V_WELD


SEAMS = [
    Seam('A', 'lower flange fillet', 'circ', z=0.0, sigma=1.0),
    Seam('B', 'upper flange fillet', 'circ', z=L_PIPE, sigma=-1.0),
    Seam('C', 'longitudinal seam 1', 'long', theta=0.0),
    Seam('D', 'longitudinal seam 2', 'long', theta=np.pi),
]


def seam_geometry(seam, s, d):
    """Point, joint normal (bisector) and travel direction in frame k; s in [0, 1], d = +-1."""
    if seam.kind == 'circ':
        th = d * 2 * np.pi * s
        c, sn = np.cos(th), np.sin(th)
        return (np.array([R_PIPE * c, R_PIPE * sn, seam.z]),
                np.array([c, sn, seam.sigma]) / np.sqrt(2), d * np.array([-sn, c, 0.0]))
    z = ZA + (ZB - ZA) * (s if d > 0 else 1 - s)
    c, sn = np.cos(seam.theta), np.sin(seam.theta)
    return np.array([R_PIPE * c, R_PIPE * sn, z]), np.array([c, sn, 0.0]), np.array([0, 0, float(d)])


def torch_frame(p, n, tv, psi):
    """^k T_h: torch axis into the joint with a push angle, turned psi about its axis."""
    a = -n * np.cos(TRAVEL_ANGLE) + tv * np.sin(TRAVEL_ANGLE)
    a /= np.linalg.norm(a)
    x0 = tv - (tv @ a) * a
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(a, x0)
    x = np.cos(psi) * x0 + np.sin(psi) * y0
    return pl.hom(np.column_stack([x, np.cross(a, x), a]), p)


def phi2_of(seam, c, s, d):
    """Circular seams: the positioner turns the pipe so that the weld point stays still."""
    return c - d * 2 * np.pi * s if seam.kind == 'circ' else c


def workpiece_frame(phi1, phi2):
    return pl.T_E_M0 @ pl.positioner(phi1, phi2) @ pl.T_M2_K


def base_frame(base):
    """Pose of the robot pedestal in Sigma_e: an (x, y) offset or a 4x4 matrix."""
    if np.ndim(base) == 2:
        return base
    return pl.hom(np.eye(3), [base[0], base[1], 0.0])


def psi_fn(knots):
    from scipy.interpolate import CubicSpline
    return CubicSpline([0.0, 0.5, 1.0], knots, bc_type='clamped')


def flat_error(seam, phi1, c, d, s=0.0):
    """Angle of the joint bisector from vertical and of the seam axis from horizontal."""
    _, n, tv = seam_geometry(seam, s, d)
    R = workpiece_frame(phi1, phi2_of(seam, c, s, d))[:3, :3]
    return np.arccos(np.clip((R @ n)[2], -1, 1)), np.arcsin(min(1.0, abs((R @ tv)[2])))


def gap(q, phi1, phi2, base, skip_tip=True):
    """Smallest distance [m] from the wrist and torch to the part, riser, chuck and faceplate."""
    Tk = np.linalg.inv(workpiece_frame(phi1, phi2)) @ base_frame(base)
    pts = pl.torch_points(q)
    if skip_tip:
        pts = pts[:-1]          # neck point next to the arc
    g = np.inf
    for p, r in pts:
        x, y, z = (Tk @ np.append(p, 1.0))[:3]
        rho = np.hypot(x, y)
        for R, z0, z1 in SOLIDS:
            dr, dz = rho - R, max(z0 - z, z - z1)
            if dr <= 0 and dz <= 0:
                dist = max(dr, dz)
            else:
                dist = np.hypot(max(dr, 0), max(dz, 0))
            g = min(g, dist - r)
    return g


def urdf_ok(u):
    return np.all(u >= URDF_LIM[:, 0]) and np.all(u <= URDF_LIM[:, 1])


# ------------------------------------------------- 1. flat-position seam
def plan_seam(seam, d, x, base, q0, n=21, dt=None):
    """Joint trajectory of one seam. x = [phi1, c, psi0, psi1, psi2]."""
    T = seam.duration
    s_all = np.arange(0, T + 1e-9, dt) / T if dt else np.linspace(0, 1, n)
    psi = psi_fn(x[2:5])
    Binv = np.linalg.inv(base_frame(base))
    q = np.array(q0, float)
    rows = []
    for s in s_all:
        p, nk, tv = seam_geometry(seam, s, d)
        ph2 = phi2_of(seam, x[1], s, d)
        Td = workpiece_frame(x[0], ph2) @ torch_frame(p, nk, tv, psi(s))
        q, ok = pl.ik(Binv @ Td, q)
        if not ok:
            raise RuntimeError(f'IK failed, seam {seam.name}, s = {s:.2f}')
        wt, wr = pl.manipulabilities(q)
        dev, inc = flat_error(seam, x[0], x[1], d, s)
        rows.append(dict(t=s * T, phi1=x[0], phi2=ph2, psi=float(psi(s)), q=q.copy(),
                         u=pl.dh_to_urdf(q), wt=wt, wr=wr, gap=gap(q, x[0], ph2, base),
                         dev=dev, inc=inc, pk=p, nk=nk))
    return rows


def score(seam, d, x, base, q0, n=21):
    """Objective (24) of the paper; values above 1 mean the plan breaks a rule."""
    if abs(x[0]) > TILT_LIM:
        return 1.0 + abs(x[0])
    try:
        rows = plan_seam(seam, d, x, base, q0, n)
    except RuntimeError:
        return 10.0
    U = np.unwrap(np.array([r['u'] for r in rows]), axis=0)
    if np.any(np.abs(np.diff(U, axis=0)) > 0.4) or not all(urdf_ok(r['u']) for r in rows):
        return 5.0
    pen = sum(max(0.0, max(r['dev'], r['inc']) - FLAT_TOL) for r in rows) / len(rows)
    pen += max(0.0, GAP_MIN - min(r['gap'] for r in rows)) * 20
    if pen > 0:
        return 1.0 + pen
    return pl.objective([pl.mixed(r['wt'], r['wr']) for r in rows])


def aimed_seeds(Tb):
    """Starting postures with joint 1 turned toward the target (Tb in the robot base frame)."""
    j1 = np.arctan2(Tb[1, 3], Tb[0, 3])
    out = []
    for j2, j3, j4, j5 in itertools.product((0.0, 0.6), (-0.6, 0.3), (-1.5, 0.0, 1.5), (-1.2, 1.2)):
        out.append(pl.OFFS + pl.SIGN * np.array([j1, j2, j3, j4, j5, 0.0]))
    return out


def best_posture(Td, base, phi1, phi2, seeds):
    """IK solution with the best manipulability among several starting points."""
    Binv = np.linalg.inv(base_frame(base))
    best, best_w = None, -np.inf
    for q0 in aimed_seeds(Binv @ Td) + list(seeds):
        q, ok = pl.ik(Binv @ Td, q0, iters=80)      # cold starts: give up early
        if not ok or not urdf_ok(pl.dh_to_urdf(q)) or gap(q, phi1, phi2, base) < GAP_MIN:
            continue
        w = min(pl.manipulabilities(q))
        if w > best_w:
            best, best_w = q, w
    return best


def optimize_seam(seam, d, base=(0.0, 0.0), maxiter=300, n=21, seed=0):
    """Tilt, rotation offset and torch angle that keep the seam flat with the best index."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    seeds = [pl.first_posture()] + list(rng.uniform(-np.pi, np.pi, (15, 6)))
    starts = []
    for p1 in np.arange(-135, 136, 7.5) * D2R:
        for c in np.arange(0, 360, 15) * D2R:
            e = max(max(flat_error(seam, p1, c, d, s)) for s in (0.0, 1.0))
            if e <= FLAT_TOL:
                starts.append((e, p1, c))
    starts.sort()
    picked = []                                   # distinct flat configurations
    for e, p1, c in starts:
        if all(abs(p1 - a) > 20 * D2R or abs(pl.wrap(c - b)) > 20 * D2R for a, b in picked):
            picked.append((p1, c))
    cands = []
    for p1, c in picked[:6]:
        for ps in np.arange(0, 360, 45) * D2R:
            x = np.array([p1, c, ps, ps, ps])
            p, nk, tv = seam_geometry(seam, 0.0, d)
            Td = workpiece_frame(p1, phi2_of(seam, c, 0, d)) @ torch_frame(p, nk, tv, ps)
            q0 = best_posture(Td, base, p1, phi2_of(seam, c, 0, d), seeds)
            if q0 is not None:
                cands.append((score(seam, d, x, base, q0, 11), x, q0))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    best = None
    for f0, x0, q0 in cands[:2]:
        simplex = np.vstack([x0, x0 + np.diag(np.array([5, 5, 30, 30, 30]) * D2R)])
        res = minimize(lambda x: score(seam, d, x, base, q0, n), x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 1e-5,
                                'initial_simplex': simplex})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x, q0)
    return {'seam': seam.name, 'd': d, 'f': float(best[0]), 'x': best[1], 'q0': best[2]}


# ----------------------------------------------------- 2. sequencing
def ptp_time(a, b):
    """Duration of a synchronized point-to-point move (cycloidal profile) between axis vectors."""
    D = np.abs(np.asarray(b) - np.asarray(a))
    return float(np.max(np.maximum(2 * D / V_MAX, np.sqrt(2 * np.pi * D / A_MAX))))


def retracted(q, phi1, phi2, base):
    """Robot joints with the torch moved RETRACT back along its axis."""
    T = base_frame(base) @ pl.fk_tool(q)
    T[:3, 3] -= RETRACT * T[:3, 2]
    qr, ok = pl.ik(np.linalg.inv(base_frame(base)) @ T, q)
    if not ok:
        raise RuntimeError('retract IK failed')
    return qr


def axes(u, phi1, phi2):
    """Axis vector in the order of V_MAX: tilt, rotation, J1..J6."""
    return np.concatenate([[phi1, phi2], u])


def nearest(target, ref, period=2 * np.pi):
    return target + period * np.round((ref - target) / period)


def fit(U, ref=None):
    """Shift each robot joint by multiples of 2*pi so the path stays within the URDF limits,
    as close as possible to ref (a joint vector)."""
    U = np.array(U, float)
    one = U.ndim == 1
    U = np.atleast_2d(U)
    for j, (lo, hi) in enumerate(URDF_LIM):
        ks = [k for k in range(-2, 3)
              if U[:, j].min() + 2 * np.pi * k >= lo - 1e-6 and U[:, j].max() + 2 * np.pi * k <= hi + 1e-6]
        if ks:
            k = min(ks, key=lambda k: abs(U[0, j] + 2 * np.pi * k - (ref[j] if ref is not None else 0.0)))
            U[:, j] += 2 * np.pi * k
    return U[0] if one else U


def ptp_clear(a, b, base, n=15):
    """True if the torch stays away from the part during the move a -> b (axis vectors)."""
    for s in np.linspace(0, 1, n):
        v = a + (b - a) * s
        q = pl.OFFS + pl.SIGN * v[2:]
        if gap(q, v[0], v[1], base, skip_tip=False) < GAP_MIN:
            return False
    return True


def seam_ends(plan, base):
    """Retracted axis vectors at the start and end of a planned seam."""
    out = []
    for r in (plan[0], plan[-1]):
        qr = retracted(r['q'], r['phi1'], r['phi2'], base)
        out.append(axes(pl.dh_to_urdf(qr), r['phi1'], r['phi2']))
    return out


def move(a, b, base):
    """Time and route of a move between two retracted states (via home if the direct move collides)."""
    b = b.copy()
    b[1] = nearest(b[1], a[1])
    b[2:] = fit(b[2:], a[2:])
    if ptp_clear(a, b, base):
        return ptp_time(a, b), [b]
    h = axes(HOME_URDF, *HOME_POS)
    h[1] = nearest(h[1], a[1])
    b2 = b.copy()
    b2[1] = nearest(b[1], h[1])
    return ptp_time(a, h) + ptp_time(h, b2), [h, b2]


def sequence(plans, base, order=None):
    """Best order and direction of the seams. plans[(name, d)] -> planned seam."""
    home = axes(HOME_URDF, *HOME_POS)
    ends = {k: seam_ends(p, base) for k, p in plans.items()}
    fixed = sum(s.duration + T_ARC_ON + T_ARC_OFF + 2 * T_RETRACT for s in SEAMS)
    names = [s.name for s in SEAMS]
    best = None
    perms = [order] if order else itertools.permutations(names)
    for perm in perms:
        for dirs in itertools.product([1, -1], repeat=len(perm)):
            if order and dirs != tuple([1] * len(perm)):
                continue
            keys = [(nm, d) for nm, d in zip(perm, dirs)]
            if any(k not in plans for k in keys):
                continue
            t, cur = 0.0, home
            for k in keys:
                dt, _ = move(cur, ends[k][0], base)
                t += dt
                cur = ends[k][1]
            t += move(cur, home, base)[0]
            if best is None or t < best[0]:
                best = (t, keys)
    return {'cycle_s': fixed + best[0], 'moves_s': best[0], 'order': best[1]}


# ----------------------------------------------------- timeline for Gazebo
def cyc(s):
    return s - np.sin(2 * np.pi * s) / (2 * np.pi)


def timeline(plans, keys, base, dt=0.05):
    """Samples of the whole cycle: columns of the trajectory file."""
    rows = []
    t = 0.0
    cur = axes(HOME_URDF, *HOME_POS)
    names = [s.name for s in SEAMS]

    def add(v, arc=0, seam=-1, pk=(0, 0, 0), nk=(0, 0, 0), psi=0.0, wt=None, wr=None, g=None):
        q = pl.OFFS + pl.SIGN * v[2:]
        if wt is None:
            wt, wr = pl.manipulabilities(q)
            g = gap(q, v[0], v[1], base, skip_tip=False)
        rows.append([t, v[0], v[1], psi, *v[2:], wt, wr, g, arc, seam, *pk, *nk])

    def ptp(a, b):
        nonlocal t
        T = ptp_time(a, b)
        for s in np.arange(dt, T + 1e-9, dt) / max(T, 1e-9):
            t += dt
            add(a + (b - a) * cyc(min(s, 1.0)))
        return b

    def dwell(v, T):
        nonlocal t
        for _ in range(int(round(T / dt))):
            t += dt
            add(v)

    add(cur)
    for k in keys:
        plan = plans[k]
        ret_start, _ = seam_ends(plan, base)
        _, route = move(cur, ret_start, base)
        for b in route:
            cur = ptp(cur, b)
        # weld trajectory, continuous with the current axes
        U = np.unwrap(np.array([r['u'] for r in plan]), axis=0)
        P2 = np.array([r['phi2'] for r in plan])
        P2 = P2 + (nearest(P2[0], cur[1]) - P2[0])
        U = fit(U, cur[2:])
        first = axes(U[0], plan[0]['phi1'], P2[0])
        cur = ptp(cur, first)                          # approach
        dwell(cur, T_ARC_ON)
        si = names.index(k[0])
        t0 = t
        for r, u, p2 in zip(plan, U, P2):
            t = t0 + r['t']
            v = axes(u, r['phi1'], p2)
            add(v, 1, si, r['pk'], r['nk'], r['psi'], r['wt'], r['wr'], r['gap'])
        cur = axes(U[-1], plan[-1]['phi1'], P2[-1])
        dwell(cur, T_ARC_OFF)
        qr = retracted(plan[-1]['q'], plan[-1]['phi1'], plan[-1]['phi2'], base)
        back = axes(fit(pl.dh_to_urdf(qr), cur[2:]), cur[0], cur[1])
        cur = ptp(cur, back)                           # retract
    home = axes(HOME_URDF, *HOME_POS)
    _, route = move(cur, home, base)
    for b in route:
        cur = ptp(cur, b)
    return np.array(rows)


HEADER = ('t,phi1,phi2,psi,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,wT,wR,clearance,'
          'arc,seam,pk_x,pk_y,pk_z,nk_x,nk_y,nk_z')


# ------------------------------------------------------- 3. placement
def _layout_job(args):
    base, seam_name = args
    seam = next(s for s in SEAMS if s.name == seam_name)
    r = optimize_seam(seam, 1, base, maxiter=120, n=13)
    return base, seam_name, (r['f'] if r else 10.0)


def layout(step=0.10, span=0.30, workers=None):
    """Mean objective of the four seams for robot places on a grid around the paper's place."""
    g = np.round(np.arange(-span, span + 1e-9, step), 3)
    jobs = [((float(bx), float(by)), s.name) for bx in g for by in g for s in SEAMS]
    with Pool(workers) as p:
        out = p.map(_layout_job, jobs)
    grid = {}
    for base, _, f in out:
        grid.setdefault(base, []).append(f)
    return {b: (float(np.mean(fs)) if max(fs) < 1 else np.nan) for b, fs in grid.items()}


def _seam_job(args):
    name, d, base = args
    seam = next(s for s in SEAMS if s.name == name)
    return optimize_seam(seam, d, base)


def job(base, workers=None):
    """Applications 1 and 2 for one robot place."""
    with Pool(workers) as p:
        res = p.map(_seam_job, [(s.name, d, base) for s in SEAMS for d in (1, -1)])
    plans, info = {}, {}
    for r in res:
        if r is None or r['f'] >= 1:
            continue
        seam = next(s for s in SEAMS if s.name == r['seam'])
        plans[(r['seam'], r['d'])] = plan_seam(seam, r['d'], r['x'], base, r['q0'], dt=0.05)
        rows = plans[(r['seam'], r['d'])]
        info[f"{r['seam']}{'+' if r['d'] > 0 else '-'}"] = {
            'f': round(pl.objective([pl.mixed(a['wt'], a['wr']) for a in rows]), 4),
            'tilt_deg': round(float(np.degrees(r['x'][0])), 1),
            'flat_dev_max_deg': round(float(np.degrees(max(max(a['dev'], a['inc']) for a in rows))), 1),
            'wT_min': round(min(a['wt'] for a in rows), 3), 'wR_min': round(min(a['wr'] for a in rows), 3),
            'gap_min_mm': round(1000 * min(a['gap'] for a in rows), 0)}
    best = sequence(plans, base)
    naive = sequence(plans, base, order=tuple(s.name for s in SEAMS))
    return plans, info, best, naive


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = args[0] if args else 'config/industrial'
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'summary.json')
    summary = json.load(open(path)) if os.path.exists(path) else {}
    base = tuple(summary.get('layout', {}).get('best_place', (0.0, 0.0)))
    if '--layout' in sys.argv:
        grid = layout()
        rows = [[bx, by, f] for (bx, by), f in sorted(grid.items())]
        np.savetxt(os.path.join(out, 'layout.csv'), rows, delimiter=',', header='dx,dy,f', comments='')
        ok = {b: f for b, f in grid.items() if np.isfinite(f)}
        base = min(ok, key=ok.get)
        summary['layout'] = {'paper_place_f': grid.get((0.0, 0.0)), 'best_place': base, 'best_f': ok[base]}
        print('layout:', summary['layout'])
    for tag, b in [('paper_place', (0.0, 0.0)), ('best_place', base)]:
        plans, info, best, naive = job(b)
        data = timeline(plans, best['order'], b)
        naive_t = timeline(plans, naive['order'], b)[-1, 0]
        np.savetxt(os.path.join(out, f'trajectory_{tag}.csv'), data, delimiter=',', header=HEADER,
                   comments='', fmt='%.6g')
        summary[tag] = {'base_offset_m': b, 'seams': info,
                        'best_order': [f"{n}{'+' if d > 0 else '-'}" for n, d in best['order']],
                        'cycle_s': round(float(data[-1, 0]), 1), 'naive_cycle_s': round(float(naive_t), 1),
                        'arc_time_s': round(sum(s.duration for s in SEAMS), 1)}
        print(tag, json.dumps(summary[tag], indent=1))
    with open(path, 'w') as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == '__main__':
    main()
