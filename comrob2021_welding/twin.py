"""Beyond the paper: two robots welding at the same time on one positioner.

NOT PART OF THE COMRob 2021 PAPER. A second ARC Mate 100iC faces the first
one across the positioner, and both weld the flanged spool of industrial.py
at the same time:

  task AB: the two flange fillets. The positioner turns the pipe for both, so
           both weld points stay still; the start point of seam B is free.
  task CD: the two longitudinal seams, with the pipe still.

The seams may be welded in the flat or the horizontal position, never
vertical or overhead (AWS A3.0: seam axis within 15 deg of horizontal; face
rotation within 55 deg of face-up for fillets and 100 deg for grooves). The
shared positioner motion, the torch angles, which robot welds which seam and
the place of robot 2 are optimized for the worse of the two robots in the
index (24) of the paper.

Usage:  python3 -m comrob2021_welding.twin [output_dir]
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from . import industrial as ind
from . import planner as pl

D2R = np.pi / 180.0
DEV_MAX = {'circ': 55 * D2R, 'long': 100 * D2R}   # face rotation from face-up
INC_MAX = 15 * D2R
ROBOT_GAP = 0.10        # between the wrists and torches of the two robots [m]
TASKS = {'AB': ('A', 'B'), 'CD': ('C', 'D')}
SEAM = {s.name: s for s in ind.SEAMS}


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def base2(place):
    """Robot 2 pedestal: (x, y) in Sigma_e, turned 180 deg to face robot 1."""
    return pl.hom(rz(np.pi), [place[0], place[1], 0.0])


def geometry(seam, s, d, offset):
    p, n, tv = ind.seam_geometry(seam, s, d)
    R = rz(offset)
    return R @ p, R @ n, R @ tv


def unpack(task, x):
    """x -> phi1, c, start offset of the second seam, torch knots of both robots."""
    if task == 'AB':
        return x[0], x[1], x[2], x[3:6], x[6:9]
    return x[0], x[1], 0.0, x[2:5], x[5:8]


def plan_pair(task, assign, dirs, x, place, q0s, n=21, dt=None):
    """Joint trajectories of both robots. assign: seam of robot 1 and robot 2."""
    phi1, c, off, k1, k2 = unpack(task, x)
    seams = [SEAM[a] for a in assign]
    T = seams[0].duration
    s_all = np.arange(0, T + 1e-9, dt) / T if dt else np.linspace(0, 1, n)
    bases = [ind.base_frame((0.0, 0.0)), base2(place)]
    offs = [0.0, off] if assign[0] == TASKS[task][0] else [off, 0.0]
    out = []
    for r, (seam, d, B, knots, q0, o) in enumerate(zip(seams, dirs, bases, (k1, k2), q0s, offs)):
        psi = ind.psi_fn(knots)
        Binv = np.linalg.inv(B)
        q = np.array(q0, float)
        rows = []
        for s in s_all:
            p, nk, tv = geometry(seam, s, d, o)
            ph2 = ind.phi2_of(seam, c, s, d)
            W = ind.workpiece_frame(phi1, ph2)
            q, ok = pl.ik(Binv @ W @ ind.torch_frame(p, nk, tv, psi(s)), q)
            if not ok:
                raise RuntimeError('IK failed')
            wt, wr = pl.manipulabilities(q)
            nw, tw = W[:3, :3] @ nk, W[:3, :3] @ tv
            rows.append(dict(t=s * T, phi1=phi1, phi2=ph2, psi=float(psi(s)), q=q.copy(),
                             u=pl.dh_to_urdf(q), wt=wt, wr=wr, gap=ind.gap(q, phi1, ph2, B),
                             dev=np.arccos(np.clip(nw[2], -1, 1)), inc=np.arcsin(min(1, abs(tw[2]))),
                             pk=p, nk=nk, seam=seam, B=B))
        out.append(rows)
    return out


def robots_gap(qa, Ba, qb, Bb):
    """Smallest distance between the wrists and torches of two robots."""
    pa = [(Ba @ np.append(p, 1))[:3] for p, _ in pl.torch_points(qa)]
    pb = [(Bb @ np.append(p, 1))[:3] for p, _ in pl.torch_points(qb)]
    ra = [r for _, r in pl.torch_points(qa)]
    rb = [r for _, r in pl.torch_points(qb)]
    return min(np.linalg.norm(a - b) - u - v for a, u in zip(pa, ra) for b, v in zip(pb, rb))


def score(task, assign, dirs, x, place, q0s, n=13):
    if abs(x[0]) > ind.TILT_LIM:
        return 1.0 + abs(x[0])
    try:
        plans = plan_pair(task, assign, dirs, x, place, q0s, n)
    except RuntimeError:
        return 10.0
    pen, fs = 0.0, []
    for rows in plans:
        U = np.unwrap(np.array([r['u'] for r in rows]), axis=0)
        if np.any(np.abs(np.diff(U, axis=0)) > 0.4) or not all(ind.urdf_ok(r['u']) for r in rows):
            return 5.0
        dmax = DEV_MAX[rows[0]['seam'].kind]
        pen += sum(max(0, r['dev'] - dmax) + max(0, r['inc'] - INC_MAX) for r in rows) / len(rows)
        pen += max(0.0, ind.GAP_MIN - min(r['gap'] for r in rows)) * 20
        fs.append(pl.objective([pl.mixed(r['wt'], r['wr']) for r in rows]))
    g = min(robots_gap(a['q'], a['B'], b['q'], b['B']) for a, b in zip(*plans))
    pen += max(0.0, ROBOT_GAP - g) * 20
    return 1.0 + pen if pen > 0 else max(fs)


def optimize(task, assign, dirs, place, maxiter=400, n=13, seed=0):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    seeds = [pl.first_posture()] + list(rng.uniform(-np.pi, np.pi, (15, 6)))
    seams = [SEAM[a] for a in assign]
    bases = [ind.base_frame((0.0, 0.0)), base2(place)]
    cands = []
    offsets = np.arange(0, 360, 45) * D2R if task == 'AB' else [0.0]
    for p1 in np.arange(-135, 136, 15) * D2R:
        for c in np.arange(0, 360, 30) * D2R:
            for off in offsets:
                x = [p1, c, off] if task == 'AB' else [p1, c]
                xx = np.r_[x, np.zeros(6)]
                _, _, o, _, _ = unpack(task, xx)
                offs = [0.0, o] if assign[0] == TASKS[task][0] else [o, 0.0]
                ok, q0s, knots = True, [], []
                for seam, d, B, of in zip(seams, dirs, bases, offs):
                    W = ind.workpiece_frame(p1, ind.phi2_of(seam, c, 0, d))
                    p, nk, tv = geometry(seam, 0.0, d, of)
                    nw, tw = W[:3, :3] @ nk, W[:3, :3] @ tv
                    if np.arccos(np.clip(nw[2], -1, 1)) > DEV_MAX[seam.kind] or \
                            np.arcsin(min(1, abs(tw[2]))) > INC_MAX:
                        ok = False
                        break
                    best = None
                    for ps in np.arange(0, 360, 60) * D2R:
                        q = ind.best_posture(W @ ind.torch_frame(p, nk, tv, ps), B, p1,
                                             ind.phi2_of(seam, c, 0, d), seeds[:6])
                        if q is not None:
                            w = min(pl.manipulabilities(q))
                            if best is None or w > best[0]:
                                best = (w, ps, q)
                    if best is None:
                        ok = False
                        break
                    q0s.append(best[2])
                    knots += [best[1]] * 3
                if ok:
                    cands.append((np.r_[x, knots], q0s))
    if not cands:
        return None
    scored = sorted((score(task, assign, dirs, x, place, q0s, 7), i) for i, (x, q0s) in enumerate(cands))
    best = None
    for f0, i in scored[:3]:
        x0, q0s = cands[i]
        steps = np.r_[[5, 5], [20] * (len(x0) - 2)] * D2R
        res = minimize(lambda x: score(task, assign, dirs, x, place, q0s, n), x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 1e-5,
                                'initial_simplex': np.vstack([x0, x0 + np.diag(steps)])})
        if best is None or res.fun < best['f']:
            best = {'f': float(res.fun), 'x': res.x, 'q0s': q0s}
    best.update(task=task, assign=assign, dirs=dirs, place=place)
    return best


def _job(args):
    return optimize(*args)


def variants(task):
    a, b = TASKS[task]
    return [(task, assign, dirs) for assign in ((a, b), (b, a))
            for dirs in itertools.product([1, -1], repeat=2)
            if task == 'CD' or dirs[0] == dirs[1]]          # one turn of the pipe for both fillets


def search(places, maxiter, n, workers=None):
    jobs = [(*v, place, maxiter, n) for place in places for task in TASKS for v in variants(task)]
    with Pool(workers) as p:
        res = p.map(_job, jobs)
    return [r for r in res if r is not None and r['f'] < 1]


# ------------------------------------------------------------ timeline
def state(rows1, rows2, i):
    a, b = rows1[i], rows2[i]
    return np.r_[a['phi1'], a['phi2'], a['u'], b['u']]


def retract_state(rows1, rows2, i):
    a, b = rows1[i], rows2[i]
    q1 = ind.retracted(a['q'], a['phi1'], a['phi2'], a['B'])
    q2 = ind.retracted(b['q'], b['phi1'], b['phi2'], b['B'])
    return np.r_[a['phi1'], a['phi2'], pl.dh_to_urdf(q1), pl.dh_to_urdf(q2)]


V2 = np.r_[ind.V_MAX, ind.V_MAX[2:]]
A2 = np.r_[ind.A_MAX, ind.A_MAX[2:]]
HOME2 = np.r_[ind.HOME_POS, ind.HOME_URDF, ind.HOME_URDF]


def ptp_time(a, b):
    D = np.abs(b - a)
    return float(np.max(np.maximum(2 * D / V2, np.sqrt(2 * np.pi * D / A2))))


def clear(a, b, B2, n=15):
    for s in np.linspace(0, 1, n):
        v = a + (b - a) * s
        q1, q2 = pl.OFFS + pl.SIGN * v[2:8], pl.OFFS + pl.SIGN * v[8:]
        if (ind.gap(q1, v[0], v[1], (0.0, 0.0), skip_tip=False) < ind.GAP_MIN
                or ind.gap(q2, v[0], v[1], B2, skip_tip=False) < ind.GAP_MIN
                or robots_gap(q1, np.eye(4), q2, B2) < ROBOT_GAP):
            return False
    return True


def align(b, a):
    b = b.copy()
    b[1] = ind.nearest(b[1], a[1])
    b[2:8], b[8:] = ind.fit(b[2:8], a[2:8]), ind.fit(b[8:], a[8:])
    return b


def route(a, b, B2):
    b = align(b, a)
    if clear(a, b, B2):
        return [b]
    h = align(HOME2, a)
    return [h, align(b, h)]


HEADER = (ind.HEADER + ',' + ','.join(f'r2_joint_{i}' for i in range(1, 7))
          + ',wT2,wR2,clearance2,arc2,seam2,pk2_x,pk2_y,pk2_z,nk2_x,nk2_y,nk2_z')


def timeline(plans, order, place, dt=0.05):
    B2 = base2(place)
    names = [s.name for s in ind.SEAMS]
    rows, t = [], 0.0
    cur = HOME2.copy()

    def add(v, w=None):
        q1, q2 = pl.OFFS + pl.SIGN * v[2:8], pl.OFFS + pl.SIGN * v[8:]
        if w is None:
            m1, m2 = pl.manipulabilities(q1), pl.manipulabilities(q2)
            g1 = ind.gap(q1, v[0], v[1], (0.0, 0.0), skip_tip=False)
            g2 = ind.gap(q2, v[0], v[1], B2, skip_tip=False)
            w = [(*m1, g1, 0, -1, 0, 0, 0, 0, 0, 0), (*m2, g2, 0, -1, 0, 0, 0, 0, 0, 0)]
        rows.append([t, v[0], v[1], 0.0, *v[2:8], *w[0], *v[8:], *w[1]])

    def ptp(a, b):
        nonlocal t
        T = ptp_time(a, b)
        for s in np.arange(dt, T + 1e-9, dt) / max(T, 1e-9):
            t += dt
            add(a + (b - a) * ind.cyc(min(s, 1.0)))
        return b

    def dwell(v, T):
        nonlocal t
        for _ in range(int(round(T / dt))):
            t += dt
            add(v)

    add(cur)
    for task in order:
        r1, r2 = plans[task]
        for b in route(cur, retract_state(r1, r2, 0), B2):
            cur = ptp(cur, b)
        U = np.unwrap(np.array([np.r_[a['phi2'], a['u'], b['u']] for a, b in zip(r1, r2)]), axis=0)
        U[:, 0] += ind.nearest(U[0, 0], cur[1]) - U[0, 0]
        U[:, 1:7], U[:, 7:] = ind.fit(U[:, 1:7], cur[2:8]), ind.fit(U[:, 7:], cur[8:])
        first = np.r_[r1[0]['phi1'], U[0]]
        cur = ptp(cur, first)
        dwell(cur, ind.T_ARC_ON)
        t0 = t
        for a, b, u in zip(r1, r2, U):
            t = t0 + a['t']
            w = [(a['wt'], a['wr'], a['gap'], 1, names.index(a['seam'].name), *a['pk'], *a['nk']),
                 (b['wt'], b['wr'], b['gap'], 1, names.index(b['seam'].name), *b['pk'], *b['nk'])]
            add(np.r_[a['phi1'], u], w)
        cur = np.r_[r1[-1]['phi1'], U[-1]]
        dwell(cur, ind.T_ARC_OFF)
        cur = ptp(cur, align(retract_state(r1, r2, -1), cur))
    for b in route(cur, HOME2, B2):
        cur = ptp(cur, b)
    return np.array(rows)


def place_study(places):
    """Robot 2 place with the best worst-case index over both tasks."""
    res = search(places, maxiter=150, n=9)
    by_place = {}
    for r in res:
        k = (r['place'], r['task'])
        if k not in by_place or r['f'] < by_place[k]['f']:
            by_place[k] = r
    score_place = {p: max(by_place[(p, t)]['f'] for t in TASKS) for p in places
                   if all((p, t) in by_place for t in TASKS)}
    place = min(score_place, key=score_place.get)
    print('robot 2 place:', place, 'worst f:', round(score_place[place], 4))
    return place, score_place


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = args[0] if args else 'config/industrial'
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'summary.json')
    allsum = json.load(open(path)) if os.path.exists(path) else {}
    old = allsum.get('twin', {})
    if '--reuse-place' in sys.argv and 'robot2_place' in old:
        place = tuple(old['robot2_place'])
        score_place = {tuple(map(float, k.split(','))): f for k, f in old['places'].items()}
    else:
        place, score_place = place_study(
            [(x, y) for x in (2.2, 2.4, 2.6) for y in (0.0, 0.25, 0.5, 0.75, 1.0)])
    final = search([place], maxiter=500, n=21)
    plans, info = {}, {}
    for task in TASKS:
        r = min((x for x in final if x['task'] == task), key=lambda x: x['f'])
        p1, p2 = plan_pair(task, r['assign'], r['dirs'], r['x'], place, r['q0s'], dt=0.05)
        plans[task] = (p1, p2)
        info[task] = {
            'robot1': r['assign'][0], 'robot2': r['assign'][1],
            'f_robot1': round(pl.objective([pl.mixed(a['wt'], a['wr']) for a in p1]), 4),
            'f_robot2': round(pl.objective([pl.mixed(a['wt'], a['wr']) for a in p2]), 4),
            'tilt_deg': round(float(np.degrees(r['x'][0])), 1),
            'face_rotation_max_deg': [round(float(np.degrees(max(a['dev'] for a in p))), 1) for p in (p1, p2)],
            'gap_min_mm': round(1000 * min(a['gap'] for a in p1 + p2), 0),
            'robots_gap_min_mm': round(1000 * min(robots_gap(a['q'], a['B'], b['q'], b['B'])
                                                   for a, b in zip(p1, p2)), 0)}
    best = None
    for order in (('AB', 'CD'), ('CD', 'AB')):
        data = timeline(plans, order, place)
        if best is None or data[-1, 0] < best[1][-1, 0]:
            best = (order, data)
    order, data = best
    np.savetxt(os.path.join(out, 'trajectory_twin.csv'), data, delimiter=',', header=HEADER,
               comments='', fmt='%.6g')
    summary = {'robot2_place': place, 'tasks': info, 'order': order,
               'cycle_s': round(float(data[-1, 0]), 1),
               'places': {f'{p[0]},{p[1]}': round(f, 4) for p, f in score_place.items()}}
    allsum = json.load(open(path)) if os.path.exists(path) else {}
    allsum['twin'] = summary
    with open(path, 'w') as f:
        json.dump(allsum, f, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
