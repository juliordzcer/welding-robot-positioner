"""Figures of the planned task: positioner and torch angles, robot joints and manipulabilities.

Usage:  python3 -m comrob2021_welding.plots [trajectory.csv] [output.png]
        python3 -m comrob2021_welding.plots compare
        python3 -m comrob2021_welding.plots industrial
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else 'config/trajectory.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'media/results.png'
    d = np.loadtxt(csv, delimiter=',', skiprows=1)
    t = d[:, 0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ax[0].plot(t, np.degrees(d[:, 2]), label=r'$\varphi_2$ (positioner)')
    ax[0].plot(t, np.degrees(d[:, 3]), label=r'$\psi$ (torch)')
    ax[0].set_title('Synthesized time functions')
    for k in range(6):
        ax[1].plot(t, np.degrees(np.unwrap(d[:, 4 + k])), label=f'joint {k + 1}')
    ax[1].set_title('Robot joints (URDF convention)')
    ax[2].plot(t, d[:, 10], label=r'$w_T^*$ translational')
    ax[2].plot(t, d[:, 11], label=r'$w_R$ rotational')
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title('Manipulabilities')
    for a, yl in zip(ax, ['deg', 'deg', '']):
        a.set_xlabel('t [s]')
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.savefig(out, dpi=130)
    print(out)


def compare(paper='config/trajectory.csv', two='config/trajectory_two_axis.csv',
            out='media/two_axis.png'):
    """Paper (fixed tilt) against the study with both positioner axes moving."""
    a = np.loadtxt(paper, delimiter=',', skiprows=1)
    b = np.loadtxt(two, delimiter=',', skiprows=1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for d, lab, ls in [(a, 'paper: fixed tilt', '--'), (b, 'study: tilt + rotation', '-')]:
        ax[0].plot(d[:, 0], np.degrees(d[:, 1]), ls, label=lab)
        ax[1].plot(d[:, 0], d[:, 10], ls, label=lab)
        ax[2].plot(d[:, 0], d[:, 11], ls, label=lab)
    for a_, title in zip(ax, [r'Tilt $\varphi_1$ [deg]', r'Translational $w_T^*$', r'Rotational $w_R$']):
        a_.set_title(title)
        a_.set_xlabel('t [s]')
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)
    ax[1].set_ylim(0, 1.05)
    ax[2].set_ylim(0, 1.05)
    fig.savefig(out, dpi=130)
    print(out)


def _load(csv):
    with open(csv) as f:
        names = f.readline().strip().split(',')
    d = np.loadtxt(csv, delimiter=',', skiprows=1)
    return {n: d[:, i] for i, n in enumerate(names)}


def _shade(ax, c, arc='arc', seam='seam', color='#ebcb8b', seam2=None):
    """Shades the time the arc is on and writes the seam letter."""
    on = c[arc] > 0.5
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(c['t'][a], c['t'][b - 1], color=color, alpha=0.25, lw=0)
        label = chr(65 + int(c[seam][a])) + (f"+{chr(65 + int(c[seam2][a]))}" if seam2 else '')
        ax.text((c['t'][a] + c['t'][b - 1]) / 2, 1.0, label, ha='center', va='bottom',
                transform=ax.get_xaxis_transform(), fontsize=9)


def industrial(cfg='config/industrial', out='media'):
    """Figures of the industrial applications (not part of the paper)."""
    import json
    with open(f'{cfg}/summary.json') as f:
        summ = json.load(f)
    # one robot: positioner and manipulabilities along the cycle
    c = _load(f'{cfg}/trajectory_best_place.csv')
    on = c['arc'] > 0.5
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, constrained_layout=True)
    ax[0].plot(c['t'], np.degrees(c['phi1']), label=r'tilt $\varphi_1$')
    ax[0].plot(c['t'], np.degrees(c['phi2']), label=r'rotation $\varphi_2$')
    ax[0].set_ylabel('deg')
    ax[1].plot(c['t'], np.where(on, c['wT'], np.nan), label=r'$w_T^*$')
    ax[1].plot(c['t'], np.where(on, c['wR'], np.nan), label=r'$w_R$')
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel('t [s]')
    for a in ax:
        _shade(a, c)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc='lower right')
    s = summ['best_place']
    fig.suptitle(f"One robot: seams {' '.join(s['best_order'])} in the flat position, "
                 f"cycle {s['cycle_s']} s (arc on in the shaded parts)", fontsize=10)
    fig.savefig(f'{out}/industrial_cycle.png', dpi=130)
    # robot placement
    if 'layout' in summ:
        g = np.loadtxt(f'{cfg}/layout.csv', delimiter=',', skiprows=1)
        xs, ys = np.unique(g[:, 0]), np.unique(g[:, 1])
        F = np.full((len(ys), len(xs)), np.nan)
        for x, y, v in g:
            F[np.searchsorted(ys, y), np.searchsorted(xs, x)] = -v
        fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
        im = ax.imshow(F, origin='lower', cmap='viridis',
                       extent=[xs[0] - 0.05, xs[-1] + 0.05, ys[0] - 0.05, ys[-1] + 0.05])
        fig.colorbar(im, label=r'mean $-f$ of the four seams (higher is better)')
        b = summ['layout']['best_place']
        ax.plot(0, 0, 'wo', ms=8, label="paper's place")
        ax.plot(b[0], b[1], 'r*', ms=14, label='best place')
        ax.set_xlabel('robot offset x [m]')
        ax.set_ylabel('robot offset y [m]')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title('Robot placement (blank: a seam cannot be welded)', fontsize=10)
        fig.savefig(f'{out}/industrial_layout.png', dpi=130)
    # two robots
    if 'twin' in summ:
        c2 = _load(f'{cfg}/trajectory_twin.csv')
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True,
                               gridspec_kw={'width_ratios': [2.2, 1]})
        for k, (wt, arc, lab) in enumerate([('wT', 'arc', 'robot 1'), ('wT2', 'arc2', 'robot 2')]):
            m = c2[arc] > 0.5
            ax[0].plot(c2['t'], np.where(m, c2[wt], np.nan), label=fr'$w_T^*$ {lab}')
            ax[0].plot(c2['t'], np.where(m, c2[wt.replace('T', 'R')], np.nan), '--', label=fr'$w_R$ {lab}')
        _shade(ax[0], c2, seam2='seam2')
        ax[0].set_ylim(0, 1.05)
        ax[0].set_xlabel('t [s]')
        ax[0].grid(alpha=0.3)
        ax[0].legend(fontsize=7, ncol=2, loc='lower right')
        ax[0].set_title('Two robots: manipulabilities while welding', fontsize=10)
        labels = ['one robot,\nnaive order', 'one robot,\nbest order', 'two robots']
        vals = [summ['best_place']['naive_cycle_s'], summ['best_place']['cycle_s'], summ['twin']['cycle_s']]
        ax[1].bar(labels, vals, color=['#9aa5b8', '#5e81ac', '#a3be8c'])
        for i, v in enumerate(vals):
            ax[1].text(i, v, f'{v:.0f} s', ha='center', va='bottom', fontsize=9)
        ax[1].set_ylabel('cycle time [s]')
        ax[1].set_title('Cycle time of one spool', fontsize=10)
        fig.savefig(f'{out}/industrial_twin.png', dpi=130)
    print('media/industrial_*.png')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare()
    elif len(sys.argv) > 1 and sys.argv[1] == 'industrial':
        industrial()
    else:
        main()
