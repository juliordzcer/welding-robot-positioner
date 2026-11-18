"""Figures of the planned task: positioner and torch angles, robot joints and manipulabilities.

Usage:  python3 -m comrob2021_welding.plots [trajectory.csv] [output.png]
        python3 -m comrob2021_welding.plots compare
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


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare()
    else:
        main()
