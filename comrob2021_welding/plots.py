"""Figures of the planned task: positioner and torch angles, robot joints and manipulabilities.

Usage:  python3 -m comrob2021_welding.plots [trajectory.csv] [output.png]
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


if __name__ == '__main__':
    main()
