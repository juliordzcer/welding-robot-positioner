#!/usr/bin/env python3
"""Writes urdf/welding_cell.urdf: Fanuc M-10iA, torch, tilt-rotate positioner and workpiece.

The world frame is the workstation frame Sigma_e of the paper. The initial
joint positions are taken from the first row of config/trajectory.csv.
Usage (from the repository folder):  python3 scripts/gen_urdf.py
"""
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI2 = 1.57079633


def inertial(m=1.0, i=0.01):
    return (f'<inertial><mass value="{m}"/><inertia ixx="{i}" ixy="0" ixz="0" '
            f'iyy="{i}" iyz="0" izz="{i}"/></inertial>')


def vis(geom, mat, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    return (f'    <visual><origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" '
            f'rpy="{rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}"/><geometry>{geom}</geometry>'
            f'<material name="{mat}"/></visual>')


def box(x, y, z):
    return f'<box size="{x} {y} {z}"/>'


def cyl(r, h):
    return f'<cylinder radius="{r}" length="{h}"/>'


def link(name, visuals, m=1.0, i=0.01):
    return f'  <link name="{name}">\n    {inertial(m, i)}\n' + '\n'.join(visuals) + '\n  </link>'


def joint(name, parent, child, xyz, rpy=(0, 0, 0), jtype='fixed', axis=None, lim=None):
    s = (f'  <joint name="{name}" type="{jtype}">\n    <parent link="{parent}"/><child link="{child}"/>\n'
         f'    <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>')
    if axis:
        s += f'<axis xyz="{axis}"/>\n    <limit lower="{lim[0]}" upper="{lim[1]}" effort="1000" velocity="10"/>'
    return s + '\n  </joint>'


MATERIALS = {
    'fanuc_yellow': '0.96 0.76 0.13 1', 'fanuc_gray': '0.40 0.40 0.40 1',
    'fanuc_black': '0.12 0.12 0.12 1', 'steel': '0.66 0.68 0.70 1',
    'dark': '0.20 0.21 0.23 1', 'motor': '0.08 0.08 0.09 1',
    'pipe': '0.48 0.50 0.53 1', 'chalk': '0.92 0.92 0.88 1',
    'copper': '0.80 0.45 0.20 1', 'arc': '1.0 0.85 0.45 1', 'slot': '0.25 0.26 0.28 1',
}


def robot():
    names = ['base_link'] + [f'link_{i}' for i in range(1, 7)]
    color = {'base_link': 'fanuc_gray', 'link_6': 'fanuc_black'}
    out = []
    for n in names:
        out.append(f'''  <link name="{n}">
    {inertial()}
    <visual><geometry><mesh filename="package://comrob2021_welding/meshes/m10ia/visual/{n}.stl"/></geometry><material name="{color.get(n, 'fanuc_yellow')}"/></visual>
    <collision><geometry><mesh filename="package://comrob2021_welding/meshes/m10ia/collision/{n}.stl"/></geometry></collision>
  </link>''')
    J = [((0, 0, .45), '0 0 1', (-3.14, 3.14)), ((.15, 0, 0), '0 1 0', (-1.57, 2.79)),
         ((0, 0, .6), '0 -1 0', (-3.14, 4.61)), ((0, 0, .2), '-1 0 0', (-3.31, 3.31)),
         ((.64, 0, 0), '0 -1 0', (-3.31, 3.31)), ((.1, 0, 0), '-1 0 0', (-6.28, 6.28))]
    for i, (o, a, lim) in enumerate(J):
        out.append(joint(f'joint_{i + 1}', names[i], names[i + 1], o, jtype='revolute', axis=a, lim=lim))
    # pedestal (Sigma_0 is 0.45 m above Sigma_e, eq. 2; floor at z = -0.75)
    out.append(link('pedestal', [vis(box(0.5, 0.5, 0.7), 'dark', (0, 0, -0.40)),
                                 vis(box(0.62, 0.62, 0.04), 'fanuc_gray', (0, 0, -0.73)),
                                 vis(box(0.56, 0.56, 0.03), 'fanuc_gray', (0, 0, -0.035))]))
    out.append(joint('world_to_pedestal', 'world', 'pedestal', (0, 0, 0)))
    out.append(joint('world_to_base', 'world', 'base_link', (0, 0, 0)))
    return out


def torch():
    # tip 0.404 m ahead of the flange and 0.0498 m below, bent 22 deg (Fig. 2)
    neck_len = math.hypot(0.104, 0.0498)
    neck_pitch = math.atan2(0.104, -0.0498)
    return [
        link('torch', [vis(cyl(0.045, 0.05), 'dark', (0.025, 0, 0), (0, PI2, 0)),
                       vis(box(0.07, 0.09, 0.07), 'fanuc_black', (0.08, 0, 0.0)),
                       vis(cyl(0.022, 0.20), 'dark', (0.21, 0, 0), (0, PI2, 0)),
                       vis(cyl(0.012, 0.25), 'motor', (0.14, 0.03, 0.055), (0, PI2, 0)),
                       vis(cyl(0.017, neck_len), 'copper', (0.352, 0, -0.0249), (0, neck_pitch, 0))],
             0.5, 0.001),
        joint('link_6_to_torch', 'link_6', 'torch', (0, 0, 0)),
        link('torch_tip', [vis('<sphere radius="0.006"/>', 'arc')], 0.01, 1e-5),
        joint('torch_to_tip', 'torch', 'torch_tip', (0.404, 0, -0.0498), (3.14159265, -1.18682389, 0)),
    ]


def positioner():
    """Single-sided tilt-rotate positioner. Sigma_m0 placed with eq. (3):
    z_m0 (tilt axis) = world x, y_m0 = world z. Tilt axis at (1.2, 0.5, -0.01)."""
    out = []
    # fixed frame: housing of the tilt drive, beside the workpiece (at world x = 0.70)
    base = [
        vis(box(0.90, 0.70, 0.03), 'dark', (1.10, 0.50, -0.735)),                 # floor plate
        vis(box(0.30, 0.46, 0.66), 'fanuc_yellow', (0.70, 0.50, -0.40)),          # column
        vis(box(0.34, 0.50, 0.04), 'fanuc_gray', (0.70, 0.50, -0.71)),
        vis(cyl(0.16, 0.30), 'fanuc_yellow', (0.70, 0.50, -0.01), (0, PI2, 0)),   # gearbox housing
        vis(cyl(0.11, 0.06), 'fanuc_gray', (0.88, 0.50, -0.01), (0, PI2, 0)),     # output flange
        vis(cyl(0.075, 0.20), 'motor', (0.45, 0.50, -0.01), (0, PI2, 0)),         # tilt servo
        vis(cyl(0.08, 0.03), 'fanuc_gray', (0.35, 0.50, -0.01), (0, PI2, 0)),
        vis(box(0.12, 0.20, 0.16), 'dark', (0.70, 0.22, -0.45)),                  # junction box
    ]
    out.append(link('positioner_stand', base))
    out.append(joint('world_to_stand', 'world', 'positioner_stand', (0, 0, 0)))
    out.append(link('positioner_base', []))
    out.append(joint('stand_to_base', 'world', 'positioner_base', (1.2, 0.5, -0.01), (PI2, 0, PI2)))
    # tilting L-arm. Frame m1: z = tilt axis (world x), y = normal of the faceplate.
    arm = [
        vis(cyl(0.10, 0.06), 'fanuc_gray', (0, 0, -0.29)),                        # hub on the flange
        vis(box(0.26, 0.10, 0.34), 'fanuc_yellow', (0, -0.04, -0.14)),            # arm
        vis(cyl(0.20, 0.12), 'fanuc_yellow', (0, 0.02, 0.02), (PI2, 0, 0)),       # rotary housing
        vis(cyl(0.07, 0.18), 'motor', (0, -0.13, 0.12), (PI2, 0, 0)),             # rotation servo
        vis(cyl(0.075, 0.025), 'fanuc_gray', (0, -0.225, 0.12), (PI2, 0, 0)),
    ]
    out.append(link('positioner_link_1', arm, 5, 0.1))
    out.append(joint('positioner_joint_1', 'positioner_base', 'positioner_link_1', (0, 0, 0),
                     jtype='revolute', axis='0 0 1', lim=(-2.0, 2.0)))
    # faceplate, chuck and workpiece. Frame m2 = frame k of the workpiece (z = pipe axis).
    plate = [vis(cyl(0.26, 0.035), 'steel', (0, 0, -0.0175))]
    for k in range(4):                                                            # T-slots
        a = k * math.pi / 4
        plate.append(vis(box(0.50, 0.018, 0.004), 'slot', (0, 0, 0.0005), (0, 0, a)))
    plate.append(vis(cyl(0.135, 0.05), 'dark', (0, 0, 0.025)))                    # 3-jaw chuck
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.5
        plate.append(vis(box(0.05, 0.035, 0.07), 'steel',
                         (0.125 * math.cos(a), 0.125 * math.sin(a), 0.05), (0, 0, a)))
    plate.append(vis(cyl(0.100, 0.45), 'pipe', (0, 0, 0.225)))                     # pipe
    plate.append(vis(cyl(0.088, 0.003), 'dark', (0, 0, 0.4515)))                   # bore
    # chalk guide line of the helical seam (Fig. 4)
    N, r, h = 120, 0.1008, 0.45
    pts = [np.array([r * math.cos(2 * math.pi * i / N), r * math.sin(2 * math.pi * i / N), h * i / N])
           for i in range(N + 1)]
    for a, b in zip(pts, pts[1:]):
        m, d = (a + b) / 2, b - a
        L = np.linalg.norm(d)
        u = d / L
        plate.append(vis(cyl(0.0012, L * 1.05), 'chalk', m,
                         (0, math.acos(u[2]), math.atan2(u[1], u[0]))))
    out.append(link('positioner_link_2', plate, 5, 0.1))
    out.append(joint('positioner_joint_2', 'positioner_link_1', 'positioner_link_2', (0, 0.130, 0),
                     (-PI2, 0, 0), jtype='revolute', axis='0 0 1', lim=(-6.3, 6.3)))
    return out


def gazebo(init):
    names = ['positioner_joint_1', 'positioner_joint_2'] + [f'joint_{i}' for i in range(1, 7)]
    pl = ['    <plugin filename="gz-sim-joint-state-publisher-system" '
          'name="gz::sim::systems::JointStatePublisher">\n'
          '      <topic>/welding/joint_states</topic>\n    </plugin>']
    for n, v in zip(names, init):
        pl.append(f'''    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>{n}</joint_name>
      <topic>/welding/{n}/cmd_pos</topic>
      <use_velocity_commands>true</use_velocity_commands>
      <p_gain>20</p_gain>
      <cmd_max>6</cmd_max>
      <cmd_min>-6</cmd_min>
      <initial_position>{v:.6f}</initial_position>
    </plugin>''')
    return '  <gazebo>\n' + '\n'.join(pl) + '\n  </gazebo>'


def main():
    d = np.loadtxt(os.path.join(ROOT, 'config', 'trajectory.csv'), delimiter=',', skiprows=1)[0]
    init = [d[1], d[2]] + list(d[4:10])
    mats = '\n'.join(f'  <material name="{k}"><color rgba="{v}"/></material>' for k, v in MATERIALS.items())
    body = '\n\n'.join(['\n'.join(robot()), '\n'.join(torch()), '\n'.join(positioner())])
    urdf = f'''<?xml version="1.0"?>
<!-- Generated by scripts/gen_urdf.py. Welding cell of the paper "Optimal Motions of a
     Welding Robot for Cooperative Tasks with a 2 DOF Positioner" (COMRob 2021).
     World frame = workstation frame Sigma_e. Robot meshes: ROS-Industrial (BSD-3-Clause). -->
<robot name="welding_cell">
{mats}
  <link name="world"/>

{body}

{gazebo(init)}
</robot>
'''
    with open(os.path.join(ROOT, 'urdf', 'welding_cell.urdf'), 'w') as f:
        f.write(urdf)
    print('urdf/welding_cell.urdf')


if __name__ == '__main__':
    main()
