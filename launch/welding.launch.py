"""Gazebo welding cell: Fanuc ARC Mate 100iC + 2-DOF positioner (COMRob 2021)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

JOINTS = ['positioner_joint_1', 'positioner_joint_2'] + [f'joint_{i}' for i in range(1, 7)]


def generate_launch_description():
    share = get_package_share_directory('comrob2021_welding')
    with open(os.path.join(share, 'urdf', 'welding_cell.urdf')) as f:
        urdf = f.read()
    gui = LaunchConfiguration('gui')
    sim = {'use_sim_time': True}

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': [PythonExpression(["'-r ' if '", gui, "' == 'true' else '-r -s --headless-rendering '"]),
                                      LaunchConfiguration('world')],
                          'on_exit_shutdown': 'true'}.items())

    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-name', 'welding_cell', '-topic', 'robot_description'])

    bridge_args = ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                   '/welding/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model']
    bridge_args += [f'/welding/{j}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double' for j in JOINTS]
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', arguments=bridge_args,
                  remappings=[('/welding/joint_states', '/joint_states')], parameters=[sim])

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': urdf}, sim])

    player = Node(package='comrob2021_welding', executable='trajectory_player.py', output='screen',
                  parameters=[sim, {'loop': LaunchConfiguration('loop')}])

    rviz = Node(package='rviz2', executable='rviz2', condition=IfCondition(LaunchConfiguration('rviz')),
                arguments=['-d', os.path.join(share, 'config', 'welding.rviz')], parameters=[sim])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('world', default_value=os.path.join(share, 'worlds', 'welding_cell.sdf')),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('loop', default_value='true'),
        gz, spawn, bridge, rsp, player, rviz,
    ])
