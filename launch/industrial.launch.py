"""Industrial applications (NOT part of the COMRob 2021 paper): flanged pipe spool.

cell:=single  one robot, flat-position seams in the best order (applications 1 and 2)
cell:=twin    two robots welding at the same time on the same positioner (application 4)
place:=paper_place | best_place   robot place of the paper or of the placement study (single cell)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def nodes(context):
    share = get_package_share_directory('comrob2021_welding')
    cfg = os.path.join(share, 'config', 'industrial')
    cell = LaunchConfiguration('cell').perform(context)
    place = LaunchConfiguration('place').perform(context)
    name = 'twin' if cell == 'twin' else place
    csv = os.path.join(cfg, f'trajectory_{name}.csv')
    with open(os.path.join(share, 'urdf', f'welding_cell_industrial_{name}.urdf')) as f:
        urdf = f.read()
    with open(csv) as f:
        header = f.readline().strip().split(',')
    joints = ['positioner_joint_1', 'positioner_joint_2'] + [
        n for n in header if n.startswith('joint_') or n.startswith('r2_joint_')]
    sim = {'use_sim_time': True}
    gui = LaunchConfiguration('gui')
    world = LaunchConfiguration('world').perform(context) or os.path.join(
        share, 'worlds', 'welding_cell_twin.sdf' if cell == 'twin' else 'welding_cell.sdf')
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': [PythonExpression(["'-r ' if '", gui, "' == 'true' else '-r -s --headless-rendering '"]),
                                      world],
                          'on_exit_shutdown': 'true'}.items())
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 condition=IfCondition(LaunchConfiguration('spawn')),
                 arguments=['-name', 'welding_cell', '-topic', 'robot_description'])
    bridge_args = ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                   '/welding/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model']
    bridge_args += [f'/welding/{j}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double' for j in joints]
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', arguments=bridge_args,
                  remappings=[('/welding/joint_states', '/joint_states')], parameters=[sim])
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': urdf}, sim])
    player = Node(package='comrob2021_welding', executable='trajectory_player.py', output='screen',
                  parameters=[sim, {'loop': LaunchConfiguration('loop'), 'bead': LaunchConfiguration('bead'),
                                     'csv': csv, 'mode': f'industrial ({name})'}])
    panel = Node(package='comrob2021_welding', executable='control_panel.py',
                 condition=IfCondition(LaunchConfiguration('panel')))
    return [gz, spawn, bridge, rsp, player, panel]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('cell', default_value='single', choices=['single', 'twin']),
        DeclareLaunchArgument('place', default_value='best_place', choices=['paper_place', 'best_place']),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('world', default_value=''),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('panel', default_value='true'),
        DeclareLaunchArgument('bead', default_value='true'),
        DeclareLaunchArgument('spawn', default_value='true'),
        OpaqueFunction(function=nodes),
    ])
