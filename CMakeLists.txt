cmake_minimum_required(VERSION 3.16)
project(comrob2021_welding)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS nodes/trajectory_player.py nodes/control_panel.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch urdf worlds config meshes DESTINATION share/${PROJECT_NAME})

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/50.${PROJECT_NAME}.dsv.in")

ament_package()
