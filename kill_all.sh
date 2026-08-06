#!/bin/bash
# Kills every process related to the local_nav stack (sim, bag, and real
# hardware modes) - Gazebo, Nav2 nodes, our own nodes, the Velodyne
# driver, UART nodes, bag playback, and RViz. Safe to run any time before
# a fresh launch, whether or not anything is actually running.
#
# Targets executables directly, not just the "ros2 launch"/"ros2 bag play"
# wrapper process - once a wrapper is killed its children get reparented
# and a wrapper-only kill leaves them running as orphans (bitten by this
# more than once this session).
#
# Usage: ./kill_all.sh          - kill everything
#        ./kill_all.sh shm      - also clear stale FastDDS shared memory
#                                 (only needed if nodes are running but not
#                                 showing up in `ros2 node list`)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Killing all local_nav-related processes...${NC}"

pkill -9 -f "ros2 launch" 2>/dev/null
pkill -9 -f "ros2 bag play" 2>/dev/null
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null
pkill -9 -f rviz2 2>/dev/null

pkill -9 -f "nav2_controller/controller_server" 2>/dev/null
pkill -9 -f "nav2_planner/planner_server" 2>/dev/null
pkill -9 -f "nav2_lifecycle_manager/lifecycle_manager" 2>/dev/null

pkill -9 -f "ground_segmentation/segment_ground" 2>/dev/null
pkill -9 -f "ground_segmentation/velodyne_static_tf" 2>/dev/null

pkill -9 -f "local_nav/self_hit_filter" 2>/dev/null
pkill -9 -f "local_nav/wheel_odometry" 2>/dev/null
pkill -9 -f "local_nav/tf_odom_relay" 2>/dev/null
pkill -9 -f "local_nav/carrot_path_publisher" 2>/dev/null
pkill -9 -f "local_nav/steering_uart_bridge" 2>/dev/null
pkill -9 -f "local_nav/initialpose_to_slam_toolbox" 2>/dev/null

pkill -9 -f "uart_bridge" 2>/dev/null
pkill -9 -f "uart_sender_node" 2>/dev/null
pkill -9 -f "velodyne_driver_node" 2>/dev/null
pkill -9 -f "velodyne_transform_node" 2>/dev/null
pkill -9 -f "slam_toolbox" 2>/dev/null
pkill -9 -f "pointcloud_to_laserscan" 2>/dev/null

pkill -9 -f "steering_teleop" 2>/dev/null

sleep 2

REMAINING=$(ps -ef | grep -iE "ros2 launch|ros2 bag play|rviz2|gzserver|gzclient|nav2_controller|nav2_planner|nav2_lifecycle_manager|segment_ground|velodyne_static_tf|self_hit_filter|wheel_odometry|tf_odom_relay|carrot_path_publisher|steering_uart_bridge|steering_teleop|uart_bridge|uart_sender_node|velodyne_driver_node|velodyne_transform_node|slam_toolbox|initialpose_to_slam_toolbox|pointcloud_to_laserscan" | grep -v grep | grep -v "kill_all.sh")
if [ -z "$REMAINING" ]; then
    echo -e "${GREEN}All clear - nothing left running.${NC}"
else
    echo -e "${YELLOW}Still alive (needs manual attention):${NC}"
    echo "$REMAINING"
fi

if [ "$1" == "shm" ]; then
    echo -e "\n${YELLOW}Clearing stale FastDDS shared memory (nodes running but not discoverable)...${NC}"
    source /opt/ros/humble/setup.bash
    ros2 daemon stop
    rm -f /dev/shm/fastrtps_*
    ros2 daemon start
    echo -e "${GREEN}Done.${NC}"
fi
