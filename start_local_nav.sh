#!/bin/bash
# Equivalent of catkin_ws/start_radar_system.sh for the new local_nav stack.
#
# Usage:
#   ./start_local_nav.sh          -> loopback:=true  (safe, no real serial writes)
#   ./start_local_nav.sh live     -> loopback:=false (ACTUAL serial writes - buggy will move)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

LOOPBACK=true
if [ "$1" == "live" ]; then
    LOOPBACK=false
    echo -e "${RED}LIVE MODE: loopback:=false - the buggy WILL move on real steering/UART commands.${NC}"
    echo -e "${RED}Make sure an e-stop / manual override is within reach before continuing.${NC}"
else
    echo -e "${YELLOW}Safe mode: loopback:=true - no real serial writes will occur.${NC}"
    echo -e "${YELLOW}Run './start_local_nav.sh live' once /steering_angle, /odom, and /wheel_uart all look correct.${NC}"
fi

echo "====================================="
echo -e "${YELLOW}Starting Local Nav System${NC}"
echo "====================================="

# Step 1: Clean up any leftover processes from previous runs - repeated
# hard-kills without this can leave stale nodes/discovery state behind.
echo -e "${YELLOW}[1/3] Cleaning up any leftover processes...${NC}"
pkill -9 -f "ros2 launch local_nav" 2>/dev/null
pkill -9 -f velodyne_driver_node 2>/dev/null
pkill -9 -f velodyne_transform_node 2>/dev/null
pkill -9 -f uart_sender_node 2>/dev/null
pkill -9 -f uart_bridge 2>/dev/null
pkill -9 -f "nav2_controller/controller_server" 2>/dev/null
pkill -9 -f "nav2_planner/planner_server" 2>/dev/null
pkill -9 -f "nav2_lifecycle_manager/lifecycle_manager" 2>/dev/null
pkill -9 -f "ground_segmentation/segment_ground" 2>/dev/null
pkill -9 -f "ground_segmentation/velodyne_static_tf" 2>/dev/null
pkill -9 -f "local_nav/self_hit_filter" 2>/dev/null
pkill -9 -f "local_nav/wheel_odometry" 2>/dev/null
pkill -9 -f "local_nav/carrot_path_publisher" 2>/dev/null
pkill -9 -f "local_nav/steering_uart_bridge" 2>/dev/null
sleep 2
echo -e "${GREEN}Cleanup done${NC}"

# Step 2: Source ROS environment
echo -e "\n${YELLOW}[2/3] Setting up ROS environment...${NC}"
source /opt/ros/humble/setup.bash
source /home/divagar/ros2_r_ws/buggy_demo_01/install/setup.bash
echo -e "${GREEN}ROS workspace sourced${NC}"

# Step 3: Launch the ROS nodes
echo -e "\n${YELLOW}[3/3] Launching real_navigate.launch.py (loopback:=${LOOPBACK})...${NC}"
ros2 launch local_nav real_navigate.launch.py loopback:=${LOOPBACK}

echo -e "\n${YELLOW}System shutdown complete${NC}"
