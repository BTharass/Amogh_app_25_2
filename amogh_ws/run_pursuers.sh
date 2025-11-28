#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/install/setup.bash"
ros2 run pursuers pursuers_node 