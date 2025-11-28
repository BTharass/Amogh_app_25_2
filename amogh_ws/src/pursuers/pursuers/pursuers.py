#!/usr/bin/env python3
"""
ROS 2 Node: TeleopKeyNode
Handles pursuer-evader dynamics, subscribes to evader actions,
updates agent states, and performs visualization and condition checks.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from . import initialise as init_init
from .encirclement import Pursuers

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Point, Polygon
import subprocess
import math


class TeleopKeyNode(Node):
    """Node for managing pursuer-evader interactions and visualization."""

    def __init__(self):
        super().__init__('teleop_key')

        # Initialize states, agent list, and parameters
        self.agent_states, self.agents,empty_empty= init_init.initialise_states()
        self.render_initialized = 0
        self.prev_all_state = self.agent_states
        self.dt = 0.1
        self.t = 0.0
        self.flag = True

        # Timer callback for periodic updates
        self.timer = self.create_timer(self.dt, self.run)

        # Initialize evader action (assumed stationary initially)
        self.evader_action = [0, 0]

        # Subscriber to receive evader actions
        self.evader_action_sub = self.create_subscription(
            Float64MultiArray,
            '/evader_action',
            self.evader_action_callback,
            10
        )

    def evader_action_callback(self, msg):
        """Callback to receive and normalize evader action velocity."""
        self.evader_action = msg.data
        velocity = self.evader_action
        mag = math.sqrt(velocity[0]**2 + velocity[1]**2)

        # Normalize if velocity magnitude exceeds 7
        if mag > 7.0:
            velocity[0] = (velocity[0] / mag) * 7
            velocity[1] = (velocity[1] / mag) * 7
            self.evader_action = velocity

    def run(self):
        """Main update loop executed at each timer tick."""
        self.t += self.dt

        evader_action = self.evader_action
        a, b = [], []
        for i in self.prev_all_state:
            a += list(self.prev_all_state[i])
            b.append(self.prev_all_state[i])

        # Compute next actions for pursuers
        pursuers_encirclement_obj = Pursuers()
        actions = list(pursuers_encirclement_obj.forward(a))
        actions.append(evader_action)

        # Update all agent states
        for agent_id, action in zip(self.agents, actions):
            a = self.agent_states[agent_id][:2]
            self.agent_states[agent_id][0:2] = [ai + self.dt * bi for ai, bi in zip(a, action)]
            self.agent_states[agent_id][2:] = action

        # Render and check conditions
        if self.flag:
            self.render_only_scatter()
            if self.t > 0.5:
                self.should_i_stop(b)

        self.prev_all_state = self.agent_states

    def render_only_scatter(self, mode="rgb_array"):
        """Render agent positions in a 2D scatter plot."""
        if self.render_initialized < 2:
            self.fig, self.ax = plt.subplots()
            plt.ion()
            self.render_initialized += 1
            self.ax.set_xlim(-120, 120)
            self.ax.set_ylim(-120, 120)
            self.ax.set_title("Agent Positions")

        x_max, y_max = 120, 120
        self.ax.clear()
        self.ax.set_xlim(-x_max, x_max)
        self.ax.set_ylim(-y_max, y_max)
        self.ax.set_title(f"Timestep: {self.t:.2f}")

        states = np.array([self.agent_states[agent] for agent in self.agents])
        pursuer_pose = states[:4, 0:2]
        evader_pose = states[-1, 0:2]

        self.ax.plot(evader_pose[0], evader_pose[1], 'ro', label='Evader',markersize=2)
        for i, pursuer in enumerate(pursuer_pose):
            self.ax.plot(pursuer[0], pursuer[1], 'bo',markersize=2)
            self.ax.text(pursuer[0] + 1, pursuer[1] + 1, f"P{i}", fontsize=8)
            circle = patches.Circle((pursuer[0], pursuer[1]), 0.5, fill=False, edgecolor='red', linewidth=1)
            self.ax.add_patch(circle)

        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def should_i_stop(self, b):
        """Check if evader is caught or escaped, and trigger outcomes."""
        all_state = b
        evader_poses = np.array(all_state[4][:2]).reshape(1, 2)
        evader_vel = all_state[4][2:]
        pursuer_states = np.array(all_state[0:4]).reshape(4, 4)
        pursuer_poses = pursuer_states[:, :2] - evader_poses[:, :2]

        alpha_i = np.arctan2(pursuer_poses[:, 1], pursuer_poses[:, 0])
        alpha_i = (alpha_i + 2 * np.pi) % (2 * np.pi)

        sorted_indices = np.argsort(alpha_i)
        pursuer_states_in_clockwise_order = pursuer_poses[sorted_indices]
        p_0, p_1, p_2, p_3 = pursuer_states_in_clockwise_order[:4, :2]

        inside_check_bool = self.inside_check(p_0, p_1, p_2, p_3)
        escape_ah = not inside_check_bool
        caught_ah = self.caught_check(p_0, p_1, p_2, p_3)

        if escape_ah:
            subprocess.Popen(['python3', 'win.py'])
            self.flag = False

        elif caught_ah:
            subprocess.Popen(['python3', 'lose.py'])
            self.flag = False

    def inside_check(self, p0, p1, p2, p3):
        """Check if evader (at origin) is inside the polygon formed by pursuers."""
        polygon_points = [(p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1])]
        poly = Polygon(polygon_points)
        origin = Point(0, 0)
        return poly.contains(origin)

    def caught_check(self, p0, p1, p2, p3):
        """Check if any pursuer is within radius 2 of the evader."""
        for p in [p0, p1, p2, p3]:
            if self.distance(p, [0, 0]) <= 2:
                return True
        return False

    def distance(self, p1, p2):
        """Compute Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))


def main(args=None):
    """ROS 2 entry point."""
    rclpy.init(args=args)
    teleop_node = TeleopKeyNode()
    rclpy.spin(teleop_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
