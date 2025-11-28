import numpy as np
import random
from math import sin, cos, pi


# Number of agents
num_pursuers = 4
num_evadors = 1

# Initial encirclement radius (can be tuned per policy)
initial_encirclement_radius = 100.0


# -------------------------------
#   INITIALIZATION FUNCTIONS
# -------------------------------

def initialise_evador_position():
    """
    Initializes the evader's position at the origin.
    Returns:
        np.ndarray: 2x1 array representing the evader's (x, y) position.
    """
    return np.array([[0.0, 0.0]]).T


def initialise_pursuer_position(num_pursuers, num_evaders, evader_initial_poses, radius):
    """
    Initializes pursuers in a circular formation around the evader.

    Args:
        num_pursuers (int): Number of pursuers.
        num_evaders (int): Number of evaders.
        evader_initial_poses (list): Initial positions of evaders.
        radius (float): Encirclement radius.

    Returns:
        np.ndarray: 2xN array containing the (x, y) positions of pursuers.
    """
    if evader_initial_poses is None:
        evader_initial_poses = [[0.0, 0.0, 0.0] for _ in range(num_evaders)]

    # Center around the first evader (at origin by default)
    cx, cy = 0.0, 0.0

    # Distribute pursuers uniformly around the evader
    angles = np.linspace(0, 2 * np.pi, num_pursuers, endpoint=False)
    x_positions = cx + radius * np.cos(angles)
    y_positions = cy + radius * np.sin(angles)

    return np.stack((x_positions, y_positions))


def initialise_states(num_pursuers=num_pursuers,
                      num_evaders=num_evadors,
                      evader_initial_poses_fn=initialise_evador_position,
                      radius=initial_encirclement_radius):
    """
    Creates and initializes all agent states.

    Args:
        num_pursuers (int): Number of pursuers.
        num_evaders (int): Number of evaders.
        evader_initial_poses_fn (function): Function that returns initial evader positions.
        radius (float): Radius of initial encirclement.

    Returns:
        tuple: (initial_state, agent_names, num_pursuers)
            - initial_state (dict): Mapping of agent names to their state vectors.
            - agent_names (list): List of all agent names.
            - num_pursuers (int): Number of pursuers.
    """
    evader_poses = evader_initial_poses_fn()
    pursuer_poses = initialise_pursuer_position(
        num_pursuers, num_evaders, evader_poses, radius
    )

    initial_state = {}
    agent_names = []

    for i in range(num_evaders + num_pursuers):
        if i < num_pursuers:
            # Pursuer agents
            name = f"pursuer{i}"
            pose = pursuer_poses[:, i]
        else:
            # Evader agents
            name = f"evader{i - num_pursuers}"
            pose = evader_poses[:, 0]

        agent_names.append(name)
        state = np.zeros((4))
        state[0:2] = pose
        initial_state[name] = state

    return initial_state, agent_names, num_pursuers


# -------------------------------
#   HELPER FUNCTIONS
# -------------------------------

def _uniform_noise_method(radius, center):
    """
    Generates four points roughly evenly spaced around a circle,
    adding random angular noise to each.

    Args:
        radius (float): Distance from center.
        center (list): [x, y] coordinates of center.

    Returns:
        list: List of (x, y) tuples for each point.
    """
    base_angles = [i * 2 * pi / 4 for i in range(4)]
    max_noise = pi / 6
    angles = []

    for base_angle in base_angles:
        noise = random.uniform(-max_noise, max_noise)
        angle = (base_angle + noise) % (2 * pi)
        angles.append(angle)

    points = []
    for angle in angles:
        # (Hardcoded values used for debugging originally)
        x = center[0] + radius * cos(angle)
        y = center[1] + radius * sin(angle)
        points.append((x, y))

    return points


# -------------------------------
#   SIMULATION PARAMETERS
# -------------------------------

time_step = 0.1
max_time = 200
