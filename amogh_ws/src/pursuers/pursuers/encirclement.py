import numpy as np
import math


class Pursuers:
    def __init__(self):
        """
        Initializes encirclement control parameters and constants.
        """
        super().__init__()
        self.sorted_indices = None
        self.k = 2  # Encirclement gain
        self.h = 0.1  # Radial contraction gain
        self.first_iteration = True
        self.guys = ["pursuer0", "pursuer1", "pursuer2", "pursuer3", "evader"]
        self.max_pursuers_speed = 11  # Maximum pursuer speed

    def forward(self, states):
        """
        Computes the pursuers' actions given all agent states.
        The evader is assumed to be the last entry in the state list.

        Args:
            states (list or np.ndarray): Flattened list of agent states.

        Returns:
            np.ndarray: Array of 2D velocity actions for each pursuer.
        """
        evader_state = np.array([states[-4:]])  # Shape: (1, 4)
        pursuer_state = np.array(states[:4 * 4]).reshape(4, -1)  # Shape: (4, 4)

        # Compute actions for encirclement and hunting
        actions_encirclement, r_i, alpha_i, speeds = self.encirclement(
            pursuer_states=pursuer_state, evador_state=evader_state, k=1
        )
        h = self.h
        actions_hunting = self.hunting(r_i=r_i, h=h, alpha_i=alpha_i)

        # Adjust encirclement and hunting parameters
        self.k, self.h = self.tradeoff(speeds, r_i=r_i)

        # Combine encirclement and hunting actions
        actions = actions_encirclement + actions_hunting
        return actions

    def encirclement(self, pursuer_states, evador_state, k):
        """
        Computes tangential (encircling) motion for each pursuer.

        Args:
            pursuer_states (np.ndarray): Pursuer positions and orientations.
            evador_state (np.ndarray): Evader position and orientation.
            k (float or np.ndarray): Encirclement gain.

        Returns:
            tuple: (delta_xy, r_i_unsorted, alpha_i_unsorted, speeds)
        """
        num_pursuers = pursuer_states.shape[0]
        self.dt = 0.1
        max_evador_velocity = 7

        # Compute pursuer position in evader-centered frame
        pursuer_e_frame = np.hstack((
            pursuer_states[:, 0:2] - evador_state[:, 0:2],
            np.expand_dims(pursuer_states[:, 2], 1)
        ))

        # Compute relative bearing (angle from evader)
        alpha_i = np.arctan2(pursuer_e_frame[:, 1], pursuer_e_frame[:, 0])
        alpha_i = (alpha_i + 2 * np.pi) % (2 * np.pi)

        # Sort pursuers by angular position around evader
        self.sorted_indices = np.argsort(alpha_i)
        pursuer_e_frame = pursuer_e_frame[self.sorted_indices]
        alpha_i = alpha_i[self.sorted_indices]
        r_i = np.sqrt(pursuer_e_frame[:, 0] ** 2 + pursuer_e_frame[:, 1] ** 2)

        # Store unsorted radius and angle for later steps
        r_i_unsorted = np.zeros_like(r_i)
        r_i_unsorted[self.sorted_indices] = r_i
        alpha_i_unsorted = np.zeros_like(alpha_i)
        alpha_i_unsorted[self.sorted_indices] = alpha_i

        # Compute angular coverage and speed allocation
        test_pursuer_speed = np.full(num_pursuers, self.max_pursuers_speed)
        lambda_i = test_pursuer_speed / max_evador_velocity
        lambda_i = np.clip(lambda_i, None, 0.9999)
        self.occupied_i = 2 * np.arcsin(lambda_i)

        # Compute angular spacing between pursuers
        coverage_angle_i = (
            np.roll(alpha_i, -1) - alpha_i
            - (np.roll(self.occupied_i, -1) + self.occupied_i) / 2
        )
        coverage_angle_i[-1] += 2 * np.pi

        # Angular rate of change (with or without adaptive gain)
        if isinstance(k, int):
            d_alpha = k * (coverage_angle_i - np.roll(coverage_angle_i, 1))
            d_alpha_without_k = coverage_angle_i - np.roll(coverage_angle_i, 1)
        else:
            d_alpha = k[self.sorted_indices] * (coverage_angle_i - np.roll(coverage_angle_i, 1))

        speeds = d_alpha * r_i
        dx = -speeds * np.sin(alpha_i)
        dy = speeds * np.cos(alpha_i)

        # Unsort motion vectors back to original pursuer order
        self.d_alpha_unsorted = np.zeros_like(d_alpha)
        self.d_alpha_without_k_unsorted = np.zeros_like(d_alpha)
        dx_unsorted = np.zeros_like(dx)
        dy_unsorted = np.zeros_like(dy)

        self.d_alpha_unsorted[self.sorted_indices] = d_alpha
        self.d_alpha_without_k_unsorted[self.sorted_indices] = d_alpha_without_k
        dx_unsorted[self.sorted_indices] = dx
        dy_unsorted[self.sorted_indices] = dy

        # Combine x and y velocities
        delta_xy = np.stack((dx_unsorted, dy_unsorted), axis=1)

        return delta_xy, r_i_unsorted, alpha_i_unsorted, speeds

    def hunting(self, r_i, h, alpha_i):
        """
        Computes radial contraction motion (hunting behavior).

        Args:
            r_i (np.ndarray): Distances from evader.
            h (float): Radial contraction gain.
            alpha_i (np.ndarray): Angles of pursuers relative to evader.

        Returns:
            np.ndarray: Radial velocity components for each pursuer.
        """
        dr = -h * r_i
        dx = dr * np.cos(alpha_i)
        dy = dr * np.sin(alpha_i)
        delta_xy = np.stack((dx, dy), axis=1)
        return delta_xy

    def tradeoff(self, speeds, r_i):
        """
        Adjusts tradeoff gains (k, h) between encirclement and contraction.

        Args:
            speeds (np.ndarray): Pursuer speed magnitudes.
            r_i (np.ndarray): Distances from evader.

        Returns:
            tuple: (k_i, h_i) updated gains.
        """
        max_speed = self.max_pursuers_speed
        delta = 2 * abs(self.d_alpha_without_k_unsorted) / (
            4 * math.pi - np.roll(self.occupied_i, -1) + np.roll(self.occupied_i, 1)
        )
        gamma = np.sin(math.pi * ((r_i / (r_i + np.roll(r_i, 1) + np.roll(r_i, -1))) ** (math.log(2) / math.log(3))))
        beta = (math.pi / 2) * (1 - np.exp(-delta * gamma))

        if any(x == 0 for x in abs(speeds)):
            k_i = 2
            h_i = 0.1
        else:
            k_i = max_speed * np.sin(beta) / abs(speeds)
            h_i = max_speed * np.cos(beta) / r_i

        return k_i, h_i
