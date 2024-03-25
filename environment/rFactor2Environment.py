import math
import pickle

import gym
import numpy as np
from gym import spaces
from trajectoryGenerator import generate_trajectory

VELOCITY = 11  # m/s
L = 2.25  # Distance between rear and front axel
TIME_DIVISION = 10  # Each timestep represents 1/10th of a second


class RFactor2Environment(gym.Env):
    def __init__(self):
        # Steering and Acceleration
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=float)
        # Previous state (for reward purposes)
        self.current_steps = 0
        self.position = [0.0, 0.0]
        self.ep_max_pl = 0
        self.heading = 0.0
        self.resets = 0
        self.current_index = 0
        self.max_index = 0

        self.trajectory, _ = generate_trajectory()

        with open("ep_max_dist.txt", "w") as file:
            file.write(str([self.max_index, self.resets]) + "\n")

        with open("positions.txt", "w") as file:
            file.write(str([self.position, self.heading, self.resets]) + "\n")

        with open("tests.txt", "w") as file:
            file.write(str([self.max_index, self.resets, 0]) + "\n")

    def update_trajectory_point(self):
        distances = np.linalg.norm(self.trajectory - (self.position[0], self.position[1]), axis=1)
        closest_index = np.argmin(distances)

        if closest_index > self.max_index and closest_index - self.max_index < 20:
            self.max_index = closest_index

            with open("ep_max_dist.txt", "a") as file:
                file.write(str([self.max_index, self.resets]) + '\n')

        return closest_index

    def reset(self, seed=None, options=None):
        print(f"RESETING {self.resets}...\n"
              f"Final Position: [{round(self.position[0], 2)}, {round(self.position[1], 2)}]\n"
              f"Current Index: {self.current_index}\tMax Index: {self.max_index}")

        self.current_steps = 0
        self.ep_max_pl = 0
        self.heading = 0.0
        self.position = [0.0, 0.0]
        self.resets += 1
        self.max_index = 0
        self.current_index = 0

        return np.array([self.position[0], self.position[1], 0, self.heading])

    def step(self, action):
        self.current_steps += 1

        self.heading += (VELOCITY / (TIME_DIVISION * L)) * math.tan(action * math.pi / 2)
        self.heading = round(self.heading % (2 * math.pi), 2)

        new_x = VELOCITY / TIME_DIVISION * math.cos(self.heading)
        new_y = VELOCITY / TIME_DIVISION * math.sin(self.heading)
        self.position[0] += round(new_x, 2)
        self.position[1] += round(new_y, 2)

        closest_index = self.update_trajectory_point()

        path_lateral = round(math.dist(self.position, self.trajectory[closest_index]), 2)

        done = path_lateral > 10 or self.max_index - self.current_index > 5

        progress = closest_index - self.current_index
        backwards_penalty = 0
        if progress > 25:
            # Agent has either skipped track or turned around at the start
            distances = np.linalg.norm(self.trajectory[:self.current_index + 1] - (self.position[0], self.position[1]), axis=1)
            closest_index = np.argmin(distances)
            progress = closest_index - self.current_index
            backwards_penalty = -5

        self.current_index = closest_index

        if path_lateral > self.ep_max_pl:
            self.ep_max_pl = path_lateral

        reward = round((-path_lateral + progress + backwards_penalty) / 4, 2)

        with open("positions.txt", "a") as file:
            file.write(str([self.position, path_lateral, action, done, reward, self.resets]) + '\n')

        return np.array([self.position[0], self.position[1], path_lateral, self.heading]), reward, done, False, dict()
