import math

import gym
import numpy as np
from gym import spaces

VELOCITY = 11  # m/s
L = 2.25  # Distance between rear and front axel
TIME_DIVISION = 10  # Each timestep represents 1/10th of a second

class RFactor2Environment(gym.Env):
    def __init__(self):
        # Steering and Acceleration
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
        # Previous state (for reward purposes)
        self.current_steps = 0
        self.position = [0.0, 0.0]
        self.ep_max_dist = 0
        self.ep_max_pl = 0
        self.heading = 0.0
        self.resets = 0

        with open("ep_max_dist.txt", "w") as file:
            file.write(str([self.ep_max_dist, self.resets]) + "\n")

        with open("positions.txt", "w") as file:
            file.write(str([self.position, self.heading, self.resets]) + "\n")

        with open("tests.txt", "w") as file:
            file.write(str([self.ep_max_dist, self.resets, 0]) + "\n")

    def reset(self, seed=None, options=None):
        print(f"RESETING {self.resets}...\nFinal X: {round(self.position[0], 2)}\tMax X: {round(self.ep_max_dist, 2)}"
              f"\nFinal Y: {round(self.position[1], 2)}\tMax Y: {round(self.ep_max_pl, 2)}")

        self.current_steps = 0
        self.ep_max_dist = 0
        self.ep_max_pl = 0
        self.heading = 0.0
        self.position = [0.0, 0.0]
        self.resets += 1

        return np.array([self.position[1], self.heading])

    def step(self, action):
        self.current_steps += 1

        self.heading += (VELOCITY / (TIME_DIVISION * L)) * math.tan(action * math.pi / 2)
        self.heading = round(self.heading % (2 * math.pi), 2)

        new_x = VELOCITY / TIME_DIVISION * math.cos(self.heading)
        new_y = VELOCITY / TIME_DIVISION * math.sin(self.heading)
        self.position[0] += round(new_x, 2)
        self.position[1] += round(new_y, 2)

        if abs(self.position[1]) > self.ep_max_pl:
            self.ep_max_pl = self.position[1]

        if self.position[0] > self.ep_max_dist:
            self.ep_max_dist = self.position[0]

            with open("ep_max_dist.txt", "a") as file:
                file.write(str([self.ep_max_dist, self.resets]) + '\n')

        done = abs(self.position[1]) > 10 or self.position[0] > 1000 or self.position[0] < self.ep_max_dist - 10

        reward = -abs(self.position[1])
        reward = round((reward + new_x) / 4, 2)

        with open("positions.txt", "a") as file:
            file.write(str([self.position, action, done, reward, self.resets]) + '\n')

        return np.array([self.position[1], self.heading]), reward, done, False, dict()
