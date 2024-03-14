import math

import gym
import numpy as np
from gym import spaces

VELOCITY = 11  # m/s
L = 2.25  # Distance between rear and front axel


class RFactor2Environment(gym.Env):
    def __init__(self):
        # Steering and Acceleration
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
        # Previous state (for reward purposes)
        self.current_steps = 0
        self.position = [0.0, 0.0]
        self.max_dist = 0
        self.heading = 0.0
        self.resets = 0

        with open("max_dist.txt", "w") as file:
            file.write(str([self.max_dist, self.resets]) + "\n")

        with open("positions.txt", "w") as file:
            file.write(str([self.position, self.heading, self.resets]) + "\n")

        with open("max_y.txt", "w") as file:
            file.write(str([self.max_dist, self.resets, 0]) + "\n")

    def reset(self, seed=None, options=None):
        print(f"RESETING {self.resets}...\nFinal X: {self.position[0]}\tMax X: {self.max_dist}")

        self.current_steps = 0
        self.heading = 0.0
        self.position = [0.0, 0.0]
        self.resets += 1

        return np.array([self.position[1], self.heading])

    def step(self, action):
        self.current_steps += 1

        self.heading += (VELOCITY / L) * math.tan(action * math.pi / 2)
        self.heading = self.heading % (2 * math.pi)

        new_x = VELOCITY / 10 * math.cos(self.heading)
        new_y = VELOCITY / 10 * math.sin(self.heading)
        self.position[0] += new_x
        self.position[1] += new_y

        if self.position[0] > self.max_dist:
            self.max_dist = self.position[0]
            print("NEW MAX DISTANCE: ", self.max_dist)
            print("Path Lateral: ", self.position[1])

            with open("max_dist.txt", "a") as file:
                file.write(str([self.max_dist, self.resets]) + '\n')

        done = abs(self.position[1]) > 10 or self.position[0] > 1000

        if self.position[1] == 0:
            reward = 10
        else:
            reward = np.clip(1 / (0.4 * abs(self.position[1])), 0, 10) \
                if abs(self.position[1]) < 4 \
                else -abs(self.position[1])

        if new_x < 0:
            reward -= abs(self.position[0])
            done = True

        with open("positions.txt", "a") as file:
            file.write(str([self.position, action, done, reward, self.resets]) + '\n')

        return np.array([self.position[1], self.heading]), reward, done, False, dict()
