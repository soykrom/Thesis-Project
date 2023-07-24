import gym
from gym import spaces
from gym.envs.registration import register

class rFactor2Environment(gym.Env):
    def __init__(self):
        # Steering and Acceleration
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
        # Velocity and Heading (X and Z axis)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,2), dtype=float)

    def calculate_reward(self):
        return 1

    def step(self, action):
        steering_action, acceleration_action = action

        # Execute the action using VJoy

        # Obtain in-game state (after action execution, it awaits until the info is received)

        # Calculate reward
        reward = self.calculate_reward()

        if self.is_done():
            self.reset_race()
            # Obtain observations related to reset

        return reward

    def is_done(self):
        return False
    def reset_race(self):
        # Send F command via VJoy
        pass

# Register Custom Environment
register(
    id='rFactor2Environment-v0',
    entry_point='envs.custom_env:rFactor2Environment',
)