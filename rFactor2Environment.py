import time

import gym
import pyvjoy
from gym import spaces

import fidgrovePluginUtils as utils

NEUTRAL_POSITION = 16384


class RFactor2Environment(gym.Env):
    def __init__(self):
        # Steering and Acceleration
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=float)
        # Vjoy Device
        self.vjoy_device = pyvjoy.VJoyDevice(1)
        # Previous state (for reward purposes)
        self.prev_state = None

    def reset(self, seed=None, options=None):
        # Send resets command via VJoy
        self.vjoy_device.set_button(1, 1)
        print("RESET")

        time.sleep(0.5)
        # Turn reset button off
        self.vjoy_device.set_button(1, 0)

        # Obtain observations related to reset
        new_state = utils.obtain_state()
        self.prev_state = new_state

        self.vjoy_device.data.wAxisX = NEUTRAL_POSITION
        self.vjoy_device.data.wAxisY = 0
        self.vjoy_device.data.wAxisY = 0

        self.vjoy_device.update()

        time.sleep(9.0)

        return utils.scale_features(new_state), dict()

    def step(self, action):
        # Execute the action using VJoy
        print("Action in Step: ", action)
        if action.ndim > 1:
            action = action[0]

        self.vjoy_device.data.wAxisX = int(NEUTRAL_POSITION + float(action[0]) * NEUTRAL_POSITION)

        self.vjoy_device.data.wAxisY = utils.calculate_throttle_action(utils.convert_mps_to_kph(self.prev_state[3]))
        # self.vjoy_device.data.wAxisY = int(NEUTRAL_POSITION + float(action[1]) * NEUTRAL_POSITION)

        self.vjoy_device.update()

        # Obtain in-game data (after action execution, it waits until the info is received)
        new_state = utils.obtain_state()

        if self.prev_state is None:
            self.prev_state = new_state

        done = utils.episode_finish(new_state)
        reward = utils.calculate_reward(self.prev_state, new_state, done)

        utils.reset_events()

        time.sleep(0.05)
        return utils.scale_features(new_state), reward, done, False, dict()


gym.register(id='RFactor2-v0', entry_point='rFactor2Environment:RFactor2Environment')
