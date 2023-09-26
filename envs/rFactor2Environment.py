import pyvjoy
import gym
from gym import spaces
from gym.envs.registration import register
import win32event
import mmap
import math
import time

# Setting up rFactor2 plugin reader
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=73, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

vehicleScoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteVehicleScoring")
vehicleScoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)


def calculate_heading(x, z):
    if x > 0 and z > 0:
        # 1st quadrant
        heading = math.asin(x)
    elif x > 0 > z:
        # 2nd quadrant
        heading = math.pi - math.asin(x)
    elif x < 0 and z < 0:
        # 3rd quadrant
        heading = math.pi - math.asin(x)
    else:
        # 4th quadrant
        heading = 2 * math.pi + math.asin(x)

    return heading


class RFactor2Environment(gym.Env):
    def __init__(self):
        # Steering and Acceleration
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1, 2), dtype=float)
        # Velocity and Heading (X and Z axis)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=float)
        # Vjoy Device
        self.vjoy_device = pyvjoy.VJoyDevice(1)
        # Previous state (for reward purposes)
        self.prev_state = None

    # Calculated based on how much distance was advanced since last state
    def calculate_reward(self, state):
        return state[4] - self.prev_state[4]

        # Checks if a lap is completed or if the agent goes backwards, or if it drives out of bounds

    def episode_finish(self, state):
        return (self.prev_state[4] - state[4] > 0) or abs(state[5]) >= 9.0

    def obtain_state(self):
        win32event.WaitForSingleObject([vehicleScoringH], False, win32event.INFINITE)
        win32event.WaitForSingleObject([telemetryH], False, win32event.INFINITE)

        telemetry_data = telemetryMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
        telemetryMMfile.seek(0)

        vehicle_data = vehicleScoringMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
        vehicleScoringMMfile.seek(0)

        # Position
        position = [float(telemetry_data[0]), float(telemetry_data[2])]
        # Angle
        orientation = [float(telemetry_data[3]), float(telemetry_data[5])]
        heading = calculate_heading(orientation[0], orientation[1])
        # Velocity
        velocity = -float(telemetry_data[8])
        # Acceleration
        acceleration = -float(telemetry_data[11])
        # Lap Distance
        lap_dist = float(vehicle_data[0])
        # Path Lateral - Distance to center of track
        path_lateral = float(vehicle_data[1])

        # Compile information into state variable (Maybe turn into class/dict)
        state = [position, heading, velocity, acceleration, lap_dist, path_lateral]

        if self.prev_state is None:
            self.prev_state = state

        reward = self.calculate_reward(state)
        done = self.episode_finish(state)

        print(f"Position: {position}")
        print(f"Heading: {heading}")
        print(f"Velocity: {velocity}")  # In m/s
        print(f"Acceleration: {acceleration}")  # In m/s^2
        print(f"Path Lateral: {path_lateral}")
        print(f"Reward: {reward}")

        # Update previous state and Reset Event
        self.prev_state = state
        win32event.ResetEvent(telemetryH)
        return state, reward, done

    def reset(self, seed=None, options=None):
        # Send F command (resets race) via VJoy
        self.vjoy_device.set_button(1, 1)

        # Obtain observations related to reset
        new_state, _, _ = self.obtain_state()
        self.prev_state = new_state

        # Turn reset button off
        self.vjoy_device.set_button(1, 0)

        return new_state

    # actions = [steering_action, throttle_action]
    def step(self, actions):
        # Execute the action using VJoy
        self.vjoy_device.data.wAxisX = int(16384 + float(actions[0]) * 16384)
        self.vjoy_device.data.wAxisY = int(16384 + float(actions[1]) * 16384)

        # Obtain in-game data (after action execution, it waits until the info is received)
        time.sleep(0.01)
        new_state, reward, done = self.obtain_state()

        return new_state, reward, done


# Register Custom Environment
register(
    id='rFactor2Environment-v0',
    entry_point='envs.custom_env:rFactor2Environment',
)
