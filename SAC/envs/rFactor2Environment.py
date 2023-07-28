import gym
from gym import spaces
from gym.envs.registration import register
import win32event
import mmap
import math

# Setting up rFactor2 plugin reader
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=73, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)


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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
        # Velocity and Heading (X and Z axis)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=float)

    def calculate_reward(self, state):
        return 1

    def step(self, action):
        steering_action, acceleration_action = action

        # Execute the action using VJoy

        # Obtain in-game data (after action execution, it waits until the info is received)
        win32event.WaitForSingleObject([telemetryH], False, win32event.INFINITE)

        data = telemetryMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
        telemetryMMfile.seek(0)

        # Position
        position = [float(data[0]), float(data[2])]
        # Angle
        orientation = [float(data[3]), float(data[5])]
        heading = calculate_heading(orientation[0], orientation[1])
        # Velocity
        velocity = -float(data[8])
        # Acceleration
        acceleration = -float(data[11])

        # Compile information into state variable (Maybe turn into class/dict)
        state = [position, heading, velocity, acceleration]

        print(f"Position: {position}")
        print(f"Heading: {heading}")
        print(f"Velocity: {velocity}")  # Car speed in m/s
        print(f"Acceleration: {acceleration}")  # Car acceleration m/s^2

        # Calculate reward
        reward = self.calculate_reward(state)

        if self.is_done():
            self.reset_race()
            # Obtain observations related to reset

        # Reset Event
        win32event.ResetEvent(telemetryH)

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
