import win32event
import mmap
import math

# rFactor2 plugin Setup
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=73, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

vehicleScoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteVehicleScoring")
vehicleScoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)

# Reward coefficients
c_vel = 1.2  # Velocity
c_pl = 0.65  # Path Lateral
c_dist = 1.5  # Distance
pen_dist = 3.0  # Penalty


def reset_events():
    win32event.ResetEvent(telemetryH)
    win32event.ResetEvent(vehicleScoringH)


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


# Calculated based on how much distance was advanced since last state and current velocity
def calculate_reward(prev_state, state, done):
    # Necessary because of resetting and plugin interaction
    if abs(state[5] - prev_state[5]) > 500:
        return 0

    reward = c_dist * (state[5] - prev_state[5]) + \
             c_vel * state[3] - \
             c_pl * abs(state[6])

    reward = reward if not done else reward - pen_dist * state[5]
    return reward


# Checks if a lap is completed or if the agent goes backwards (with some margin and reset care)
# Or if it drives out of bounds
def episode_finish(prev_state, state):
    print(f"Difference: {prev_state[5] - state[5]}\tPath Lateral: {state[6]}")

    return (30.0 > prev_state[5] - state[5] > 0.25) or \
        abs(state[6]) >= 8.0 or \
        (state[5] < 200 and state[6] > 5.5)


def obtain_state():
    win32event.WaitForMultipleObjects([vehicleScoringH, telemetryH], True, win32event.INFINITE)

    telemetry_data = telemetryMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
    telemetryMMfile.seek(0)

    vehicle_data = vehicleScoringMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
    vehicleScoringMMfile.seek(0)

    # Position
    position = [round(float(telemetry_data[0]), 2), round(float(telemetry_data[2]), 2)]
    position_x, position_y = position
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
    state = [position_x, position_y, heading, velocity, acceleration, lap_dist, path_lateral]

    return state
