import math
import mmap
import os.path
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import win32event

# rFactor2 plugin Setup
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=80, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

vehicleScoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteVehicleScoring")
vehicleScoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)

# CONSTANTS
ACTION_TIMEOUT_LIMIT = 100
CO_PL, CO_DIST, CO_VEL, CO_DONE = 0.8, 1.7, 0.8, 0.75  # Reward Coefficients default values
SPEED_LIMIT = 50  # Km/h

# Normalization values
with open(os.path.abspath('environment/common/scale_factors.pkl'), 'rb') as file:
    scaling_factors = pickle.load(file)
    min_values = pickle.load(file)


def load_coefficients(coefficients):
    global CO_PL, CO_DIST, CO_DONE

    CO_PL, CO_DIST, CO_DONE = coefficients


def load_initial(file_path):
    print("Loading training file")
    with open(file_path, 'rb') as filename:
        agent = pickle.load(filename)

    return agent


def save_initial(file_path, agent):
    print("Saving training file")
    with open(file_path, 'wb') as filename:
        pickle.dump(agent, filename)


def plot(previous_states_df, agent):
    print("Plotting...")
    num_samples = len(previous_states_df)

    selected_indices = np.random.choice(len(np.array(previous_states_df)), num_samples, replace=False)
    state_samples = np.array([np.array(previous_states_df)[i] for i in selected_indices], dtype=float)

    # Initialize arrays to store the action values for each state
    actions_steering = np.zeros(num_samples)
    actions_throttle = np.zeros(num_samples)

    # Calculate the actions for each state based on your policy
    for i in range(num_samples):
        action = agent.choose_action(scale_features(state_samples[i]))

        actions_steering[i] = action
    #        actions_throttle[i] = action[1]

    dist_state_samples = [el[2] for el in state_samples]

    # Create the action heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Plot data on the first subplot
    ax1.scatter(dist_state_samples, actions_steering)
    ax1.set_title('Steering Actions')
    ax1.set_xlabel('Lap Distance')
    ax1.set_ylabel('Steering')

    # # Plot data on the second subplot
    # ax2.scatter(dist_state_samples, actions_throttle)
    # ax2.set_title('Throttle Actions')
    # ax2.set_xlabel('Distance')
    # ax2.set_ylabel('Throttle')

    # Display both subplots using a single plt.show() call
    plt.show()

    with open(os.path.join('environment/common/lists.pkl'), 'wb') as filename:
        pickle.dump(state_samples, filename)
        pickle.dump(actions_steering, filename)
        pickle.dump(actions_throttle, filename)


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


def convert_mps_to_kph(velocity):
    return velocity * 3.6


def calculate_throttle_action(speed):
    diff = SPEED_LIMIT - speed

    action = 1.4 * diff / max(SPEED_LIMIT, speed)

    return max(-1, min(1, action))


# Calculated based on how much distance was advanced since last state and current velocity
def calculate_reward(prev_state, state):
    lap_dist_prev = float(prev_state[2])
    lap_dist_new = float(state[2])
    vel = float(state[1])
    pl = float(state[3])

    # Necessary because of resetting and plugin interaction
    if abs(lap_dist_new - lap_dist_prev) > 500:
        return 0
    else:
        penalty = 0

    reward = CO_DIST * (lap_dist_new - lap_dist_prev) + \
             CO_VEL * vel - \
             CO_PL * abs(pl) - \
             CO_DONE * penalty

    # print(f"Reward: {reward}")

    return reward


count = 0
timeout_dist = 0


def episode_finish(prev_state, state):
    cond_timeout = False
    global count
    global timeout_dist
    lap_dist_prev = float(prev_state[2])
    lap_dist_new = float(state[2])
    pl = float(state[3])

    if count == 0:
        timeout_dist = lap_dist_new

    count += 1

    # print(f"Difference: {lap_dist_prev - lap_dist_new}\tPath Lateral: {pl}")
    if count % ACTION_TIMEOUT_LIMIT == 0:
        cond_timeout = abs(timeout_dist - lap_dist_new) < 30
        timeout_dist = lap_dist_new

    cond_pl = abs(pl) >= 8.0
    cond_start_pl = lap_dist_new < 200 and pl > 5.5
    cond_finish = lap_dist_prev > lap_dist_new and count > 800

    done = (cond_pl or cond_start_pl or cond_timeout or cond_finish) and count > 20
    if done:
        print("PL: ", cond_pl)
        print("Start: ", cond_start_pl)
        print("Timeout: ", cond_timeout)
        print("Finish: ", cond_finish)
        count = 0

    return done


def scale_features(state):
    scaled_state = [(state[i] - min_value_i) * scale_factor_i - 1.0 for i, scale_factor_i, min_value_i in
                    zip(range(len(state)), scaling_factors, min_values)]

    return np.array(scaled_state)


def obtain_state():
    win32event.WaitForMultipleObjects([vehicleScoringH, telemetryH], True, win32event.INFINITE)

    telemetry_data = telemetryMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
    telemetryMMfile.seek(0)

    vehicle_data = vehicleScoringMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
    vehicleScoringMMfile.seek(0)

    # Angle
    orientation = [float(telemetry_data[3]), float(telemetry_data[5])]
    heading = calculate_heading(orientation[0], orientation[1])
    # Velocity m/s
    velocity = -float(telemetry_data[8])
    # Lap Distance
    lap_dist = float(vehicle_data[0])
    # Path Lateral - Distance to center of track
    path_lateral = float(vehicle_data[1])

    # Compile information into state variable (Maybe turn into class/dict)
    state = np.array([heading, velocity, lap_dist, path_lateral])

    return state
