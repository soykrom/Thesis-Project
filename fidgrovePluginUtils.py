import win32event
import mmap
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

# rFactor2 plugin Setup
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=80, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

vehicleScoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteVehicleScoring")
vehicleScoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)

# Reward coefficients
c_vel = 1.2  # Velocity
c_pl = 1.0  # Path Lateral
c_dist = 1.5  # Distance

# Normalization values
with open('../scale_factors.pkl', 'rb') as file:
    scaling_factors = pickle.load(file)
    min_values = pickle.load(file)


def plot(previous_states_df, agent):
    num_samples = len(previous_states_df)

    selected_indices = np.random.choice(len(np.array(previous_states_df)), num_samples, replace=False)
    state_samples = [np.array(previous_states_df)[i] for i in selected_indices]

    # Initialize arrays to store the action values for each state
    actions_steering = np.zeros(num_samples)
    actions_throttle = np.zeros(num_samples)

    # Calculate the actions for each state based on your policy
    for i in range(num_samples):
        state = np.array(state_samples[i], dtype=float)
        action = agent.select_action(state)

        actions_steering[i] = action[0]
        actions_throttle[i] = action[1]

    dist_state_samples = [el[5] for el in state_samples]

    # Create the action heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Plot data on the first subplot
    ax1.scatter(dist_state_samples, actions_steering)
    ax1.set_title('Subplot 1')
    ax1.set_xlabel('Lap Distance')
    ax1.set_ylabel('Steering')

    # Plot data on the second subplot
    ax2.scatter(dist_state_samples, actions_throttle)
    ax2.set_title('Subplot 2')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Throttle')

    # Display both subplots using a single plt.show() call
    plt.show()

    with open('../lists.pkl', 'wb') as filename:
        pickle.dump(state_samples, filename)
        pickle.dump(actions_steering, filename)
        pickle.dump(actions_throttle, filename)


def process_transitions(actions_df, states_df, agent, memory, batch_size, updates_per_step):
    print("Processing initial transitions")
    timer = time.process_time()
    actions = []
    prev_states = []
    next_states = []
    updates = 0

    previous_states_df = states_df['Previous State'].apply(lambda x: x.strip('[]').split(','))
    new_states_df = states_df['New State'].apply(lambda x: x.strip('[]').split(','))

    for index, action in actions_df.iterrows():
        action = np.array(action)
        if action.ndim == 1:
            action = [action]
        else:
            action = action.tolist()

        prev_state = np.array(previous_states_df[index], dtype=float)
        new_state = np.array(new_states_df[index], dtype=float)

        actions.append(action)
        prev_states.append(prev_state)
        next_states.append(new_state)

        done = episode_finish(prev_state, new_state)
        reward = calculate_reward(prev_state, new_state, done)

        memory.push(prev_state, action, reward, new_state, float(not done))

        if len(memory) > batch_size:
            for i in range(updates_per_step):
                # Update parameters of all the networks
                agent.update_parameters(memory, batch_size, updates)
                updates += 1

    elapsed_time = time.process_time() - timer
    print(f"Initial inputs and parameter updates finished after {elapsed_time} seconds.")

    plot(previous_states_df, agent)

    return updates


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
    lap_dist_prev = float(prev_state[5])
    lap_dist_new = float(state[5])
    vel = float(state[3])
    pl = float(state[6])

    # Necessary because of resetting and plugin interaction
    if abs(lap_dist_new - lap_dist_prev) > 500:
        return 0

    reward = c_dist * (lap_dist_new - lap_dist_prev) + \
             c_vel * vel - \
             c_pl * abs(pl)

    return reward


# Checks if a lap is completed or if the agent goes backwards (with some margin and reset care)
# Or if it drives out of bounds
def episode_finish(prev_state, state):
    lap_dist_prev = float(prev_state[5])
    lap_dist_new = float(state[5])
    pl = float(state[6])

    # print(f"Difference: {lap_dist_prev - lap_dist_new}\tPath Lateral: {pl}")

    return (30.0 > lap_dist_prev - lap_dist_new > 0.25) or \
        abs(pl) >= 8.0 or \
        (lap_dist_new < 200 and pl > 5.5)


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
    state = np.array([position_x, position_y, heading, velocity, acceleration, lap_dist, path_lateral])

    return state
