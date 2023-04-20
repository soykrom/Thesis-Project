# Fidgrove
# Python VJoy interface to control sim-racing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import json
import mmap
import time

import pickle
import numpy
import pyvjoy
import win32event
from simple_pid import PID
from sklearn import gaussian_process as gp
from sklearn import preprocessing

# Proportional - Response to changes in error
# Integral - Response to overtime and persistent errors
# Derivative - Response to sudden changes 
pidSteering = PID(0.01, 0, 0.05, setpoint=0)
pidThrottle = PID(0.05, 0, 0.05)


# scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
# scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def predict_position(gpr_model, current_lap_dist, scaler):

    current_lap_dist = numpy.array(current_lap_dist).reshape(-1, 1)
    print("Current: ", current_lap_dist)

    lap_dist_scaled = scaler.transform(current_lap_dist)
#    lap_dist_min = numpy.percentile(lap_dist, 1)
#    lap_dist_max = numpy.percentile(lap_dist, 99)

#    print(f"Min: {lap_dist_min}\tMax: {lap_dist_max}")

#    lap_dist_scaled = (current_lap_dist - lap_dist_min) / (lap_dist_max - lap_dist_min)
    print("Scaled: ", lap_dist_scaled)

    pos_predict = gpr_model.predict(lap_dist_scaled)
    return numpy.array(pos_predict)


def train_gpr_model(filename='gpr_model.pkl'):
    lap_dist, positions, scaler = get_trajectory()

    lap_dist_scaled = scaler.transform(lap_dist)

#    lap_dist_min = numpy.percentile(lap_dist, 1)
#    lap_dist_max = numpy.percentile(lap_dist, 99)
#    lap_dist_scaled = (lap_dist - lap_dist_min) / (lap_dist_max - lap_dist_min)

    # Train the Gaussian Process
    print("Beginning training of gaussian process model")
    kernel = gp.kernels.Matern(length_scale=0.5)
    gpr_model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=10, random_state=0)
    gpr_model.fit(lap_dist_scaled, positions)
    print("Training Done")

    with open(filename, 'wb') as f:
        pickle.dump(gpr_model, f)


def get_grp_model(filename='gpr_model.pkl'):
    with open(filename, 'rb') as f:
        gpr_model = pickle.load(f)
    return gpr_model


def get_trajectory(filename='data.json'):
    with open(filename, 'r') as file:
        data = json.load(file)

    lap_dist = numpy.empty(len(data), dtype=numpy.float32)
    positions = numpy.empty((len(data), 2), dtype=numpy.float32)
    for i in range(len(data)):
        lap_dist[i] = data[i]["carScoring"]["currentLapDist"]
        positions[i] = [data[i]["telemetry"]["positionX"], data[i]["telemetry"]["positionZ"]]

    # Pre process data
    lap_dist = lap_dist.reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(lap_dist)
    # Remove duplicate values based on the positions vector (it keeps far more values)
#    positions, unique_indices = numpy.unique(numpy.array(positions), return_index=True, axis=0)
#    lap_dist = lap_dist[unique_indices]
#
#    # Sort both arrays in ascending order of LapDistance
#    sorted_indices = numpy.argsort(lap_dist)
#    lap_dist = lap_dist[sorted_indices]
#    positions = positions[sorted_indices]
#
#    print(len(lap_dist))
#    print(len(positions))
    return lap_dist, positions, scaler


def calculate_path_lateral(current_position, trajectory_position):
    # Distance from current point to desired trajectory point
    path_lateral_distance = numpy.linalg.norm(current_position - trajectory_position)

    # Calculate if it's to turn left or right
    # ???????
    # Profit

    return path_lateral_distance


def calculate_control(vjoy_device, path_lateral, lap_dist):
    error_steering = pidSteering(path_lateral)

    pidThrottle.setpoint = lap_dist + 20
    error_throttle = pidThrottle(lap_dist)

    steering_control = 16384 - float(error_steering) * 16384
    throttle_control = 16384 + float(error_throttle) * 16384

    apply_control(vjoy_device, steering_control, throttle_control)


def apply_control(vjoy_device, steering_control, throttle_control):
    # Apply control
    vjoy_device.data.wAxisX = int(steering_control)
    vjoy_device.data.wAxisY = int(throttle_control)

    vjoy_device.update()


# Reward Calculation and Weight updating
def calculate_reward(start_time):
    timer = time.time()

    return timer, time.time() - start_time


# def updateWeights(reward):
#     proportional += 0.01 * reward
#     integral += 0.001 * reward
#     derivative += 0.0001 * reward
#     
#     pid.tunings(proportional, integral, derivative)

# Main loop
def do_main_loop():
    print("Getting trajectory and scaler")
    lap_dist, positions, scaler = get_trajectory()
    current_lap_dist = lap_dist[0]

    print("Getting gpr_model")
    gp_model = get_grp_model()

    user_input = input("Press Enter to continue")
    while user_input != "":
        user_input = input("Press Enter to continue")

    print("Setting up vJoy and rFactor2 plugin")
    # Setting up vJoy interface
    vjoy_device = pyvjoy.VJoyDevice(1)

    # Setting up rFactor2 plugin reader
    telemetry_h = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
    telemetry_m_mfile = mmap.mmap(-1, length=35, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

    vehicle_scoring_h = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteVehicleScoring")
    vehicle_scoring_m_mfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)

    print("Entering Main Loop")
    while True:
        event_result = win32event.WaitForMultipleObjects([telemetry_h, vehicle_scoring_h], False, win32event.INFINITE)
        if event_result == win32event.WAIT_OBJECT_0:
            # Read telemetry data from shared memory
            data = telemetry_m_mfile.read().decode("utf-8").replace("\n", "").split(',')
            telemetry_m_mfile.seek(0)

            # Position
            position = numpy.array([float(data[0]), float(data[2])])

            index = numpy.argmin(numpy.abs(lap_dist - current_lap_dist))
            if lap_dist[index] < current_lap_dist:
                index += 1

            pos_predict = predict_position(gp_model, current_lap_dist, scaler)

            print(f"""Test Lap Dist: {lap_dist[index]}\tTest Position: {positions[index]}
            Current Lap Dist: {current_lap_dist}\tPredicted Positions: {pos_predict}""")

            path_lateral = calculate_path_lateral(position, pos_predict)

            calculate_control(vjoy_device, path_lateral, current_lap_dist)

            time.sleep(1)
            win32event.ResetEvent(telemetry_h)

        # CURRENT PROBLEMS: 1 - Euclidean distance is always positive, so it will never turn left. 2 - It seeing
        # ahead/behind depending on currentLapDist and time to obtain position (in theory intervals should solve
        # this) 3 - Related to 2, obtaining a predictions vector where it complements the pieces of the previous
        # vector where intervals are bigger than X

        elif event_result == win32event.WAIT_OBJECT_0 + 1:
            # Read vehicle scoring data from shared memory
            data = vehicle_scoring_m_mfile.read().decode("utf-8").replace("\n", "").split(',')
            vehicle_scoring_m_mfile.seek(0)

            current_lap_dist = float(data[0])

            win32event.ResetEvent(vehicle_scoring_h)

        else:
            print("WARN: Event wait timeout or event error")
