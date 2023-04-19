# Fidgrove
# Python VJoy interface to control sim-racing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import json
import mmap
import time

import numpy
import pyvjoy
import win32event
from simple_pid import PID
from sklearn import gaussian_process as gp

# Proportional - Response to changes in error
# Integral - Response to overtime and persistent errors
# Derivative - Response to sudden changes 
pidSteering = PID(0.01, 0, 0.05, setpoint=0)
pidThrottle = PID(0.05, 0, 0.05)


# scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
# scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def predict_position(gp_model, current_lap_dist):
    pos_predict = gp_model.predict([[current_lap_dist]])
    return pos_predict


def get_gaussian_process_regressor(lap_dist, positions):
    # Train the Gaussian Process
    gpr_model = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2)))
    gpr_model.fit(lap_dist.reshape(-1, 1), positions)

    return gpr_model


# Array with points every 0.5 meters.
# newLapDist = numpy.linspace(lapDist.min(), lapDist.max(), int((lapDist.max() - lapDist.min()) // 0.5))

# Predict the positions
# posPredict, deviation = gaussianProcess.predict(newLapDist.reshape(-1, 1), return_std=True)

# sortedIndices = numpy.argsort(newLapDist)
# newLapDist = newLapDist[sortedIndices]
# posPredict = posPredict[sortedIndices]

# testLapDist = lapDist[3452]
# testPosition = positions[3452]

# index = numpy.argmin(numpy.abs(newLapDist - testLapDist))
# predictedLapDist = newLapDist[index]
# predictedPosition = posPredict[index]

# print("Standard Deviation: ", deviation)
# print(f"""TestLapDist: {testLapDist}\tTestPosition: {testPosition}
#    PredictedLapDist: {predictedLapDist}\tPredictedPosition: {predictedPosition}""")

def get_trajectory(filename='data.json'):
    with open(filename, 'r') as file:
        data = json.load(file)

    lap_dist = numpy.empty(len(data), dtype=numpy.float16)
    positions = numpy.empty((len(data), 2), dtype=numpy.float16)
    for i in range(len(data)):
        lap_dist[i] = data[i]["carScoring"]["currentLapDist"]
        positions[i] = [data[i]["telemetry"]["positionX"], data[i]["telemetry"]["positionZ"]]

    # Remove duplicate values based on the positions vector (it keeps far more values)
    # positions, uniqueIndices = numpy.unique(numpy.array(positions), return_index=True, axis=0)
    # lapDist = lapDist[uniqueIndices]
    #
    # # Sort both arrays in ascending order of LapDistance
    # sortedIndices = numpy.argsort(lapDist)
    # lapDist = lapDist[sortedIndices]
    # positions = positions[sortedIndices]

    return lap_dist, positions


def calculate_path_lateral(current_position, trajectory_position):
    # Distance from current point to desired trajectory point
    path_lateral_distance = numpy.linalg.norm(current_position - trajectory_position)

    return path_lateral_distance


# Calculate if it's to turn left or right
# ???????

# return pathLateral

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
    print("Setting up vJoy and rFactor2 plugin")
    # Setting up vJoy interface
    vjoy_device = pyvjoy.VJoyDevice(1)

    # Setting up rFactor2 plugin reader
    telemetry_h = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
    telemetry_m_mfile = mmap.mmap(-1, length=35, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

    vehicle_scoring_h = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteVehicleScoring")
    vehicle_scoring_m_mfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)

    # timer = time.time()

    lap_dist, positions = get_trajectory()
    current_lap_dist = lap_dist[0]
    print("Beginning training of gaussian process model")
    gp_model = get_gaussian_process_regressor(lap_dist, positions)
    print("Training Done")
    user_input = input("Press Enter to continue")
    while user_input != "":
        user_input = input("Press Enter to continue")

    print("Entering Main Loop")
    while True:
        event_result = win32event.WaitForMultipleObjects([telemetry_h, vehicle_scoring_h], False, win32event.INFINITE)
        if event_result == win32event.WAIT_OBJECT_0:
            # Read telemetry data from shared memory
            data = telemetry_m_mfile.read().decode("utf-8").replace("\n", "").split(',')
            telemetry_m_mfile.seek(0)

            # Position
            position = [float(data[0]), float(data[2])]

            index = numpy.argmin(numpy.abs(lap_dist - current_lap_dist))
            if lap_dist[index] < current_lap_dist:
                index += 1

            pos_predict = predict_position(gp_model, current_lap_dist)

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
