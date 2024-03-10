# Fidgrove
# Python VJoy interface to control simracing cars from python
# Based on https://github.com/maxofbritton/pyvjoy
import math
import pickle

import pyvjoy
import win32event
import mmap
import time
from simple_pid import PID

# Proportional - Response to changes in error
# Integral - Response to overtime and persistent errors
# Derivative - Response to sudden changes
pidSteering = PID(0.06, 0, 0.05, setpoint=0)
pidThrottle = PID(0.05, 0, 0.05)

# Setting up vJoy interface
j = pyvjoy.VJoyDevice(1)

# Setting up rFactor2 plugin reader
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=80, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

vehicleScoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteVehicleScoring")
vehicleScoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)


# scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
# scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)


def calculate_control(pathLateral, lapDist):
    error_steering = pidSteering(pathLateral)

    pidThrottle.setpoint = lapDist + 15
    error_throttle = pidThrottle(lapDist)

    steering_control = 16384 - float(error_steering) * 16384
    throttle_control = 16384 + float(error_throttle) * 16384

    apply_control(steering_control, throttle_control)


def apply_control(steeringControl, throttleControl):
    # Apply control
    j.data.wAxisX = int(steeringControl)
    j.data.wAxisY = int(throttleControl)

    j.update()


# Reward Calculation and Weight updating
def calculate_reward(startTime):
    timer = time.time()

    return timer, time.time() - startTime


# def updateWeights(reward):
#     proportional += 0.01 * reward
#     integral += 0.001 * reward
#     derivative += 0.0001 * reward
#
#     pid.tunings(proportional, integral, derivative)
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


# Main loop
def do_main_loop():
    timer = time.time()
    old_lap_dist = 0

    state = []
    count = 0
    while True:
        win32event.WaitForMultipleObjects([vehicleScoringH, telemetryH], True, win32event.INFINITE)

        telemetry_data = telemetryMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
        telemetryMMfile.seek(0)

        vehicle_data = vehicleScoringMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
        vehicleScoringMMfile.seek(0)  # Read telemetry data from shared memory

        # Position
        pos_x = round(float(telemetry_data[0]), 2)
        pos_y = round(float(telemetry_data[1]), 2)
        pos_z = round(float(telemetry_data[2]), 2)

        # Angle
        orientation = [float(telemetry_data[3]), float(telemetry_data[5])]
        heading = round(calculate_heading(orientation[0], orientation[1]), 2)

        win32event.ResetEvent(telemetryH)

        lap_dist = round(float(vehicle_data[0]), 2)
        path_lateral = round(float(vehicle_data[1]), 2)
        # trackEdge = float(data[2])

        calculate_control(path_lateral, lap_dist)

        state.append([[pos_x, pos_y, pos_z], lap_dist, path_lateral, heading])

        if old_lap_dist - lap_dist > 0:
            timer, lap_time = calculate_reward(timer)
            minutes, seconds = divmod(lap_time, 60)
            print(f"Lap Time: {str(int(minutes))}:{str(seconds)}")

        old_lap_dist = lap_dist
        time.sleep(0.01)
        win32event.ResetEvent(vehicleScoringH)

        count += 1

        if count % 10000 == 0:
            break

    start = []
    reset = []
    for i in range(1, len(state)):
        if state[i][1] == state[i - 1][1]:
            start.append(i)

        if state[i][1] < state[i - 1][1]:
            reset.append(i)

    start_index = start.pop()
    reset.pop()
    low_index = reset.pop()
    data = []
    for i in range(len(state)):
        if start_index < i < low_index:
            data.append(state[i])

    with open('states.pkl', 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    do_main_loop()
exit()
