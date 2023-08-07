# Fidgrove
# Python VJoy interface to control sim-racing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import json
import mmap
import time
import random
import pickle
import numpy
import math
import pyvjoy
import win32event
from simple_pid import PID
from sklearn import gaussian_process as gp
from sklearn import preprocessing

# Proportional - Response to changes in error
# Integral - Response to overtime and persistent errors
# Derivative - Response to sudden changes 
pidSteering = PID(0.3, 0, 0.01, setpoint=0)
pidThrottle = PID(0.05, 0, 0.05, setpoint=20)


# scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
# scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def test(gpr_model, scaler, lap_dist, positions):
    # Generate 10 random integer values between 0 and 360
    random_values = [random.randint(0, 360) for _ in range(10)]

    # Print the random values
    print(random_values)
    for el in random_values:
        el = (el + 180) % 360 - 180
        print(el)

    return


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

    return lap_dist, positions, scaler


def get_grp_model(filename='gpr_model.pkl'):
    with open(filename, 'rb') as f:
        gpr_model = pickle.load(f)
    return gpr_model


def train_gpr_model(filename='gpr_model.pkl'):
    lap_dist, positions, scaler = get_trajectory()

    lap_dist_scaled = scaler.transform(lap_dist)

    # Train the Gaussian Process
    print("Beginning training of gaussian process model")
    kernel = gp.kernels.Matern(length_scale=0.5)
    gpr_model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=10, random_state=0)
    gpr_model.fit(lap_dist_scaled, positions)
    print("Training Done")

    with open(filename, 'wb') as f:
        pickle.dump(gpr_model, f)


def predict_position(gpr_model, scaler, lap_dist):
    lap_dist = numpy.array(lap_dist).reshape(-1, 1)

    lap_dist_scaled = scaler.transform(lap_dist)

    pos_predict = gpr_model.predict(lap_dist_scaled)

    return numpy.array(pos_predict)


def calculate_heading(ori_x, ori_z):
    heading_angle = 0

    if ori_x > 0 and ori_z > 0:
        print("1st quadrant")
        heading_angle = math.asin(ori_x)

    elif ori_x > 0 and ori_z < 0:
        print("2nd quadrant")
        heading_angle = (math.pi - math.asin(ori_x))

    elif ori_x < 0 and ori_z < 0:
        print("3rd quadrant")
        heading_angle = (math.pi - math.asin(ori_x))

    elif ori_x < 0 and ori_z > 0:
        print("4th quadrant")
        heading_angle = (2 * math.pi + math.asin(ori_x))

    return heading_angle


def calculate_look_ahead(current_velocity=60):
    constant = 0.1

    # Get look ahead distance
    look_ahead = constant * current_velocity

    return look_ahead


def calculate_steering(current_position, current_heading, target_point):
    # Axis distance from current position to the target point and angle
    tp_x = target_point[0] - current_position[0]
    tp_y = target_point[1] - current_position[1]

    print("Current: ", current_heading, " = ", numpy.rad2deg(current_heading))
    target_angle = -numpy.rad2deg(math.atan2(tp_y, tp_x))
    print("Target: ", math.atan2(tp_y, tp_x), " = ", target_angle)

    min_angle = target_angle - numpy.rad2deg(current_heading)

    print("Min: ", min_angle)

    return min_angle


def calculate_control(vjoy_device, delta_angle, current_lap_dist):
    error_steering = pidSteering(delta_angle)

    error_throttle = pidThrottle(current_lap_dist)

    steering_control = 16384 + float(error_steering) * 16384
    throttle_control = 16384 + float(error_throttle) * 16384

    print(f"Error: {error_steering}\tControl: {steering_control}")

    apply_control(vjoy_device, steering_control, throttle_control)


def apply_control(vjoy_device, steering_control, throttle_control):
    # Apply control
    vjoy_device.data.wAxisX = int(steering_control)
    vjoy_device.data.wAxisY = int(throttle_control)

    vjoy_device.update()


# Main loop
def do_main_loop(flag=0):
    print("Getting trajectory and scaler")
    lap_dist, positions, scaler = get_trajectory()
    current_lap_dist = lap_dist[0]

    print("Getting gpr_model")
    gpr_model = get_grp_model()

    if flag == 1:
        print("Testing time")
        test(gpr_model, scaler, lap_dist, positions)

    user_input = input("Press Enter to continue")
    while user_input != "":
        user_input = input("Press Enter to continue")

    print("Setting up vJoy and rFactor2 plugin")
    # Setting up vJoy interface
    vjoy_device = pyvjoy.VJoyDevice(1)

    # Setting up rFactor2 plugin reader
    telemetry_h = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0, "WriteEventCarData")
    telemetry_m_mfile = mmap.mmap(-1, length=40, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

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
            current_position = numpy.array([float(data[0]), float(data[2])])

            # Current Angle
            ori_x = float(data[3])
            ori_z = float(data[5].rstrip('\x00'))

            look_ahead = calculate_look_ahead()  # FIX TO CURRENT_VELOCITY
            target_point = predict_position(gpr_model, scaler, current_lap_dist + look_ahead)

            current_heading = calculate_heading(ori_x, ori_z)
            turn_error = calculate_steering(current_position, current_heading, target_point[0])

            # print(f"Current: {math.degrees(current_heading)}\tError: {turn_error}")

            calculate_control(vjoy_device, numpy.deg2rad(turn_error), current_lap_dist)

            time.sleep(0.2)
            win32event.ResetEvent(telemetry_h)

        elif event_result == win32event.WAIT_OBJECT_0 + 1:
            # Read vehicle scoring data from shared memory
            data = vehicle_scoring_m_mfile.read().decode("utf-8").replace("\n", "").split(',')
            vehicle_scoring_m_mfile.seek(0)

            current_lap_dist = float(data[0])

            win32event.ResetEvent(vehicle_scoring_h)

        else:
            print("WARN: Event wait timeout or event error")
