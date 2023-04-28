# Fidgrove
# Python VJoy interface to control sim-racing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import json
import mmap
import time
import matplotlib.pyplot as plt
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
pidSteering = PID(0.3, 0, 0, setpoint=0)
pidThrottle = PID(0.05, 0, 0.05, setpoint=20)


# scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
# scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def test(gpr_model, scaler, lap_dist, positions):
    k = 0.2
    max_value = numpy.max(lap_dist)

    points = numpy.arange(0, max_value, k).reshape(-1, 1)
    points_scaled = scaler.transform(points)

    predicted_positions = gpr_model.predict(points_scaled)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the values
    x, y = zip(*predicted_positions)
    ax.plot(x, y, color='red')

    x, y = zip(*positions)
    ax.plot(x, y, color='blue')

    # Hide the axis
    ax.axis('off')

    # Display the plot
    plt.show()

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


def calculate_look_ahead(current_velocity=60):
    constant = 0.1

    # Get look ahead distance
    look_ahead = constant * current_velocity

    return look_ahead


def calculate_steering_angle(current_position, target_point, look_ahead):
    car_length = 2.85

    # Axis distance from current position to the target point and angle
    tp_x = target_point[0] - current_position[0]

    curvature = 2 * tp_x / look_ahead ** 2

    steering_angle = math.atan(car_length * curvature)
    return steering_angle


def calculate_control(vjoy_device, delta_angle, current_lap_dist):
    error_steering = pidSteering(delta_angle)

    error_throttle = pidThrottle(current_lap_dist)

    steering_control = 16384 - float(error_steering) * 16384

    throttle_control = 16384 + float(error_throttle) * 16384

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
        test(gpr_model, scaler, lap_dist, positions)

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
            current_position = numpy.array([float(data[0]), float(data[2])])

            # Current Angle
            current_angle = -float(data[5].rstrip('\x00'))

            look_ahead = calculate_look_ahead()  # FIX TO CURRENT_VELOCITY
            target_point = predict_position(gpr_model, scaler, current_lap_dist + look_ahead)

            print(f"""Current: {current_lap_dist}\t{predict_position(gpr_model, scaler, current_lap_dist)[0]}
            Lookahead: {current_lap_dist + look_ahead}\t{target_point[0]}""")

            desired_angle = calculate_steering_angle(current_position, target_point[0], look_ahead)
            delta = desired_angle - current_angle

            print(f"""Current: {math.degrees(current_angle)}\tDesired: {math.degrees(desired_angle)}
            Delta: {math.degrees(delta)}\n\n""")

            calculate_control(vjoy_device, delta, current_lap_dist)

            win32event.ResetEvent(telemetry_h)

        elif event_result == win32event.WAIT_OBJECT_0 + 1:
            # Read vehicle scoring data from shared memory
            data = vehicle_scoring_m_mfile.read().decode("utf-8").replace("\n", "").split(',')
            vehicle_scoring_m_mfile.seek(0)

            current_lap_dist = float(data[0])

            win32event.ResetEvent(vehicle_scoring_h)

        else:
            print("WARN: Event wait timeout or event error")
