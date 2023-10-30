# Fidgrove
# Python VJoy interface to control sim racing cars from python
# Based on https://github.com/maxofbritton/pyvjoy
import time

# import pyvjoy
import win32event
import math
import mmap
from simple_pid import PID
import fidgrovePluginUtils as utils

pid = PID(0.05, 0, 0.05, setpoint=1)

# Setting up vJoy interface
# j = pyvjoy.VJoyDevice(1)
# j.data

# scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
# scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def calculate_heading(ori_x, ori_z):
    heading_angle = 0

    if ori_x > 0 and ori_z > 0:
        # print("1st quadrant")
        heading_angle = math.asin(ori_x)

    elif ori_x > 0 > ori_z:
        # print("2nd quadrant")
        heading_angle = (math.pi - math.asin(ori_x))

    elif ori_x < 0 and ori_z < 0:
        # print("3rd quadrant")
        heading_angle = (math.pi - math.asin(ori_x))

    elif ori_x < 0 < ori_z:
        # print("4th quadrant")
        heading_angle = (2 * math.pi + math.asin(ori_x))

    return heading_angle


previous_lap_dist = 0
# Main loop
while True:
    eventResult = win32event.WaitForMultipleObjects([utils.telemetryH, utils.vehicleScoringH], False, win32event.INFINITE)
    if eventResult == win32event.WAIT_OBJECT_0:
        # Read telemetry data from shared memory
        # print("received telemetry data")
        data = utils.telemetryMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
        utils.telemetryMMfile.seek(0)

        # Position
        posX = float(data[0])
        posZ = float(data[2])
        position = [posX, posZ]

        # Angle
        oriX = float(data[3])
        oriZ = float(data[5])

        heading = calculate_heading(oriX, oriZ)

        # Velocity
        velZ = -float(data[8])

        # Acceleration
        accZ = -float(data[11])

        # print(f"Position: {position}")
        # print(f"Heading: {heading}")
        # print(f"Velocity: {velZ}")  # In m/s
        # print(f"Acceleration: {accZ}")  # In m/s^2

        time.sleep(0.2)
        win32event.ResetEvent(utils.telemetryH)

    elif eventResult == win32event.WAIT_OBJECT_0 + 1:
        # print("received vehicle scoring data")
        # Read vehicle scoring data from shared memory
        data = utils.vehicleScoringMMfile.read().decode("utf-8").replace("\n", "").split(',')
        utils.vehicleScoringMMfile.seek(0)

        lapDist = float(data[0])
        pathLateral = float(data[1])
        trackEdge = float(data[2].rstrip('\x00'))

        # print(f"Lap Distance: {lapDist}\tPrevious Lap Distance: {previous_lap_dist}")
        print(f"Path Lateral: {pathLateral}")

        previous_lap_dist = lapDist

        time.sleep(0.2)
        win32event.ResetEvent(utils.vehicleScoringH)

    # elif(eventResult == win32event.WAIT_OBJECT_0+2):
    # print("received scoring data")
    # Read  scoring data from shared memory
    # data = scoringMMfile.read().decode("utf-8").replace("\n", "").split(',')
    # scoringMMfile.seek(0)

    # print(data)
    # win32event.ResetEvent(scoringH)

    else:
        print("WARN: Event wait timeout or event error")
