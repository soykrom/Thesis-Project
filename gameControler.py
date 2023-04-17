# Fidgrove
# Python VJoy interface to control simracing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import pyvjoy
import win32event
import json
import numpy
import mmap
import time
from sklearn import gaussian_process as gp
import os
import marshal
import struct
from simple_pid import PID
import datetime

# Proportional - Response to changes in error
# Integral - Response to overtime and persistent errors
# Derivative - Response to sudden changes 
pidSteering = PID(0.05, 0, 0.05, setpoint=0)
pidThrottle = PID(0.05, 0, 0.05)

#scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
#scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def predict(lapDist, positions):
	epoch_time = time.time()
	gaussianProcess = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF())
	datetime_obj = datetime.datetime.fromtimestamp(epoch_time)

	# Print the current time in a human-readable format
	print("Current time: {}".format(datetime_obj.strftime('%Y-%m-%d %H:%M:%S')))
	gaussianProcess.fit(lapDist.reshape(-1, 1), positions)
	datetime_obj = datetime.datetime.fromtimestamp(epoch_time)

	# Print the current time in a human-readable format
	print("Current time: {}".format(datetime_obj.strftime('%Y-%m-%d %H:%M:%S')))
	# Evenly spaced points every half a meter
	interpolatedValues = numpy.linspace(0, numpy.max(lapDist), int(numpy.max(lapDist) * 0.5))
	posPredict = gaussianProcess.predict(interpolatedValues.reshape(-1, 1))

	print(f"{len(posPredict)}")

def getTrajectory():
	with open('data.json', 'r') as f:
		data = json.load(f)

	lapDist = []
	positions = []
	for i in range(len(data)):
		lapDist.append(data[i]["carScoring"]["currentLapDist"])
		positions.append([data[i]["telemetry"]["positionX"], data[i]["telemetry"]["positionZ"]])

	return numpy.array(lapDist), numpy.array(positions)

def calculatePathLateral(currentPosition, trajectoryPosition):
	return numpy.linalg.norm(trajectoryPosition - currentPosition)


def calculateControl(vJoyDevice, pathLateral, lapDist):
	errorSteering = pidSteering(pathLateral)

	pidThrottle.setpoint = lapDist + 20
	errorThrottle = pidThrottle(lapDist)
 
	steeringControl = 16384 - float(errorSteering) * 16384
	throttleControl = 16384 + float(errorThrottle) * 16384

	applyControl(vJoyDevice, steeringControl, throttleControl)
 
def applyControl(vJoyDevice, steeringControl, throttleControl):
	# Apply control
	vJoyDevice.data.wAxisX = int(steeringControl)
	vJoyDevice.data.wAxisY = int(throttleControl)

	vJoyDevice.update()

# Reward Calculation and Weight updating
def calculateReward(startTime):
    timer = time.time()

    return timer, time.time() - startTime
 
# def updateWeights(reward):
#     proportional += 0.01 * reward
#     integral += 0.001 * reward
#     derivative += 0.0001 * reward
#     
#     pid.tunings(proportional, integral, derivative)
	
# Main loop
def doMainLoop():
	# Setting up vJoy interface
	vJoyDevice = pyvjoy.VJoyDevice(1)

	# Setting up rFactor2 plugin reader
	telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventCarData")
	telemetryMMfile = mmap.mmap(-1, length=35, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

	vehicleScoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteVehicleScoring")
	vehicleScoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)

	timer = time.time()	
	
	lapDist, positions = getTrajectory()
	currentLapDist = 0
	while True:
		eventResult = win32event.WaitForMultipleObjects([telemetryH, vehicleScoringH], False , win32event.INFINITE)
		if(eventResult == win32event.WAIT_OBJECT_0):
			# Read telemetry data from shared memory
			data = telemetryMMfile.read().decode("utf-8").replace("\n", "").split(',')
			telemetryMMfile.seek(0)
	
			# Position
			posX = float(data[0])
			posZ = float(data[2])

			pathLateral = calculatePathLateral([posX, posZ], positions[numpy.where(lapDist > currentLapDist, numpy.abs(lapDist - currentLapDist), numpy.inf).argmin()])

			calculateControl(vJoyDevice, -pathLateral, currentLapDist)

			time.sleep(0.01)
			win32event.ResetEvent(telemetryH)

		# CURRENT PROBLEMS:
		#	1 - Euclidean distance is always positive so it will never turn left.
		#	2 - It seeing ahead/behind depending on currentLapDist and time to obtain position (in theory intervals should solve this)
		#	3 - Related to 2, obtaining a predictions vector where it complements the pieces of the previous vector where intervals are bigger than X
		
		elif(eventResult == win32event.WAIT_OBJECT_0+1):
			# Read vehicle scoring data from shared memory
			data = vehicleScoringMMfile.read().decode("utf-8").replace("\n", "").split(',')
			vehicleScoringMMfile.seek(0)
			
			currentLapDist = float(data[0]) + 1
			print(currentLapDist)

			win32event.ResetEvent(vehicleScoringH)

		else:
			print("WARN: Event wait timeout or event error")