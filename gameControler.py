# Fidgrove
# Python VJoy interface to control simracing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import pyvjoy
import win32event
import json
import numpy
import mmap
import time
import math
from sklearn import gaussian_process as gp
from sklearn import preprocessing
import os
import marshal
import struct
from simple_pid import PID
import datetime

# Proportional - Response to changes in error
# Integral - Response to overtime and persistent errors
# Derivative - Response to sudden changes 
pidSteering = PID(0.01, 0, 0.05, setpoint=0)
pidThrottle = PID(0.05, 0, 0.05)

#scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
#scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def predict(lapDist, positions):

	# Train the Gaussian Process
	gaussianProcess = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF())
	gaussianProcess.fit(lapDist.reshape(-1, 1), positions)

	# Array with points every 0.5 meters.
	newLapDist = numpy.linspace(lapDist.min(), lapDist.max(), int((lapDist.max() - lapDist.min()) // 0.5))

	# Predict the positions
	posPredict, deviation = gaussianProcess.predict(newLapDist.reshape(-1, 1), return_std=True)


	sortedIndices = numpy.argsort(newLapDist)
	newLapDist = newLapDist[sortedIndices]
	posPredict = posPredict[sortedIndices]

	testLapDist = lapDist[3452]
	testPosition = positions[3452]

	index = numpy.argmin(numpy.abs(newLapDist - testLapDist))
	predictedLapDist = newLapDist[index]
	predictedPosition = posPredict[index]

	print("Standard Deviation: ", deviation)
	print(f"""TestLapDist: {testLapDist}\tTestPosition: {testPosition}
       PredictedLapDist: {predictedLapDist}\tPredictedPositon: {predictedPosition}""")

def getTrajectory(filename='data.json'):
	with open(filename, 'r') as file:
		data = json.load(file)

	lapDist = numpy.empty(len(data), dtype=numpy.float64)
	positions = numpy.empty((len(data), 2), dtype=numpy.float64)
	for i in range(len(data)):
		lapDist[i] = data[i]["carScoring"]["currentLapDist"]
		positions[i] = [data[i]["telemetry"]["positionX"], data[i]["telemetry"]["positionZ"]]

	# Remove duplicate values based on the positions vector (it keeps far more values)
	positions, uniqueIndices = numpy.unique(numpy.array(positions), return_index=True, axis=0)
	lapDist = lapDist[uniqueIndices]

	# Sort both arrays in ascending order of LapDistance
	sortedIndices = numpy.argsort(lapDist)
	lapDist = lapDist[sortedIndices]
	positions = positions[sortedIndices]

	predict(lapDist, positions)


def calculatePathLateral(currentPosition, trajectoryPosition):
	# Distance from current point to desired trajectory point
	pathLateralDistance = numpy.linalg.norm(currentPosition - trajectoryPosition)

	# Calculate if it's to turn left or right
	# ???????

	# return pathLateral

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

			index = numpy.argmin(numpy.abs(lapDist - currentLapDist))
			if lapDist[index] < currentLapDist:
				index += 1

			pathLateral = calculatePathLateral([posX, posZ], positions[index])

			calculateControl(vJoyDevice, pathLateral, currentLapDist)

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
			
			currentLapDist = float(data[0])
			# print(currentLapDist)

			win32event.ResetEvent(vehicleScoringH)

		else:
			print("WARN: Event wait timeout or event error")