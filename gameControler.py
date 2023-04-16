# Fidgrove
# Python VJoy interface to control simracing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import pyvjoy
import win32event
import json
import numpy
import mmap
import time
import os
import marshal
import struct
from simple_pid import PID

# Proportional - Response to changes in error
# Integral - Response to overtime and persistent errors
# Derivative - Response to sudden changes 
pidSteering = PID(0.06, 0, 0.05, setpoint=0)
pidThrottle = PID(0.05, 0, 0.05)

#scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
#scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def getTrajectory():
	with open('data.json', 'r') as f:
		data = json.load(f)

	lapDist = []
	positions = []
	for i in range(len(data)):
		lapDist.append(data[i]["carScoring"]["currentLapDist"])
		positions.append([data[i]["telemetry"]["positionX"], data[i]["telemetry"]["positionZ"]])

	# Removing duplicates from positions and obtain the indices from the unique elements
	positions, uniqueIndexes = numpy.unique(positions, return_index=True, axis=0)

	# Pruning lapDist array by selecting the elements corresponding to the unique indices obtained
	lapDist = numpy.array(lapDist)[uniqueIndexes]
	

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
	
	oldLapDist = 0
	distanceToTrajectory = 0
	while True:
		eventResult = win32event.WaitForMultipleObjects([telemetryH, vehicleScoringH], False , win32event.INFINITE)
		if(eventResult == win32event.WAIT_OBJECT_0):
			# Read telemetry data from shared memory
			data = telemetryMMfile.read().decode("utf-8").replace("\n", "").split(',')
			telemetryMMfile.seek(0)
	
			# Position
			posX = float(data[0])
			posY = float(data[1])
			posZ = float(data[2])

			# Angle
			oriX = float(data[3])
			# oriY = float(data[4])
			# oriZ = float(data[5])

			win32event.ResetEvent(telemetryH)

		elif(eventResult == win32event.WAIT_OBJECT_0+1):
			# Read vehicle scoring data from shared memory
			data = vehicleScoringMMfile.read().decode("utf-8").replace("\n", "").split(',')
			vehicleScoringMMfile.seek(0)

			
			lapDist = float(data[0])
			pathLateral = float(data[1])
			# trackEdge = float(data[2])
			
			calculateControl(vJoyDevice, pathLateral, lapDist)
			distanceToTrajectory += pathLateral

			if oldLapDist - lapDist > 0:
				timer, lapTime = calculateReward(timer)
				minutes, seconds = divmod(lapTime, 60)
				print(f"Lap Time: {str(int(minutes))}:{str(seconds)}")
				
			oldLapDist = lapDist
			time.sleep(0.01)
			win32event.ResetEvent(vehicleScoringH)

		else:
			print("WARN: Event wait timeout or event error")