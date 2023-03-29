# Fidgrove
# Python VJoy interface to control simracing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import pyvjoy
import win32event
import mmap
import time
import os
import marshal
import struct
from simple_pid import PID

# Proportional - Response to changes in error
# Integral - Response to overtime and persistent errors
# Derivative - Response to sudden changes 
pid = PID(0.05, 0, 0.05, setpoint=0)

# Setting up vJoy interface
j = pyvjoy.VJoyDevice(1)


# Setting up rFactor2 plugin reader
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=25, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

vehicleScoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteVehicleScoring")
vehicleScoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)

#scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
#scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)

def calculateSteeringControl(pathLateral):
	error = pid(pathLateral)
	print(error)
 
	steeringControl = 16384 - float(error) * 16384
	return steeringControl
 
def applySteeringControl(steeringControl):
	# Apply control
	j.data.wAxisX = int(steeringControl)
	j.update()

# Reward Calculation and Weight updating
# def calculateReward(state):
#     return state / 5
# 
# def updateWeights(reward):
#     proportional += 0.01 * reward
#     integral += 0.001 * reward
#     derivative += 0.0001 * reward
#     
#     pid.tunings(proportional, integral, derivative)

# Main loop
while True:
	eventResult = win32event.WaitForMultipleObjects([telemetryH, vehicleScoringH], False , win32event.INFINITE)
	if(eventResult == win32event.WAIT_OBJECT_0):
		# Read telemetry data from shared memory
		#print("received telemetry data")
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
		

		# print(data)
		win32event.ResetEvent(telemetryH)

	elif(eventResult == win32event.WAIT_OBJECT_0+1):
		#print("received vehicle scoring data")
		# Read vehicle scoring data from shared memory
		data = vehicleScoringMMfile.read().decode("utf-8").replace("\n", "").split(',')
		vehicleScoringMMfile.seek(0)

		lapDist = float(data[0])
		pathLateral = float(data[1])
		# trackEdge = float(data[2])
		
		steeringControl = calculateSteeringControl(pathLateral)

		applySteeringControl(steeringControl)
  
		time.sleep(0.01)
		win32event.ResetEvent(vehicleScoringH)
  
	

	#elif(eventResult == win32event.WAIT_OBJECT_0+2):
	#	#print("received scoring data")
	#	# Read  scoring data from shared memory
	#	data = scoringMMfile.read().decode("utf-8").replace("\n", "").split(',')
	#	scoringMMfile.seek(0)

	#	print(data)
	#	win32event.ResetEvent(scoringH)

	else:
		print("WARN: Event wait timeout or event error")

	

exit()









# Main loop
while True:
	eventResult = win32event.WaitForSingleObject(PyHANDLE, 1000)
	if(eventResult == win32event.WAIT_OBJECT_0):
		# Read data from shared memory
		data = telemetryMMfile.read().decode("utf-8").replace("\n", "").split(',')
		telemetryMMfile.seek(0)

		#lapDist = float(data[0])
		pathLateral = float(data[1])
		#trackEdge = float(data[2])

		# Controller
		steeringWheelControl = pid(pathLateral)

		# Apply control
		steeringWheelControlOutput = int((steeringWheelControl + 0.5) * 32768)
		#print(steeringWheelControlOutput)
		#j.data.wAxisX = steeringWheelControlOutput
		#j.update()

	else:
		print("WARN: Event wait timeout or event error")

	win32event.ResetEvent(PyHANDLE)

exit()



