# Fidgrove
# Python VJoy interface to control simracing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


import pyvjoy
import win32event
import mmap
import os
import marshal
import struct
from simple_pid import PID

pid = PID(0.05, 0, 0.05, setpoint=1)

# Setting up vJoy interface
j = pyvjoy.VJoyDevice(1)

pid = PID(0.1, 0.01, 0.05, setpoint=0)

# Setting up rFactor2 plugin reader
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=25, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

vehicleScoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteVehicleScoring")
vehicleScoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileVehicleScoring", access=mmap.ACCESS_READ)

#scoringH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventScoringData")
#scoringMMfile = mmap.mmap(-1, length=20, tagname="MyFileMappingScoringData", access=mmap.ACCESS_READ)


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
		
		error = [posX, posZ] # - centerline
		control_x = error[0]
		control_y = error[1]

		# Apply control
		j.set_axis(pyvjoy.HID_USAGE_X, int(16834 + control_x))		
		j.set_axis(pyvjoy.HID_USAGE_Y, int(16834 + control_y))		
		j.update()
  
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



