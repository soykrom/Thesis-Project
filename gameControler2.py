# Fidgrove
# Python VJoy interface to control simracing cars from python
# Based on https://github.com/maxofbritton/pyvjoy


#import pyvjoy
import win32event
import mmap
import os
import marshal
import struct
from simple_pid import PID

pid = PID(0.05, 0, 0.05, setpoint=1)

# Setting up vJoy interface
#j = pyvjoy.VJoyDevice(1)
#j.data

# Setting up rFactor2 plugin reader
telemetryH = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, 0 , "WriteEventCarData")
telemetryMMfile = mmap.mmap(-1, length=73, tagname="MyFileMappingCarData", access=mmap.ACCESS_READ)

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
		data = telemetryMMfile.read().decode("utf-8").rstrip('\x00').replace("\n", "").split(',')
		telemetryMMfile.seek(0)

		# Position
		posX = float(data[0])
		posY = float(data[1])
		posZ = float(data[2])

		# Angle
		oriX = float(data[3])

		# Velocity
		velX = float(data[6])
		velY = float(data[7])
		velZ = float(data[8])

		# Acceleration
		accX = float(data[9])
		accY = float(data[10])
		accZ = float(data[11])


		print(-velZ) # Car speed in m/s
		print(-accZ) # Car acceleration m/s^2
		win32event.ResetEvent(telemetryH)

	elif(eventResult == win32event.WAIT_OBJECT_0+1):
		#print("received vehicle scoring data")
		# Read vehicle scoring data from shared memory
		data = vehicleScoringMMfile.read().decode("utf-8").replace("\n", "").split(',')
		vehicleScoringMMfile.seek(0)

		#lapDist = float(data[0])
		#pathLateral = float(data[1])
		#trackEdge = float(data[2])
		
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