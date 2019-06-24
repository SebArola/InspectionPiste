import matplotlib.pyplot as plt
import datetime
import datetime
import time
import cv2
from Video import Video
from dronekit import connect, VehicleMode

###################################
#   Inspection.py : 
# 	Author : Sebastien Arola
#	Description : main class
###################################

class Control:
	vehicle = ""
	
	def __init__(self, port, baud):
		self.connectToUAV(port, baud)
		self.arm_and_takeoff(2)
		
	
	def connectToUAV(self, port, baudrate):
		print("Connecting to vehicle on: %s" % (port,))
		self.vehicle = connect(port, baud=baudrate, wait_ready=True)

		# Get some vehicle attributes (state)
		print("Get some vehicle attribute values:")
		print(" GPS: %s" % self.vehicle.gps_0)
		print( " Battery: %s" % self.vehicle.battery)
		print( " Last Heartbeat: %s" % self.vehicle.last_heartbeat)
		print( " Is Armable?: %s" % self.vehicle.is_armable)
		print( " System status: %s" % self.vehicle.system_status.state)
		print( " Mode: %s" % self.vehicle.mode.name)		    # settable
		
	
	def arm_and_takeoff(self, aTargetAltitude):
		"""
		Arms vehicle and fly to aTargetAltitude.
		"""

		print("Basic pre-arm checks")
		# Don't try to arm until autopilot is ready
	
		while not self.vehicle.is_armable:
		    print(" Waiting for vehicle to initialise...")
		    time.sleep(1)
		
		print("Arming motors")
		# Copter should arm in GUIDED mode
		self.vehicle.mode    = VehicleMode("GUIDED")
		self.vehicle.armed   = True

		# Confirm vehicle armed before attempting to take off
		while not self.vehicle.armed:
		    print(" Waiting for arming...")
		    time.sleep(1)

		#print "Taking off!"
		#vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

		# Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
		#  after Vehicle.simple_takeoff will execute immediately).
		#while True:
		#    print " Altitude: ", vehicle.location.global_relative_frame.alt
		#    #Break and return from function just below target altitude.
		#    if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95:
		#        print "Reached target altitude"
		#        break
		#    time.sleep(1)
	
class Detection:

	def __init__(self, tAttente, weight, seuil, video_url):
		self.temps_attente = tAttente
		self.video = Video(weight, seuil, video_url)

	def startPredict(self):
		check = time.time()
		while(True):
			if time.time()-check >= self.temps_attente:
				tPreTraitement =time.time()
				print("\n++++++++\nTemps d'attente : "+str(int(time.time()-check)))
				result = self.video.predict()
				print("Duree traitement : " + str(time.time()-tPreTraitement))
				print("Debris : " + str(result))
				check = time.time()


def getConfigInfo(file, nb_param):
	config = []
	header = True
	fichier = open(file, "r")
	for ligne in fichier:
		if header == False :
			if ligne == "}\n":
				break;
			config.append(ligne.split("->")[1].split(";")[0])
		if ligne == "{\n" :
			header = False
			
	if len(config)<nb_param :
		print("Error : expected "+str(nb_param)+" parameters in config.txt, but found "+str(len(config))+".")
		exit(0)
	else :
		weight   = config[0]
		cam      = config[1]
		seuil    = float(config[2])
		tAttente = float(config[3])
		port	 = config[4]
		baud	 = int(config[5])
		return (weight,seuil,cam, tAttente, port, baud)
			
if __name__ == "__main__":
	(weight, seuil, video_url, tAttente, port, baud) = getConfigInfo("config.txt", 6)
	detection = Detection(tAttente,weight, seuil, video_url)
	control = Control(port, baud)
	#detection.startPredict()
	
	
	
	
	
