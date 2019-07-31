#!/usr/bin/env python

import rospy
from particle_filter import ParticleFilter
from vision.msg import *
from std_msgs.msg import Bool, Float64

class Localize:

	def __init__(self):
		
		#separate for each
		self.gateLeft = ParticleFilter()
		self.gateDiv = ParticleFilter()
		self.gateRight = ParticleFilter()

		#val, confidence
		self.gateLeftPos = [0,0]
		self.gateDivPos = [0,0]
		self.gateRightPos = [0,0]

		
		#buoy up/down. First/second in terms of area
		self.firstBuoyHeave = ParticleFilter(pixelPos = True)
		self.firstBuoyYaw = ParticleFilter()
		
		self.firstBuoyHeavePos = [0,0]
		self.firstBuoyYawPos = [0,0]
		

		self.secondBuoyHeave = ParticleFilter(pixelPos = True)
                self.secondBuoyYaw = ParticleFilter()

                self.secondBuoyHeavePos = [0,0]
                self.secondBuoyYawPos = [0,0]


		self.gateEnable = False
		self.buoyEnable = False
		
		self.buoyPub = rospy.Publisher("buoyState", buoy, queue_size=1)
		self.gatePub = rospy.Publisher("gateState", gate, queue_size=1)
		
		self.gateResetSub = rospy.Subscriber("gateReset", Bool, self.gateResetCB)
		self.buoyResetSub = rospy.Subscriber("buoyReset", Bool, self.buoyResetCB)
	
	#buoyPos = [[yaw1,heave1], [yaw2,heave2]]
	def updateBuoy(self, buoyPos):
		if len(buoyPos) == 0:
			return
		self.firstBuoyYaw.update(buoyPos[0][0])
		self.firstBuoyHeave.update(buoyPos[0][1])

		if len(buoyPos) == 2:
			self.secondBuoyYaw.update(buoyPos[1][0])
			self.secondBuoyYaw.update(buoyPos[1][1])				

		msg = buoy()
		
		msg.firstYaw, msg.firstYawConf = self.firstBuoyYaw.getPredictedState()
		msg.firstHeave, msg.firstHeaveConf = self.firstBuoyHeave.getPredictedState()

		msg.secondYaw, msg.secondYawConf = self.secondBuoyYaw.getPredictedState()
                msg.secondHeave, msg.secondHeaveConf = self.secondBuoyHeave.getPredictedState()

		self.buoyPub.publish(msg)

	#gatePos = [left, div, right]
	def updateGate(self, gatePos):
		if(gatePos[0] is not None):
			self.gateLeft.update(gatePos[0])
		if(gatePos[1] is not None):
				self.gateDiv.update(gatePos[1])
		if(gatePos[2] is not None):
				self.gateRight.update(gatePos[2])
		
		
		msg = gate()
		msg.left, msg.leftConf = self.gateLeft.getPredictedState()
		msg.right, msg.rightConf = self.gateRight.getPredictedState()
		msg.div, msg.divConf = self.gateDiv.getPredictedState()
		print(msg.left)
		self.gatePub.publish(msg)
		
	def gateResetCB(self, data):
		self.gateLeft.resetWeights()
		self.gateDiv.resetWeights()
		self.gateRight.resetWeights()
	
	def buoyResetCB(self, data):
		self.firstBuoyHeave.resetWeights()	
		self.firstBuoyYaw.resetWeights()	
		self.secondBuoyHeave.resetWeights()	
		self.secondBuoyYaw.resetWeights()	
