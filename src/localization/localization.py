import rospy
from particle_filter import ParticleFilter
frome localization.msg import *
from std_msgs import Bool, Float64

class Localization:

	def __init__():
		
		rospy.init_node('localization', anonymous=True)

		#separate for each
		self.gateLeft = ParticleFilter()
		self.gateDiv = ParticleFilter()
		self.gateRight = ParticleFilter()

		#val, confidence
		self.gateLeftPos = [0,0]
		self.gateDivPos = [0,0]
		self.gateRightPos = [0,0]

		
		#buoy up/down left/right
		self.buoyHeave = ParticleFilter()
		self.buoyYaw = ParticleFilter()
		
		self.buoyHeavePos = [0,0]
		self.buoyYawPos = [0,0]


		
		self.gateEnable = False
		self.buoyEnable = False
		
		self.buoyPub = rospy.Publisher("buoyState", buoy, queue_size=2)
		self.gatePub = rospy.Publisher("gateState", gate, queue_size=2)

	
	#buoyPos = [heave, yaw]
	def updateBuoy(self, buoyPos):
		if(buoyPos[0] is not None):
			self.buoyHeave.update(buoyPos[0], pixelPos = True)
		if(buoyPos[1] is not None):
				self.buoyYaw.update(buoyPos[1])
				
		msg = buoy()
		
		msg.yaw, msg.yawConf = self.buoyYaw.getPredictedState()
		msg.heave, msg.heaveConf = self.buoyHeave.getPredictedState()
		buoyPub.publish(msg)

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
		
		gatePub.publish(msg)
		
	
		

	
