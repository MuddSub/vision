#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('vision')
import sys
sys.path.append("./localization")
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gate import Gate
from localize import Localize

class Cameras:
	
	def __init__(self):
		rospy.init_node('Cameras', anonymous=True)

		self.cameraMap = {'down': None, 'front': None}
		self.cvImage0 = None
		self.cvImage1 = None
		self.bridge = CvBridge()
		self.image0Sub = rospy.Subscriber("usb_cam_0/image_raw", Image, self.callback0)
		self.image1Sub = rospy.Subscriber("usb_cam_1/image_raw", Image, self.callback1)

		self.imagePub = rospy.Publisher("image_out", Image,queue_size=1)

		self.loadCameras()
		
		self.cam0Ready = False
		self.cam1Ready = False

	def loadCameras(self):
		#load camera names from /dev/video[n]
		try:
			f0 = open("/sys/class/video4linux/video0/name")
		except Exception as e:
			print("Whoops, couldn't find camera 0!")
			return
		try:
			f1 = open("/sys/class/video4linux/video1/name")
		except Exception as e:
			print("Whoops, couldn't find camera 1!")
			return
			
		if(f0.read(1) == 'U'):
			#0 corresponds to /dev/video0, 1 to /dev/video1
			self.cameraMap['down'] = 0
			self.cameraMap['front'] = 1
		else:
			self.cameraMap['front'] = 0
			self.cameraMap['down'] = 1	
		
	
	def getFrame(self, camera):
		try:
			dev	= self.cameraMap[camera]
		except Exception as e:
			print("Error in get frame:", e)
			
		if(dev == 0):
			return self.cvImage0
		else:
			return self.cvImage1

	def getFrontFrame(self):
		return self.getFrame('front')
		
	def getDownFrame(self):
		return self.getFrame('down')
		
	def callback0(self, data):
		self.cam0Ready = True

		try:
			cvImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		self.cvImage0 = cvImage

	def callback1(self, data):
		self.cam1Ready = True

		try:
			cvImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		self.cvImage1 = cvImage


def main(args):
	rospy.init_node('Cameras', anonymous=True)

	cams = Cameras()
	gate = Gate()
	loc = Localize()
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown():
			if(cams.cam0Ready and cams.cam1Ready):
				img = cams.getFrontFrame()
				if(img is None):
					rospy.logwarn("none image recieved")
					continue
				bars = gate.findBars(img)
				loc.updateGate([bars[1], bars[2], bars[3]])
				try:
					cams.imagePub.publish(cams.bridge.cv2_to_imgmsg(img, "bgr8"))
				except Exception as e:
					print("HERE")
					print(e)
				
				
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__==("__main__"):
	main(sys.argv)
