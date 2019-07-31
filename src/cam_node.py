#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def main():
	rospy.init_node('cam_publish')

	pub = rospy.Publisher("/usb_cam_0/image_raw", Image, queue_size=1)

	cam = cv2.VideoCapture(0)
	cam.set(3,1920)
	cam.set(4,1080)
	bridge=CvBridge()

	rate = rospy.Rate(10)

	while not rospy.is_shutdown():
		s,im = cam.read()
		#im = cv2.resize(im,(640,480))
		cv2.imshow("a", im)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		pub.publish(bridge.cv2_to_imgmsg(im, "bgr8"))
		print(im.shape)

if __name__=="__main__":
	main()	
