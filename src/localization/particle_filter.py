#!/usr/bin/env python
# coding: utf-8

# given camera angle of the task and the change of angle, create a weight distribution in shape of a cone and shift the weight distribution as camera/sub angle changes

# In[6]:


#=============CONFIDENCE============
#if pixelPos == False->2.75
#if pixelPos == True->2.935


import math
import numpy as np
from numpy import array
import copy
import rospy

# In[200]:


# NOTE: The whole class can be sped up, if necessary, with vectorizing in numpy
# TODO: Multiple objects


class ParticleFilter():
    particleMat = []

    def __init__(self, angleAmount=360, pixelPos=False):
        ''' 
        angleAmount is the number of angle we want in particleMat. 
        coordinate: say we use polar with 0 in Cartesian positve x direction. It goes [0,len(angleAmount)), 
        which we will output by mapping this to [0,360), or whatever the sub takes
        '''
        self.pixelPos = pixelPos
        self.particles = angleAmount
        self.particleMat = [1 for i in range(self.particles)]
        self.i_to_angle = 360/angleAmount

    def resetWeights(self):
        rospy.logwarn("RESET WEIGHTS")
        self.particleMat = [1 for i in range(self.particles)]

    def addTask(self):
        pass

    def update(self, newAngle, stdev=50, mode = 'gate'):
        '''
        data will be the input data in angles, most likely one single input and error in angle. 
        angle might need adjustment to fit our coordinates.
        error might need treatment to become standard deviation
        '''
        '''
        increase prob for "angle", bc we see angle.
        '''
        if not self.pixelPos:
            newAngle = -.00000021000348518549*newAngle**3 + .000197285560199453*newAngle**2 + .0854187551642605*newAngle - 40.4349555736021
	    #if newAngle <= 319:
            #	newAngle = -math.degrees(np.arctan((319.5-newAngle)/297.73))
            #else:
            #	newAngle = math.degrees(np.arctan((newAngle-319.5)/297.73))
            print("new angle:", newAngle)
            # Might want to change to adust number of measurements to change weights
            fWeight = 0.12
	    stdev = 10

            for i in range(len(self.particleMat)):
                angle = int(round(i*self.i_to_angle))
                angleDiff = min([math.fabs(angle - newAngle),
                                 math.fabs(newAngle-angle+360)])
                gaussDelta = math.e**(-angleDiff*angleDiff /
                                      (2 * stdev**2))*360/math.sqrt(2*math.pi*stdev**2)
                self.particleMat[i] = fWeight*gaussDelta + \
                    (1-fWeight)*self.particleMat[i]
        else:
            fWeight = 0.3
	    stdev = 40
            for i in range(1, len(self.particleMat)+1):
                diff = math.fabs(newAngle-i)
                gaussDelta = math.e**(-diff*diff / (2 * stdev**2)) * \
                    self.particles/math.sqrt(2*math.pi*stdev**2)
                self.particleMat[i-1] = fWeight*gaussDelta + \
                    (1-fWeight)*self.particleMat[i-1]
        # normalize
        if np.sum(self.particleMat) > self.particles:
            total = self.particles/np.sum(self.particleMat)
            self.particleMat = [i*total for i in self.particleMat]

    def updateSubmarineAngle(self, cameraChange):
        if self.pixelPos:
            return
        cameraChange = cameraChange % 360
        newAng = [0 for i in range(len(self.particleMat))]
        for i in range(len(self.particleMat)):
            angIdx = int(round(cameraChange/self.i_to_angle))
            if i-angIdx >= 0:
                newAng[i-angIdx] = self.particleMat[i]
            else:
                newAng[i+int(round((360-cameraChange)/self.i_to_angle))
                       ] = self.particleMat[i]
        self.particleMat = newAng

    def getPredictedState(self):
        if not self.pixelPos:
	    angles = [x*self.i_to_angle for x in range(0, self.particles)]
            max_x = angles[array(self.particleMat).argmax()]
            stDev = np.std(np.array(self.particleMat), axis=0)
            confidence = self.particleMat[(int)(max_x/self.i_to_angle)]
       	    print("PIXEL_POS: ", max_x)
       	    print("CONFIDENCE: ", confidence)
            if max_x > 180:
                max_x = max_x - 360
            return max_x, confidence
	else:
	    max_x = array(self.particleMat).argmax()
	    confidence = self.particleMat[max_x]
	    max_x = max_x - self.particles/2
	    return max_x,confidence
		

