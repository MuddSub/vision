import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 
import sys
import pandas as pd
from skimage import io,transform
from torch.utils.data import Dataset, DataLoader
import random
from skimage.color import rgb2gray
# modified from algorithmia.com

###############################################
#                 model and neural net
###############################################

def 
	self.conv1 = torch.nn.Conv2d( 1, 9, kernel_size = (10,5), stride = (10,5), padding = 0)
	# might consider other pool, depending on pixel size of gates
		
	self.pool = torch.nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

		# 9 * 15 * 64 inputs, 64 outputs
		self.fc1 = torch.nn.Linear(9*15*64,15*64)
		self.fc15 = torch.nn.Linear(15*64,3*64)
		#64 input features, 2 output features
		self.fc152 = torch.nn.Linear(3*64, 64)
		self.fc2 = torch.nn.Linear(64, 2)

	def forward(self,x):
		
		#batch size is 3*32*32
		# compute activation of first covultion
		#(1,300,644) to (9,30,129)
		x= F.relu(self.conv1(x))
		
		#(9,30,128) to (9,15,64)
		x = self.pool(x)

		#reshape data to input to the input layer of neural net
		#(9,15,64) to (1, 9*15*64)
		x = x.view(1,9*15*64)
		
		#compute activation of first fully connected layer
		#(1,8640) to (1,5*64)
		x= F.relu(self.fc1(x))
		x=F.relu(self.fc15(x))
		x=F.relu(self.fc152(x))
		#compute the second fully connected layer, do not activate yet!
		x=self.fc2(x)

		return x


	def outputSize(in_size,kernel_size, stride,padding):
		return int((in_size - kernel_size + 2*padding)/stride)+1

##############################################################
#				data set and image processing
##############################################################
class landmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = transform.resize(image,(300,640))
        image = rgb2gray(image)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = [img_name,torch.tensor([image]).type('torch.FloatTensor'),torch.tensor(landmarks).type('torch.FloatTensor')]     
        return sample

###############################################
#              driver
###############################################

def driver():
	name = sys.argv[1]

	if name == "d":
		#path0 = "/home/dhyang/Desktop/Vision/Vision/gate2/"
		path0 = "/home/dhyang/Desktop/Vision/Vision/Neural_Net/"
	else:
		path0 = "/Users/rongk/Downloads/visionCode/Vision/Neural_Net/"
	path_train = path0 + "Train/"
	path_test = path0 +"Test/"
	trainNet(Net(), 1, 120, 0.01, path_train,path_test)
	#32

if __name__ == "__main__":
	driver()

