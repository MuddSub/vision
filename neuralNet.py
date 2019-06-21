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
#                 model
###############################################

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#insize = 3*300*640, output size is 9*30*128
		#might want to decrease stride size
		self.conv1 = torch.nn.Conv2d( 1, 9, kernel_size = (10,5), stride = (10,5), padding = 0)
		# might consider other pool, depending on pixel size of gates
		
		self.pool = torch.nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

		# 9 * 15 * 64 inputs, 64 outputs
		self.fc1 = torch.nn.Linear(9*15*64,9*64)
		self.fc15 = torch.nn.Linear(9*64,64)
		#64 input features, 2 output features
		self.fc2 = torch.nn.Linear(64, 4)

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
		#compute the second fully connected layer, do not activate yet!
		x=self.fc2(x)

		return x


	def outputSize(in_size,kernel_size, stride,padding):
		return int((in_size - kernel_size + 2*padding)/stride)+1

##############################################################
#				data set
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
        print(img_name)
        image = io.imread(img_name)
        image = transform.resize(image,(300,640))
        image = rgb2gray(image)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = [torch.tensor([image]).type('torch.FloatTensor'),torch.tensor(landmarks).type('torch.FloatTensor')]
        return sample

###############################################################
#                trainNet
##############################################################
def trainNet(net, batch, n_epochs, learning_rate, train_file, test_file):
	

	# 1. print out hyperparameters
	print("======== HYPERPARAMTERS =========")
	print("batch size= ",batch)
	print("epochs= ",n_epochs)
	print("learning rate= ", learning_rate)
	print("=" * 30)

	# 2. get data
	train_data = landmarksDataset(train_file+"/label.csv",train_file)
	test_data = landmarksDataset(test_file+"/label.csv",test_file)
	train = torch.utils.data.DataLoader(train_data,
                                             batch_size=1, shuffle=True,
                                             num_workers=2)
	
	test = torch.utils.data.DataLoader(test_data,
                                             batch_size=1, shuffle=True,
                                             num_workers=1)
	
	
	print("test length "+str(len(test)))

	# 3. create net and optimizer function, and get trainning time

	optimizer = optim.Adam(net.parameters(), lr= learning_rate)

	train_start_time = time.time()
	print("training length "+str(len(train)))
	# 4. training 
	for epoch in range(n_epochs):
		print("##################################")
		print("########   "+str(epoch)+"   ######")

		# 4.1 training data parameters
		running_loss = 0.0
		start_time = time.time()
		trainning_loss = 0
		for i, data in enumerate(train, 0):
			#get input, x0 and x1 are positions of two bars of the gate
			inputs,outputs = data
			[output_x0,output_y0],[output_x1,output_y1]= outputs[0]
			inputs,output_x0, output_x1,output_y1,output_y2 = Variable(inputs), Variable(output_x0), Variable(output_x1),Variable(output_y0),Variable(output_y1)

			#set parameter gradient to zero
			optimizer.zero_grad()

			#forward pass, backward pass, optimizer
			x0,y0,x1,y1 = net(inputs)[0]
			#TO DO CHANGE OVER AREA OVERLAP LOSS
			'''
			x_0_diff = (0 if abs(output_x0-x0) < 25 else ( output_x0 - x0)**2)
			x_1_diff = (0 if abs(output_x1-x1)<25 else (output_x1-x1)**2)
			x_bound =( (0 if x0<640 else (640-x0)**2) + (0 if x1<640 else (640-x1)**2) + (0 if x0<x1-30 else (x0-x1+30)**2) )
			y_diff = (abs(output_y0 - y0) +abs(output_y1-y1))
			y_bound = ((0 if y0<300 else abs(300-y0)) + (0 if y1<300 else abs(300-y1)) + (0 if y0<y1-30 else (y0-y1+30)**2))
			'''
			print("output")
			print(output_x0)
			print(x0)
			x_0_diff = (output_x0-x0)**2/100
			x_1_diff = (output_x1-x1)**2/100
			y_0_diff = (output_y0-y0)**2/100
			y_1_diff = (output_y1-y1)**2/100
			if x0<640 or x1<640 or x0<x1-30 or :	
				x0.requires_grad=False
				x1.requires_grad=False
				y0.requires_grad=False
				y1.requires_grad=False
				x_bound =(0 if x0>0 else x0**2) +(0 if x1>0 else x1**2)+ (0 if x0<640 else (640-x0)**2) + (0 if x1<640 else (640-x1)**2) + (0 if x0<x1-30 else abs(x0-x1+30)) 
				y_bound = (0 if y0>0 else y0**2) + (0 if y1>0 else y1**2)+ (0 if y0<300 else abs(300-y0))+ (0 if y1<300 else abs(300-y1)) + (0 if y0<y1-30 else abs(y0-y1+30))
			y_diff = y_0_diff+y_1_diff
			print("info")
			print("x 0 diff"+ str(x_0_diff))
			print("x1 diff "+str(x_1_diff))
			print("x_bound"+str(x_bound))
			print("y diff"+str(y_diff))
			print("y bound"+str(y_bound))
			x_diff=x_0_diff+x_1_diff
			print("sum")
			print(x_diff.item()+x_bound.item()+y_diff.item()+y_bound.item())
			loss =(x_diff+ x_bound+y_diff+y_bound)
			print("loss  "+str(loss.item()))
			print(trainning_loss)
			loss.backward()
			optimizer.step()
			trainning_loss+=loss.item()
			'''
			#print stat
			running_loss+=loss_size.item()
			total_train_loss += loss_size.item()
			
			#print every 10th batch of an epoch
			if (i+1)%(print_every+1)==0:
				print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))	
				running_loss = 0
				start_time = time.time()
			'''
		print("trainning loss:  "+str(trainning_loss))
		total_loss = 0
		test_loss = 0
		total_x=0
		total_y=0
		for i, data in enumerate(test, 0):
		
			#get input, x0 and x1 are positions of two bars of the gate
			inputs,outputs = data
			print("outputs")
			print(outputs)
			
			[output_x0,output_y0],[output_x1,output_y1]= outputs[0]
			
			#forward pass, backward pass, optimizer
			prediction = net(inputs)[0]
			print("prediction  ")
			print(prediction)
			x0,y0,x1,y1 = prediction
			x_loss= abs(output_x0 - x0) +abs(output_x1-x1)
			y_loss = abs(output_y0 - y0) +abs(output_y1-y1)
			test_loss=test_loss+x_loss+y_loss
			total_x+=x_loss
			total_y+=y_loss
			total_loss+=test_loss	
		#print("Validation loss = {:.2f}".format(test_loss / len(test)))
			print("test loss    "+ str(test_loss.item()))
			print("x,y loss    "+ str(x_loss.item())+ "   "+str(y_loss.item()))
		print("#############  total loss:   " + str(total_loss.item()))
		print("############# x loss    "+ str(total_x.item()))
	print("Training finished, took {:.2f}s".format(time.time() - train_start_time))
	torch.save(net.state_dict(), train_file+str("net.txt"))
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
	trainNet(Net(), 1, 120, 0.001, path_train,path_test)
	#32

if __name__ == "__main__":
	driver()

