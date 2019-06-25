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

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#insize = 3*300*640, output size is 9*30*128
		#might want to decrease stride size
		#self.conv1 = torch.nn.Conv2d( 1, 9, kernel_size = (10,5), stride = (10,5), padding = 0)
		# might consider other pool, depending on pixel size of gates
		
		#self.pool = torch.nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

		# 9 * 15 * 64 inputs, 64 outputs
		self.fc1 = torch.nn.Linear(1*300*640,20*640)
		self.fc2 = torch.nn.Linear(20*640,300)
		#64 input features, 2 output features
		self.fc3 = torch.nn.Linear(300,60)
		self.fc4 = torch.nn.Linear(60, 2)

	def forward(self,x):
		
		#batch size is 3*32*32
		# compute activation of first covultion
		#(1,300,644) to (9,30,129)
		#x= F.relu(self.conv1(x))
		
		#(9,30,128) to (9,15,64)
		#x = self.pool(x)

		#reshape data to input to the input layer of neural net
		#(9,15,64) to (1, 9*15*64)
		x = x.view(1,300*640)
		
		#compute activation of first fully connected layer
		#(1,8640) to (1,5*64)
		x= F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=F.relu(self.fc3(x))
		#compute the second fully connected layer, do not activate yet!
		x=self.fc4(x)

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
        io.imshow(image)
        io.show()
        image = rgb2gray(image)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = [img_name,torch.tensor([image]).type('torch.FloatTensor'),torch.tensor(landmarks).type('torch.FloatTensor')]     
        return sample

###############################################################
#                trainNet
##############################################################
def trainNet(net, batch, n_epochs, learning_rate, train_file, test_file):
	
	########################################
	# 1. SET UP
	########################################
	print("======== HYPERPARAMTERS =========")
	print("batch size= ",batch)
	print("epochs= ",n_epochs)
	print("learning rate= ", learning_rate)
	print("=" * 30)



	##############################################
	# 2. get data and obtaining data
	#################################################
	train_data = landmarksDataset(train_file+"/label.csv",train_file)
	test_data = landmarksDataset(test_file+"/label.csv",test_file)
	train = torch.utils.data.DataLoader(train_data,
                                             batch_size=1, shuffle=True,
                                             num_workers=2)
	
	test = torch.utils.data.DataLoader(test_data,
                                             batch_size=1, shuffle=True,
                                             num_workers=1)
	
	
	print("test length "+str(len(test)))
	print("training length "+str(len(train)))


	####################################################
	# 3. create net and optimizer function, and get trainning time
	#####################################################
	optimizer = optim.Adam(net.parameters(), lr= learning_rate)
	start_time = time.time()


	#######################################################
	# 4. training 
	#########################################################



	best_net = None
	best_score = 100
	
	for epoch in range(n_epochs):
		
		# record best net set up
		
		# print out format
		print("="*30)
		print("===========   "+str(epoch)+"   ============ ")

		###################################
		# 4.1 training section
		##################################
		

		start_time = time.time()
		trainning_loss = 0

		trainning_record=[]

		for i, data in enumerate(train, 0):
		
			###########   input labeling

			name,inputs,outputs = data

			[output_x0,output_y0],[output_x1,output_y1]= outputs[0]
			
			inputs,output_x0, output_x1 = Variable(inputs), Variable(output_x0), Variable(output_x1)
			
			##########    prediction labels

			optimizer.zero_grad()

			x0,x1 = net(inputs)[0]
			
			###########      Loss Function
			
			x_0_diff = (output_x0-x0)**2/100
			x_1_diff = (output_x1-x1)**2/100

			x_bound =x0-x0+( 0 if x0>0 else x0**2) +(0 if x1>0 else x1**2)+ (0 if x0<640 else (640-x0)**2) + (0 if x1<640 else (640-x1)**2) + (0 if x0<x1-30 else abs(x0-x1+30)) 
			
			x_diff=x_0_diff+x_1_diff
			
			loss =(x_diff+ x_bound)
			
			loss.backward()
			
			optimizer.step()
			
			trainning_loss+=loss.item()

			trainning_record.append(loss.item()/len(train)/10)

		print("Training Summary")
		print("Epoch trainning loss: {:.2f}".format(trainning_loss) )
		print("Average epoch training time: {:.2f}s".format( (time.time() - start_time)/len(train) ))
		print("Epoch training time: {:.2f}s".format(time.time() - start_time))

		######################################
		#   TESTING section
		########################################

		test_loss = 0

		test_record=[]
		
		test_time = time.time()

		for i, data in enumerate(test, 0):
		
			# get labels 
			name,inputs,outputs = data
			print("outputs")
			print(outputs)
			[output_x0,output_y0],[output_x1,output_y1]= outputs[0]
			

			# get predictions
			prediction = net(inputs)[0]
			print("prediction  ")
			print(prediction)
			x0,x1=prediction
			
			# loss record
			x_loss= abs(output_x0 - x0) +abs(output_x1-x1)
			
			test_loss=test_loss+x_loss
			
			test_record.append(x_loss)

			print("x loss: "+ str(x_loss.item()))
		
		# record net if it is better
		if best_score > test_loss/len(test):
			best_score=test_loss/len(test)
			store_path = train_file+str("gate.pth")
			torch.save(net.state_dict(), store_path)
			best_net = net.state_dict()
			print("*"*30)
			print("recorded")
			print("*"*30)

		# test summary print out

		print("Testing Summary")
		print("Epoch test loss: {:.2f}".format( test_loss.item() ))
		print("Average Epoch test loss: {:.2f}".format( test_loss.item()/len(test) ))
		print("Average Epoch test time: {:.2f}s".format( (time.time() - test_time)/len(test) ))
		print("Best Score: {:.2f}".format( best_score ))

	# overall summary
	print("best score {:.2f}".format(best_score))
	print("Neural Net time: {:.2f}s".format(time.time() - start_time))
	
	plt.plot(range(120),trainning_record, "r",range(120), test_record,"b")
	plt.show()

	return best_net




############################################
#    Test 
#############################################
def test(path):
	model = torch.load(path)
	model.eval()




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

