import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time 
import sys
# modified from algorithmia.com
###############################################
#              driver
###############################################

def driver():
	name = sys.argv[1]

	if name == "d":
		#path0 = "/home/dhyang/Desktop/Vision/Vision/gate2/"
		path0 = "/home/dhyang/Desktop/Vision/Vision/gate5/gate_training_"
	else:
		path0 = "/Users/rongk/Downloads/visionCode/Vision/gate5/gate_training_"
	path_train = path0 +
	path_test = path0 +
	trainNet(Net(), batch_size=32, n_epochs=5, learning_rate=0.001, path_train,path_test)


if __name__ == "__main__":
	driver()

###############################################
#             data processing
###############################################

def data(batch, train_file,test_file):
	train_set = datasets.ImageFolder(root=train_file, train = True)
	train_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch, shuffle=True,
                                             num_workers=2)
	
	test_set = datasets.ImageFolder(root=test_file, train = False)
	test_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch, shuffle=True,
                                             num_workers=2)
	return train_loader,test_loader

###############################################
#                 model
###############################################

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		# input channels = 3, output channels = 18
		self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
		# might consider other pool, depending on pixel size of gates
		self.pool = torch.nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

		# 18 * 16 * 16 inputs, 64 outputs
		self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)

		#64 input features, 2 output features
		self.fc2 = torch.nn.Linear(64, 10)

	def forward(self,x):
		
		#batch size is 3*32*32

		# compute activation of first covultion
		#(3,32,32) to (18,32,32)
		x= F.relu(self.conv1(x))
		
		#(18,32,32) to (18,16,16)
		x = self.pool(x)

		#reshape data to input to the input layer of neural net
		#(18,16,16) to (1, 18*16*16=4608)
		x = x.view(-1,18*16*16)

		#compute activation of first fully connected layer
		#(1,4608) to (1,64)
		x= F.relu(self.fc1(x))

		#compute the second fully connected layer, do not activate yet!
		x=self.fc2(2)

		return x


	def outputSize(in_size,kernel_size, stride,padding):
		return int((in_size - kernel_size + 2*padding)/stride)+1
	
def trainNet(batch, n_epochs, learning_rate, train_file, test_file):
	
	# 1. print out hyperparameters
	print("======== HYPERPARAMTERS =========")
	print("batch size= ",batch)
	print("epochs= ",epochs)
	print("learning rate= ", learning_rate)
	print("=" * 30)

	# 2. get data
	train, test = data(batch, train_file, test_file)

	n_batch = len(train)

	# 3. create net and optimizer function, and get trainning time
	net = Net()
	optimizer = optim.Adam(net.parameters(), lr= learning_rate)

	train_start_time = time.time()

	# 4. training 
	for epoch in range(n_epochs):

		# 4.1 training data parameters
		running_loss = 0.0
		print_every = n_batches // 10
		start_time = time.time()
		total_train_loss = 0
	
		for i, data in enumerate(train, 0):
			
			#get input, x0 and x1 are positions of two bars of the gate
			inputs,x0, x1 = data
			inputs,x0, x1 = Variable(inputs), Variable(x0), Variable(x1)

			#set parameter gradient to zero
			optimizer.zero_grad()

			#forward pass, backward pass, optimizer
			output0, output1 = net(inputs)
			loss_size = (output0 - x0)**2/25 + (output1-x1)**2/25
			loss_size.backward()
			optimizer.step()

			#print stat
			running_loss+=loss_size.data[0]
			total_train_loss += loss_size.data[0]
			
			#print every 10th batch of an epoch
			if (i+1)%(print_every+1)==0:
				print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))	
				running_loss = 0
				start_time = time.time()
			
			test_loss = 0
			for inputs, labels in test_loader:
				inputs,labels = Variable(inputs),Variable(labels)
				output0,output1 = labels
				x0,x1 = net(inputs)
				test_loss = (output0 - x0)**2/25 + (output1-x1)**2/25
			print("Validation loss = {:.2f}".format(test_loss / len(test_loader)))
        
	print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
if __name__ == "__main__":
    train(200)
