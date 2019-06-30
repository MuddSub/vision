import matplotlib.pyplot as plt
import sys
import cv2
import os
name = sys.argv[1]

def onclick(event):
	x, y = event.xdata, event.ydata
	print("registered "+str(x)+" "+str(y))
	word = str(ix)+" "+str(iy)
	os.write(write,str.encode(word))

if name == "d":
	#path0 = "/home/dhyang/Desktop/Vision/Vision/gate2/"
	path0 = "/home/dhyang/Desktop/Vision/Vision/Neural_Net/"
else:
	path0 = "/Users/rongk/Downloads/visionCode/Vision/Neural_Net/"
path_train = path0 + "Train_binarized0"
path_test = path0 +"Test_binarized0"

path = path_test

directory = os.fsencode(path)
#write = os.open("log.txt",os.O_APPEND)

for file in os.listdir(directory):
	filename = os.fsdecode(file)
	img = cv2.imread(path+"/"+filename)
	print(str(len(img))+" "+str(len(img[0])))
	plt.imshow(img)
	plt.title(filename)
	plt.show()
	
	#fig,ax=plt.subplots()
	#cid = fig.canvas.mpl_connect('button_press_event', onclick)
	#fig.canvas.mpl_disconnect(cid)
	#cid = fig.canvas.mpl_connect('button_press_event', onclick)
	#fig.canvas.mpl_disconnect(cid)	
	#os.write(write,str.encode("\n"))
	plt.close()

#os.close(write)
