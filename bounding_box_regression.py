import os
import time
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import cv2
import glob
totaltime=0
text_file =  open("weights.txt", "w") 
#file containing the path to images and labels
filename = './dataset.txt'
#lists where to store the paths and labels
filenames = []
labels = []

x = tf.placeholder(tf.float32, shape=[None,196608])
y_ = tf.placeholder(tf.float32, shape=[None,4])

def conv2d(x, W, b, strides = 1):
	x = tf.nn.conv2d(x,W, strides=[1,strides,strides,1], padding='SAME')
	x = tf.nn.bias_add(x,b)
	return(tf.nn.relu(x))
def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
def fclayer(x, W, b):
	fc1 = tf.add( tf.matmul(x,W), b)
	return fc1
def init_w(shape):
	return tf.Variable(tf.truncated_normal(shape,mean=0.0, stddev = 0.1))
def init_b(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

x_image = tf.reshape(x, [-1,256,256,3])

w1 = init_w([5,5,3,3])
y_ = tf.reshape(y_, [-1,4])
b1 = init_b([3])
conv1 = conv2d(x_image, w1, b1)
maxp1 = maxpool2d(conv1)

w2 = init_w([5,5,3,3])
b2 = init_b([3])
conv2 = conv2d(maxp1, w2, b2)
maxp2 = maxpool2d(conv2)
'''
w3 = init_w([5,5,3,3])
b3 = init_b([3])
conv3 = conv2d(maxp2, w3,b3)
maxp3 = maxpool2d(conv3)
'''
conv3_flatten = tf.reshape(maxp2,[-1,12288])
w_fc1 = init_w([12288,4])
b_fc1 = init_b([4])
fc1 = fclayer(conv3_flatten,w_fc1, b_fc1)
'''
w_fc2 = init_w([500, 100])
b_fc2 = init_b([100])
fc2 = fclayer(fc1,w_fc2, b_fc2)
w_fc3 = init_w([100,4])
b_fc3 = init_b([4])
fc3 = fclayer(fc2,w_fc3,b_fc3)
'''
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc1))
loss = tf.reduce_sum(tf.square((fc1) - (y_)))

'''
for i in range(20000):
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={X:batch[0], Y:batch[1], keep_prob:1.0})
		print ("Step:%d\tTraining Accuracy:%.4f" %(i,train_accuracy))
	train_step.run(feed_dict={X:batch[0], Y:batch[1], keep_prob:0.5})
print ("Test Accuracy :%.9f"%accuracy.eval(feed_dict={X:MNIST.test.images, Y:MNIST.test.labels, keep_prob:1.0}))
'''
#reading file and extracting path and labels
with open(filename, 'r') as File:
	infoFile = File.readlines()	#reading lines from files
	for line in infoFile: #reading line by line
		words = line.split(' ')
		filenames.append(words[0])
		labels.append(words[1])
		labels.append(words[2])
		labels.append(words[3])
		labels.append(words[4])
		#print (labels)
input_image = []
sess = tf.InteractiveSession()
NumFiles = len(filenames) 
#Converting filnames into tensor
tfilenames = ops.convert_to_tensor(filenames, dtype = dtypes.string)
tlabels = ops.convert_to_tensor(labels, dtype=dtypes.string)
#print (tlabels)
#creating a queue which contains the list of files to read and the values of labels
filename_queue = tf.train.slice_input_producer([tfilenames, tlabels], num_epochs=10, shuffle=False, capacity = NumFiles)
#reading image files and decoding them
rawIm = tf.read_file(filename_queue[0])
decodedIm = tf.image.decode_jpeg(rawIm)
lbl = []
#extracting the labels queue
label_queue = filename_queue[1]
#print (label_queue)
#Initializing global and local variable initializers
init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
#Creating an interactive session to run in python file
label_value = []
sess = tf.InteractiveSession()
sess.run(init_op)
Tolerance = 0
no_epoch=0
loss_to_be_minimized = 0
label_counter = 0
train_step = tf.train.GradientDescentOptimizer(1e-8).minimize(loss)
Train_Checker = []
with sess.as_default():
	
	#start populating the filename queue
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	flag = 0 #epoch
	lbl_array = []
	img_array = []
	
	while(True):
		flag = flag + 1
		begtime = time.time()
		i = 0
		Train_Checker.append(loss_to_be_minimized)	#Previous loss function
		loss_to_be_minimized = 0
		for i in range(NumFiles):
			
			if flag<=1:
				nm, image, lb = sess.run([filename_queue[0], decodedIm, label_queue])					
				if i == 0:
					l1 = labels[(4*i)+0]
					l2 = labels[(4*i)+1]
					l3 = labels[(4*i)+2]					
					l4 = labels[((4*i)+3)]					
					lbl = [l1,l2,l3,l4]
					lbl_array.append(lbl)					
				else:
					lbl = labels[i]
					lbl_array.append(lbl)
				input_image = (sess.run(tf.reshape(image, [196608])))
				img_array.append(input_image)
				ip_img = input_image
				labels =np.reshape(labels, (-1,4))
			
			#plt.imshow(image)
			#plt.title(sess.run(labels))
			#plt.show()
			#lbl = np.reshape(lbl, (-1,4))
		
			no_of_times_run = 0
			#no_epoch = no_epoch + 1
			while(True):
				train_step.run(feed_dict={x:[img_array[i]], y_:[lbl_array[i]]})
				no_of_times_run = no_of_times_run + 1
				if no_of_times_run>3:
					break
			loss_to_be_minimized = loss_to_be_minimized +  sess.run(loss, feed_dict={x:[img_array[i]], y_:[lbl_array[i]]})
			#print(input_image.shape+ " " + labels.shape
			#if (no_epoch%10)==0:
			#	if flag == 0:
			#		print ("\n"+str(no_epoch/10)+"\n")
			#		flag = flag + 1
			#print (str(i) + " Last Feed Neural(Prediction): \n"+ str(sess.run((fc1), feed_dict={x:ip_img, y_:lbl}))+"\n Label: "+ str(lbl) +"\n Loss Function: "+ str(sess.run(loss, feed_dict={x:ip_img, y_:lbl})) +str("\n Time Taken: ") + str(endtime-begtime) + str(" s"))
		endtime = time.time()
		totaltime = totaltime + (endtime-begtime)

		print ("Epoch: "+ str(flag)+ "\t"+ "Total Error: "+str(loss_to_be_minimized)+ "\t"+ "Tolerance: "+ str(Tolerance)+ "\t" + "Time Taken: "+ str(endtime- begtime))
					#text_file.write(str(loss)+" "+str(no_epoch)+" "+ "\n")
	
		text_file.write(str(flag) + " "+ str(loss_to_be_minimized))
		#plt.axis([0, 100, 0, 1e5])

		plt.ion()
		y = loss_to_be_minimized
		plt.xlabel("Epochs")
		plt.ylabel("Total_loss(L2 Loss)")
		plt.title("Loss Vs Epochs")
		plt.scatter(flag, y)
		plt.pause(0.05)
		Train_Checker.append(loss_to_be_minimized) #Updated Loss function
		if (loss_to_be_minimized < 1000):
			break
		if (Train_Checker[0] <= Train_Checker[1] and flag > 1):
			Tolerance = Tolerance + 1
			if (Tolerance > 30):	
				break
		del Train_Checker[:]
	coord.request_stop()
	coord.join(threads)
	writer.close()
	
'''		image_conv = sess.run(maxp1, feed_dict={x:image})
		op_image = tf.to_float(tf.reshape(image_conv, [128,128,3]))
		plt.imshow(sess.run(op_image))
		plt.title("Conv-Layer-1")
		plt.show()
		image_conv2 = sess.run(maxp2, feed_dict={x:ip_img})
		op_image = tf.to_float(tf.reshape(image_conv2,[64,64,3]))
		op_image = tf.image.resize_images(op_image, [128,128])
		plt.imshow(sess.run(op_image))
		plt.title("Conv-Layer-2")
		plt.show()
'''
print("TotalTime Taken: " + str(totaltime))
for imga in glob.glob("./test/*.jpg"):
	img = cv2.imread(imga)
	clone = np.copy(img)
	prediction=np.zeros(1)
	#cv2.namedWindow('image')
	#cv2.setMouseCallback('image', draw_circle)

#	while(1):
	print ("HEllo INDIA")
	cv2.imshow('image',img )
	cv2.waitKey(0)
	img_resized = cv2.resize(img, (256,256), interpolation = cv2.INTER_LINEAR)
	image_linear = np.reshape(img_resized, [-1,196608])
	prediction= (sess.run(fc1, feed_dict={x:image_linear}))
	image_restored = np.reshape(image_linear, [256,256,3])

	[[ix,iy,jx,jy]]=prediction
	print (prediction)
	cv2.rectangle(img_resized, (np.floor(ix), np.floor(iy)), (np.floor(jx), np.floor(jy)), (0,0,255), 3)
	cv2.imshow("Test_Image", img_resized)
	k = cv2.waitKey(0)

	cv2.destroyAllWindows()
text_file.close()