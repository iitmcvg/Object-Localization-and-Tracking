import os
import time
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import cv2
import glob

text_file =  open("weights1.txt", "w") 
#file containing the path to images and labels
filename = './dataset.txt'
#lists where to store the paths and labels
filenames = []
labels = []

x = tf.placeholder(tf.float32, shape=[None,196608])
y_ = tf.placeholder(tf.float32, shape=[None,4])

def conv2d(x, W, b, strides = 1):
	x1 = tf.nn.conv2d(x,W, strides=[1,strides,strides,1], padding='SAME')
	x2 = tf.nn.bias_add(x1,b)
	return(tf.nn.relu(x2))
def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
def fclayer(x, W, b):
	fc1 = tf.add( tf.matmul(x,W), b)
	return fc1
def init_w(shape):
	return tf.Variable(tf.truncated_normal(shape,mean=0.0, stddev = 0.0001))
def init_b(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

x_image = tf.reshape(x, [-1,256,256,3])

w1 = init_w([5,5,3,3])
b1 = init_b([3])
conv1 = conv2d(x_image, w1, b1)
maxp1 = maxpool2d(conv1)

w2 = init_w([5,5,3,3])
b2 = init_b([3])
conv2 = conv2d(maxp1, w2, b2)
maxp2 = maxpool2d(conv2)

'''w3 = init_w([5,5,3,3])
b3 = init_b([3])
conv3 = conv2d(maxp2, w3,b3)
maxp3 = maxpool2d(conv3)'''

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
y_ = tf.reshape(y_, [-1,4])

loss = tf.reduce_sum(tf.square((fc1) - (y_)))

'''
for i in range(20000):
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={X:batch[0], Y:batch[1], keep_prob:1.0})
		print ("Step:%d\tTraining Accuracy:%.4f" %(i,train_accuracy))
	train_step.run(feed_dict={X:batch[0], Y:batch[1], keep_prob:0.5})
print ("Test Accuracy :%.9f"%accuracy.eval(feed_dict={X:MNIST.test.images, Y:MNIST.test.labels, keep_prob:1.0}))
'''
no_epoch_run = 0

print("\n\n"+str(no_epoch_run) + "\n\n")
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

#creating a queue which contains the list of files to read and the values of labels
filename_queue = tf.train.slice_input_producer([tfilenames, tlabels], num_epochs=10, shuffle=False, capacity = NumFiles)
#reading image files and decoding them
rawIm = tf.read_file(filename_queue[0])
decodedIm = tf.image.decode_jpeg(rawIm)
lbl = []
#extracting the labels queue
label_queue = filename_queue[1]

#Initializing global and local variable initializers
lbl_array = []
init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
#Creating an interactive session to run in python file
label_value = []

while(no_epoch_run <= 1):
	sess = tf.InteractiveSession()
	sess.run(init_op)
	no_epoch=0
	train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)
	with sess.as_default():
		
		#start populating the filename queue
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)
		writer = tf.summary.FileWriter('./graphs', sess.graph)
		begtime = time.time()
		flag = 0
		i = 0
		for i in range(NumFiles):
			nm, image, lb = sess.run([filename_queue[0], decodedIm, label_queue])	
			#plt.imshow(image)
			#plt.show()
			if no_epoch_run<1:
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
			input_image = (sess.run(tf.reshape(image, [-1,196608])))			
			ip_img = input_image
			labels =np.reshape(labels, (-1,4))			
			endtime = time.time()
			print (lbl)			
			lbl = np.reshape(lbl, (-1,4))
			print(sess.run(fc1, feed_dict={x:ip_img, y_:[lbl_array[i]]}))			
			while(sess.run(loss, feed_dict={x:ip_img, y_:[lbl_array[i]]}) > 1.0):
				no_epoch = no_epoch + 1
				train_step.run(feed_dict={x:input_image, y_:[lbl_array[i]]})
				
				if (no_epoch%10)==0:
					if flag == 0:
						print ("\n"+str(no_epoch/10)+"\n")
						flag = flag + 1
					print (str(no_epoch_run)+" "+ str(no_epoch)+" "+str(i) + " Last Feed Neural(Prediction): \n"+ str(sess.run((fc1), feed_dict={x:ip_img, y_:[lbl_array[i]]}))+"\n Label: "+ str(lbl) +"\n Loss Function: "+ str(sess.run(loss, feed_dict={x:ip_img, y_:[lbl_array[i]]})) +str("\n Time Taken: ") + str(endtime-begtime) + str(" s"))
					
					
						
	no_epoch_run = no_epoch_run + 1
	text_file.write(str(sess.run(w1)) +"\n" + str(sess.run(w2))+ "\n" + str(sess.run(w_fc1)))
	coord.request_stop()
	coord.join(threads)
	writer.close()
	

text_file.close()
for imga in glob.glob("./test/*.jpg"):
	img = cv2.imread(imga)
	clone = np.copy(img)
											
	print ("HEllo INDIA")
							
	cv2.waitKey(0)
	img_resized = cv2.resize(img, (256,256), interpolation = cv2.INTER_LINEAR)
							
	image_linear = sess.run(tf.reshape(img_resized, [-1,196608]))
	prediction = sess.run(fc1, feed_dict={x:image_linear})
							#plt.imshow(np.reshape(prediction, [256,256,3]))
							#plt.show()
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	[[a,b,c,d]] = prediction
	rect = (int(a),int(b),int(c),int(d))
	print (rect)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')#cannot understand
	img = img*mask2[:,:,np.newaxis]
	cv2.imshow("Image",img)
	


	plt.imshow(img),plt.colorbar(),plt.show()
	cv2.destroyAllWindows()
	image_restored = np.reshape(image_linear, [256,256,3])
							
							#[[ix,iy,jx,jy]]=prediction
							#print (prediction)
							
	k = cv2.waitKey(0)



	cv2.destroyAllWindows()