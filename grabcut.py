import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
drawing = False
mode = True
ix, iy = -1,-1
coordinate = []
file = open("coordinates.txt", 'w')
#mouse callback function
def draw_circle(event, x, y, flags, param):
	global ix,iy,drawing,mode 
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix,iy = x,y
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == True:
				cv2.rectangle(img, (ix,iy),(x,y),(0,255,0),3)
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == True:
			cv2.rectangle(img, (ix,iy), (x,y), (0,255,0), 3)
			coordinate = [ix,iy,x,y]
			print (str(ix)+" "+str(iy)+" "+str(x)+" "+str(y))
			file.write(str(ix)+" "+str(iy)+" "+str(x)+" "+str(y)+" \n")
		


coordinates = []
img = cv2.imread('fish1.jpg')
img_original = img.copy()
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

clone = np.copy(img)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
	cv2.imshow('image',img )
	img = np.copy(clone)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		break
file.close()
with open("coordinates.txt", 'r') as File:
	infoFile = File.readlines()	#reading lines from files
	for line in infoFile: #reading line by line
		words = line.split(' ')
		coordinates.append(words[0])
		coordinates.append(words[1])
		coordinates.append(words[2])
		coordinates.append(words[3])

[a,b,c,d]= coordinates
rect = (int(a),int(b),int(c),int(d))
print (rect)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,100,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imshow("Image", img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresh", thresh)
kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(thresh.copy(), kernel, iterations = 5)
cv2.imshow("Dilation", dilation)
image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
seg_img = cv2.drawContours(img_original, contours, -1, (0,255,0),3)
cv2.imshow("seg_img",seg_img)
plt.imshow(img), plt.show()
cv2.destroyAllWindows()
File.close()
