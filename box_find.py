import cv2
import glob
import numpy as np
drawing = False
mode = True
ix, iy = -1,-1
file = open("dataset.txt", 'w')
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
			print (str(imga)+ " "+str(ix)+" "+str(iy)+" "+str(x)+" "+str(y))
			file.write(str(imga)+ " "+str(ix)+" "+str(iy)+" "+str(x)+" "+str(y)+" \n")



for imga in glob.glob("*.jpg"):
	img = cv2.imread(imga)
	clone = np.copy(img)

	cv2.namedWindow('image')
	cv2.setMouseCallback('image', draw_circle)

	while(1):
		cv2.imshow('image',img )
		img = np.copy(clone)
		k = cv2.waitKey(1) & 0xFF
		if k == ord('m'):
			break

	cv2.destroyAllWindows()
file.close()