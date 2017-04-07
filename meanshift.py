import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
cap=cv2.VideoCapture('DETECT INTERN/Find the terrorist.mp4')

ret,frame=cap.read()
refPt = []
cropping = False
image = frame
def click_and_crop(event, x, y, flags, param):
	global refPt, cropping
 

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		cropping = False
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)


clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 

while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 # if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break
 

if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)

x1,y1=refPt[0]
x2,y2=refPt[1]
c=x1
r=y1
w=x2-x1
h=y2-y1

roi1=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
blue = cv2.calcHist([roi1],[0],None,[180],[0,180])
green = cv2.calcHist([roi1],[1],None,[256],[0,256])
red = cv2.calcHist([roi1],[2],None,[256],[0,256])
plt.plot(blue)
plt.plot(green)
plt.plot(red)
plt.show()

bluey, bluex, _ = plt.hist(blue)
max_blue=bluey.max()
min_blue=bluey.min()
mean_blue=bluey.mean()
std_blue=bluey.std()
greeny, greenx, _ = plt.hist(green)
max_green=greeny.max()
min_green=greeny.min()
mean_green=greeny.mean()
std_green=greeny.std()
redy, redx, _ = plt.hist(red)
max_red=redy.max()
min_red=redy.min()
mean_red=redy.mean()
std_red=redy.std()
a=1
higher=np.array((mean_blue+a*std_blue,mean_green+a*std_green,mean_red+a*std_red))
lower=np.array((mean_blue-a*std_blue,mean_green-a*std_green, mean_red-a*std_red))
print(lower)
print(higher)
track_window=(c,r,w,h)

roi=frame[r:r+h, c:c+w]
hsv_roi=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(hsv_roi,lower,higher)
cv2.imshow('mask',mask)
roi_hist=cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
	ret ,frame = cap.read()
	if ret == True:
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
		# apply meanshift to get the new location
		ret, track_window = cv2.meanShift(dst, track_window, term_crit)
		# Draw it on image
		x,y,w,h = track_window
		img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
		cv2.imshow('img2',img2)
		k = cv2.waitKey(60) & 0xff
		if k == 27:
			 break
		else:
			 cv2.imwrite(chr(k)+".jpg",img2)
	else:
		 break
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()