'''
When you draw rectangle with the mouse, after you release the left mouse button the rectangle 
stays on the screen and does not disappear as it did on box_find.py
'''
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob

drawing = False
leftMouseLeft = False
mode = True
ix = -1
iy = -1
coordinates = []

#mouse callback functions
def crop(event, x, y, flags, param):
    global ix, iy, drawing, mode, leftMouseLeft, coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        leftMouseLeft = False
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix,iy), (x,y), (0,255,0), 3)  
            else:
                cv2.circle(img, (x,y), 5, (0,255,0), 1)
    elif event == cv2.EVENT_LBUTTONUP:
        leftMouseLeft = True
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix,iy), (x,y), (0,255,0),3)
            coordinates = (ix,iy,x,y)
        else:
            cv2.circle(img, (x, y), 5, (0,0,255), 1)


for image in glob.glob("*.jpg"):
    img = cv2.imread(image)
    img_copy = img.copy()        
    cv2.namedWindow('image', 0)
    cv2.setMouseCallback('image', crop)
    while(1):
        cv2.imshow('image', img)
        if not leftMouseLeft:
            img = img_copy.copy()
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
        else:
            continue

    rectangle = ()
    (ix,iy,x,y) = coordinates
    roi = img[iy:y,ix:x]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()