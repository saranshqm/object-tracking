
"""
ReadMe:

This code was created by Saransh Gupta, Third-year Undergraduate student at IIT Kharagpur

This code is submitted to Dr. Kobus Barnard email: kobus.barnard@gmail.com


This code will create three frames: 
1. Motion Detection by Image Difference and enhanced frame
2. Background subtracted frame
3. Original Frame

"""






# import all the necessary files

import cv2
import numpy as np
import time

#importing the video

cap = cv2.VideoCapture("file:///D:/remote%20projects/Dr.%20Kobus/10minCornerFloorOfTank.mov")

#creating a background subtractor funciton

subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=30, detectShadows=False)


#creating a motion tracker function to create the box

color=(255,0,0)
thickness=2

def equalizeHistColor(frame):
    # equalize the histogram of color image
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])     # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format



while True:
    _, frame1 = cap.read()
    #img = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    img = equalizeHistColor(frame1)
    
    img = equalizeHistColor(img)
    
    time.sleep(1/10)   # slight delay
    
    #ret, frame2 = cap.read()  # second image 
    
    _, frame2 = cap.read()
    #img = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    img2 = equalizeHistColor(frame2)
    
    img2 = equalizeHistColor(img2)
    
    img1 = cv2.absdiff(img,img2)  # image difference
    
    # get theshold image
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(21,21),0)
    ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_OTSU)
    
    
    # combine frame and the image difference
    img21 = cv2.addWeighted(img,0.1,img2,0.1,0)
    
    # get contours and set bounding box from contours
    img3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for c in contours:
            rect = cv2.boundingRect(c)
            height, width = img3.shape[:2]            
            if rect[2] > 0.1*height and rect[2] < 0.9*height and rect[3] > 0.1*width and rect[3] < 0.9*width: 
                x,y,w,h = cv2.boundingRect(c)            # get bounding box of largest contour
                img4=cv2.drawContours(img2, c, -1, color, thickness)
                img5 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)  # draw red bounding box in img
            else:
                img5=img21
    else:
        img5=img21
        
    # Display the resulting image
    cv2.imshow('Motion Detection by Image Difference and enhanced frame',img2)
    
    
    
    mask = subtractor.apply(img)
    
    #cv2.imshow("enhanced frame", img)
    cv2.imshow("background segmented", mask)
    cv2.imshow("original frame", frame1)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

###########################################################################################################
