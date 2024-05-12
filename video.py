"""
@author: Uday Rallabhandi

Use OpenCV to generate a video file .avi from .png

Must name .png files as imshow-### and must be in the folder path specified

fourcc code for windows: DIVX
fourcc code for iOS: UNKNOWN
"""

import cv2

#image parameters
folder_path=r'.' #where the images are
number_of_images=100

#import one image as numpy array to get dimensions
image=cv2.imread(folder_path+"/imshow-000.png")
height,width,layers=image.shape 

#video parameters
fps=25 #can have this be num_images/time wanted
size=(width, height)
filename='video.avi'
fourcc = cv2.VideoWriter_fourcc('D','I','V','X') #windows fourCC code (for compression format)

#create video object
video=cv2.VideoWriter(filename, fourcc, fps, size)
video.write(image)

#write the images (numpy arrays) to the video object
for i in range(1,number_of_images):
    video.write(cv2.imread(folder_path+"/imshow-%03d.png"%i)) 
    
#close video file
video.release()
