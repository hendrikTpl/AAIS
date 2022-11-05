import cv2
import os
import numpy as np
""" 
This is a very simple approach by detecting darker area in the circles region of the pipes

"""
#file path
root_dir = '/home/infinity/github/AAIS/Datasets/pipes'
pipe_fname ='sample_002.jpg'
# load pipe images
pipe_img = cv2.imread(os.path.join(root_dir,pipe_fname), cv2.IMREAD_GRAYSCALE)
x,y,w,h = 0,0,pipe_img.shape[0],pipe_img.shape[1]

# parameters.
params = cv2.SimpleBlobDetector_Params()
# thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
# Filter by Area.
params.filterByArea = True
params.minArea = 150
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
 
# pipes detector with the parameters
cvversion = (cv2.__version__).split('.')
if int(cvversion[0]) < 3 :
    pipeDetection = cv2.SimpleBlobDetector(params)
else : 
    pipeDetection = cv2.SimpleBlobDetector_create(params)
    
# blobs detectors
KeyPoint = pipeDetection.detect(pipe_img)
 
# Draw detected blobs
pipeImage = cv2.drawKeypoints(pipe_img, KeyPoint, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# #Show 
print('Total: {} pipes'.format(len(KeyPoint)))
cv2.imshow("Pipes Images", pipeImage)
cv2.putText(img=pipeImage,text=('Total: '+ str(len(KeyPoint))+'Pipes'),org=(x + int(w/10),y + int(h/1.5)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(255,0,0), thickness=7)
cv2.waitKey(0)