import cv2
import time
cam = cv2.VideoCapture(1)
time.sleep(1)
_,img = cam.read()
cv2.imwrite("ImageFromCamera2.jpg", img)
cam.release()