import cv2
import numpy as np
import time

faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
id = int(input('Enter id:'))
sampleNum = 0

while(True):
    ret, img = cam.read()
    faces = faceDetect.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.imwrite(r'C:\Users\Kiran\Desktop\mini_project\Face Recognition\dataSet\User.'+str(id)+'.'+str(sampleNum)+'.jpg',img[y:y+h,x:x+w])
        cv2.rectangle(img,(x, y), (x+w, y+h), (0, 255, 0), 2)
        sampleNum += 1
        print(sampleNum)
    cv2.imshow('Face', img)
    if(cv2.waitKey(1) & sampleNum == 200):
        break
    time.sleep(0.2)
cam.release()
cv2.destroyAllWindows()
#