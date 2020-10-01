import numpy as np
import cv2
from classes import  image_resize

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('data/third-party/frontalEyes35x16.xml')
glasses = cv2.imread("gözlük.png", -1)



while(True):
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces           = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray    = gray[y:y+h, x:x+h] # rec
        roi_color   = frame[y:y+h, x:x+h]
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
            glasses2 = image_resize(glasses.copy(), width=ew+10)

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                      #RGBA
                    if glasses2[i, j][3] != 0: # alpha 0
                        roi_color[ey + i, ex + j] = glasses2[i, j]


    

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()