import numpy as np
import cv2

profile_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
frontal_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
smile_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_smile.xml")

cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,720)

def rescale_frame(frame, percent = 0.75):
    w = int(frame.shape[1] * percent)
    h = int(frame.shape[0] * percent)
    return cv2.resize(frame, (w,h), interpolation = cv2.INTER_AREA)

while True:
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent = 0.3 )
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frontal_faces = frontal_face_cascade.detectMultiScale(grey, scaleFactor= 1.1, minNeighbors= 5)
    profile_faces = profile_face_cascade.detectMultiScale(grey, scaleFactor= 1.1, minNeighbors= 5)
    smile_faces = profile_face_cascade.detectMultiScale(grey,scaleFactor=1.1, minNeighbors=5)

    if( len(frontal_faces) > 1):
        print("Do it again")
        break

    for (x,y,w,h) in frontal_faces:
        print(x,y,w,h)
        color = (255,0,0)
        stroke = 1
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(20) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()