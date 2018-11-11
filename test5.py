import cv2
from PIL import Image
from cv2.face import MACE_create, EigenFaceRecognizer_create

from Face_Recognizer import Face_Recognizer
#from inception_blocks_v2 import faceRecoModel

frontal_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recon = cv2.face.EigenFaceRecognizer_create()
#recon = cv2.face.MACE_create()
#recon = cv2.face.FisherFaceRecognizer_create
#


recog = Face_Recognizer(recon)
recog.set_people()
recog.set_training()
recog.face_train()



cap = cv2.VideoCapture(0)

def rescale_frame(frame, percent = 0.75):
    w = int(frame.shape[1] * percent)
    h = int(frame.shape[0] * percent)
    return cv2.resize(frame, (w,h), interpolation = cv2.INTER_AREA)

while True:
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent = 0.3 )
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frontal_faces = frontal_face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in frontal_faces:
        to_check = grey[y:y+h,x:x+w]
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = (96,96)
        final_image = cv2.resize(to_check, size, interpolation=cv2.INTER_CUBIC)
        name =  recog.recognize(final_image)
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()