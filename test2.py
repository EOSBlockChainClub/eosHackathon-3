import cv2
from Face_Recognizer import Face_Recognizer
recon = cv2.face.EigenFaceRecognizer_create()
recog = Face_Recognizer(recon)


frontal_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
profile_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

recog.set_person("Veronika", frontal_face_cascade, "Show the face")
recog.set_person("Veronika", profile_face_cascade, "Turn_right_left")
