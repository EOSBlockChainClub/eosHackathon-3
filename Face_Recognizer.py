
import cv2
import os
import numpy as np
from PIL import Image


class Face_Recognizer(object):

    people = []
    y_labels = []
    x_train = []
    recognizer = None
    def __init__(self,rec):
        self.recognizer = rec


    @staticmethod
    def rescale_frame(frame, percent=0.75):
        w = int(frame.shape[1] * percent)
        h = int(frame.shape[0] * percent)
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    # Given images in the dataset add all of them to x_train and y_labels
    def set_training(self):
        frontal_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "faces")

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png"):
                    path = os.path.join(root, file)
                    image = Image.open(path)
                    size = (96,96)
                    final_image = image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image, "uint8")
                    self.x_train.append(image_array)
                    self.y_labels.append(self.people.index(root[root.rfind('/') + 1:]))

    def set_people(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "faces")
        list = os.listdir(image_dir)  # dir is your directory path
        self.people = list[1:]




    # Adding the photos of the person in the directory
    def set_person(self, userID, face_cascade, message):
        if not os.path.exists("faces/" + userID):
            os.makedirs("faces/" + userID)

        cap = cv2.VideoCapture(0)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "faces/"+userID)
        list = os.listdir(image_dir)  # dir is your directory path
        i = len(list) - 1
        start = False
        k = 0
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, message, (40,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            frame = Face_Recognizer.rescale_frame(frame, percent=0.3)
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)
            if(len(faces) > 1):
                print("Do it again")
                break;

            if start or cv2.waitKey(2) == ord('k'):
                start = True

            if start:
                for (x,y,w,h) in faces:
                    to_save = grey[y:y+h, x:x+w]
                    size = (96, 96)
                    final_image = cv2.resize(to_save, size, interpolation=cv2.INTER_CUBIC)
                    img_name = "faces/" + userID + "/" + (i + k).__str__() + ".png"
                    k = k + 1
                    cv2.imwrite(img_name,final_image)

            if( k > 9):
                break

            cv2.imshow("Frame", frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        if not userID in self.people:
            self.people.append(userID)

        cap.release()
        cv2.destroyAllWindows()

    def face_train(self):
        self.recognizer.train(np.array(self.x_train), np.array(self.y_labels))
        self.recognizer.save("recognizers/face-trainner.yml")
        #self.recognizer.compile(optimizer='adam', loss= Face_Recognizer.triplet_loss, metrics=['accuracy'])
        #load_weights_from_FaceNet(self.recognizer)

    def recognize(self, image):
        label, conf = self.recognizer.predict(image)
        if conf >= 97:
            return self.people[label]
