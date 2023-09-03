class LiveDetection(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('app/trained_models/face/haarcascade_frontalface_default.xml')
        self.model = tf.keras.models.load_model('app/trained_models/face/face_v2.h5')
        self.cap = cv2.VideoCapture(0)
        while True:
            self._, self.img = self.cap.read()
            self.img = cv2.flip(self.img, 1)
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.faces = self.face_cascade.detectMultiScale(self.gray, 1.1, 4)
            self.class_ = ""
            try:
                for (x, y, w, h) in self.faces:
                    self.face = self.img[y-50:y+h+50, x-50:x+w+50]
                    self.face = cv2.resize(self.face, (48, 48))
                    self.face_array = np.array(self.face)
                    self.face_array = np.expand_dims(self.face_array, axis=0)
                    self.prediction = np.argmax(self.model.predict([self.face_array]), axis=1)
                    self.class_ = indicies[self.prediction[0]]
                    self.font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(self.img, self.class_, (x, y), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(self.img, (x-25, y-25), (x+w+25, y+h+25), (0, 255, 0), 2)
            except Exception as ex:
                self.class_ = ""
            cv2.imshow('Face Expression Classifier', self.img)
            self.k = cv2.waitKey(1) & 0xff
            if self.k == 27:
                break
        self.cap.release()
if __name__ == "__main__":
    import cv2
    import numpy as np
    import tensorflow as tf
    indicies = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
    app = LiveDetection()