import cv2
import pickle


class FaceRecognizer:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('/home/dawid/Desktop/OpenCV/data/haarcascades'
                                                  '/haarcascade_frontalface_alt2.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {"person_name": 1}
        with open("labels.pickle", 'rb') as f:
            self.og_labels = pickle.load(f)
            self.labels = {v:k for k,v in self.og_labels.items()}
        self.recognizer.read("trainer.yml")

    def start_recognizing(self):
        captured = cv2.VideoCapture(
            'udpsrc port=9000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=('
            'string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink',
            cv2.CAP_GSTREAMER)

        while True:
            # Capture frame-by-frame
            ret, frame = captured.read()
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                print(x, y, w, h)

                roi_gray = gray[y:y+h, x:x+w]
                id_, conf = self.recognizer.predict(roi_gray)
                if conf >= 45:

                    print(str(conf) + " " + self.labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = self.labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, name, (x,y), font, 1 , color, stroke, cv2.LINE_AA)

                color = (255, 0, 0)  # BGR 0-255
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        captured.release()
        cv2.destroyAllWindows()

FaceRecognizer().start_recognizing()