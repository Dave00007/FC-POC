import cv2
import os
import imutils


class FaceGatherer:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('/home/dawid/Desktop/OpenCV/data/haarcascades'
                                                  '/haarcascade_frontalface_alt2.xml')

    def gather_faces(self):
        image_counter = 0
        captured = cv2.VideoCapture(
            'udpsrc port=9000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=('
            'string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink',
            cv2.CAP_GSTREAMER)
        while True:
            # Capture frame-by-frame
            ret, frame = captured.read()
            orig = frame.copy()
            frame = imutils.resize(frame, width=400)
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                color = (255, 0, 0)  # BGR 0-255
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('k'):
                p = os.path.sep.join(['dataset/dawid', "{}.png".format(
                    str(image_counter).zfill(5))])
                cv2.imwrite(p, orig)
                print('Image numer:' + str(image_counter) + 'saved')
                image_counter += 1

            if key == ord('q'):
                break

        # When everything done, release the capture
        captured.release()
        cv2.destroyAllWindows()

a= FaceGatherer().gather_faces()