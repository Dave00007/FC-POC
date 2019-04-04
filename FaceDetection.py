import cv2


class FaceDetection:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('/home/dawid/Desktop/OpenCV/data/haarcascades'
                                                  '/haarcascade_frontalface_alt2.xml')

    def start_detection(self):
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
FaceDetection().start_detection()