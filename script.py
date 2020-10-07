import cv2
import sys

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyecascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smilecascade=cv2.CascadeClassifier("haarcascade_smile.xml")

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    eyes= eyecascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=9
    )

    smile= smilecascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=20
    )


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    for (xe, ye, we, he) in eyes:
        cv2.rectangle(frame, (xe, ye), (xe+we, ye+he), (0, 255, 0), 2) 

    for (xs, ys, ws, hs) in smile:
        cv2.rectangle(frame, (xs, ys), (xs+ws, ys+hs), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()