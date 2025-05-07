import cv2
import os
import sys
from simple_facerec import SimpleFacerec

script_directory = os.path.dirname(os.path.abspath(sys.argv[0])) 

sfr = SimpleFacerec()
sfr.load_encoding_images(script_directory + "/images")

# load camera
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.1.43:81/stream")

while True:
    ret, frame = cap.read()

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    if len(face_names) > 0:
        break

cap.release()
cv2.destroyAllWindows()
