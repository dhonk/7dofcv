import cv2 as cv
import sys

cap = cv.VideoCapture('udp://0.0.0.0:5005')

if not cap.isOpened():
    print("Issue Opening Camera")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Issue getting frame")
        break

    cv.imshow('Feed', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
