
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.SerialModule import SerialObject

 
cap = cv2.VideoCapture(0)
 
detector = HandDetector(
    maxHands=1,
    detectionCon=0.5
)
mySerial=SerialObject("COM18", 9600, 1)
 
while True:
    success, img = cap.read()
 
    if not success:
        continue
 
    # Detect hand and draw landmarks
    hands, img = detector.findHands(img, draw=True)
 
    if hands:
        hand = hands[0]
 
        lmList = hand["lmList"]   # 21 [x, y, z] landmark points
        bbox = hand["bbox"]       # bounding box around the hand
 
        fingers = detector.fingersUp(hand)  # e.g. [0, 1, 1, 0, 0]
        print(fingers)
        mySerial.sendData(fingers)
 
    cv2.imshow("Image", img)
 
    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()