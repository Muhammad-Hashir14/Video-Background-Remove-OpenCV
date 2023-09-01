import cv2
import cvzone
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

video = cv2.VideoCapture(0)
segmenter = SelfiSegmentation()
video.set(3, 640)
video.set(4, 480)

fps = cvzone.FPS()

img = cv2.imread("images/2.jpg")
img_size = cv2.resize(img, (640, 480))
print(img_size.shape)

while True:
    success, frame = video.read()
    removal = segmenter.removeBG(frame, img_size, threshold=0.75)
    imgst = cvzone.stackImages([frame, removal],2, 1)
    _, imgst = fps.update(imgst)
    cv2.imshow("Video Background change", imgst)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
   
video.release()
cv2.destroyAllWindows()