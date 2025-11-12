import cv2
from PIL import Image
import pytesseract


myconfig = r'--psm 11 --oem 1'

text = pytesseract.image_to_string(Image.open('mektup_1.jpg'), lang='tur+eng', config=myconfig)
print(text)


img = cv2.imread('mektup_1.jpg')

height, weight, _ = img.shape

boxes = pytesseract.image_to_boxes(img, lang='tur+eng', config=myconfig)


for box in boxes.splitlines():
    
    box = box.split(' ')

    img= cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0))
    

cv2.imshow('img', img)
cv2.waitKey(0)

