import cv2
import pytesseract


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 3)


def thresholding(image):
    return cv2.threshold(image, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def ocr_core(image):

    text = pytesseract.image_to_string(image, lang='tur+eng')
    return text



img = cv2.imread('/home/huawei/Documents/internship/Tasks/ocr/1.jpg')



img = get_grayscale(img)
img = thresholding(img)
img = remove_noise(img)


print(ocr_core(img))