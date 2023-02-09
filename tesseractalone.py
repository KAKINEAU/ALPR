import cv2
import pytesseract
import easyocr
import re 
import difflib
import numpy as np
import matplotlib.pyplot as plt

def read_plate(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # turn image into black and white
    
    gray = cv2.GaussianBlur(gray, (3,3), 0) # apply GaussianBlur filter to eliminate noise 
    cv2.imshow("Grayscale befor scale up {}".format(image_path), gray)
    
    scaled_img = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("{}scaled_image.jpg".format(image_path), scaled_img)
    
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # path of tesseract.exe 

    text = pytesseract.image_to_string(gray)
    text2 = pytesseract.image_to_string(scaled_img)
    print("pytesseract text : {}  and text2 : {}".format(text,text2))
    cv2.waitKey(0)
    return text.strip()   #, image_path

read_plate("20230205-145404.jpgscaled_image.jpg")