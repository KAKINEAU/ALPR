from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import easyocr
# pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# #img = r"C:\Projet3M\Projet 3M machine learning\yolov7\crop\roi__0.jpg"
# img = cv2.imread(r"C:\Projet3M\Projet 3M machine learning\yolov7\crop\roi__0.jpg",cv2.IMREAD_COLOR)
# img = cv2.resize(img, (620,480) )
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #destination_image = cv2.bilateralFilter(source_image, diameter of pixel, sigmaColor, sigmaSpace)
# #sigmaColor  and sigmaSpace are values from 15 to above to blur out background information.
# gray = cv2.bilateralFilter(gray, 13, 15, 15)
# edged = cv2.Canny(gray, 30, 200)
# contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
# for c in contours:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#     if len(approx) == 4:
#         screenCnt = approx
#         break
# if screenCnt is None:
#     detected = 0
#     print ("No contour detected")
# else:
#      detected = 1
# if detected == 1:
#     cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
# mask = np.zeros(gray.shape,np.uint8)
# new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
# new_image = cv2.bitwise_and(img,img,mask=mask)
# (x, y) = np.where(mask == 255)
# (topx, topy) = (np.min(x), np.min(y))
# (bottomx, bottomy) = (np.max(x), np.max(y))
# Cropped = gray[topx:bottomx+1, topy:bottomy+1]
# text = pytesseract.image_to_string(Cropped, config='--psm 11')

import os
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
IMAGE_PATH = r"C:\Projet3M\Projet 3M machine learning\yolov7\crop\roi__0.jpg"
reader = easyocr.Reader(['en'])
result = reader.readtext(IMAGE_PATH,paragraph="False")
result
print(result)
#print(pytesseract.image_to_string(Image.open(img)))

# # point to license plate image (works well with custom crop function)
# gray = cv2.imread(r"C:\Projet3M\Projet 3M machine learning\yolov7\crop\2.jpg", 0)
# gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
# blur = cv2.GaussianBlur(gray, (5,5), 0)
# gray = cv2.medianBlur(gray, 3)
# # perform otsu thresh (using binary inverse since opencv contours work better with white text)
# ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# cv2.imshow("Otsu", thresh)
# cv2.waitKey(0)
# rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# # apply dilation 
# dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
# #cv2.imshow("dilation", dilation)
# #cv2.waitKey(0)
# # find contours
# try:
#     contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# except:
#     ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# # create copy of image
# im2 = gray.copy()

# plate_num = ""
# # loop through contours and find letters in license plate
# for cnt in sorted_contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     height, width = im2.shape
    
#     # if height of box is not a quarter of total height then skip
#     if height / float(h) > 6: continue
#     ratio = h / float(w)
#     # if height to width ratio is less than 1.5 skip
#     if ratio < 1.5: continue
#     area = h * w
#     # if width is not more than 25 pixels skip
#     if width / float(w) > 15: continue
#     # if area is less than 100 pixels skip
#     if area < 100: continue
#     # draw the rectangle
#     rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
#     roi = thresh[y-5:y+h+5, x-5:x+w+5]
#     roi = cv2.bitwise_not(roi)
#     roi = cv2.medianBlur(roi, 5)
#     #cv2.imshow("ROI", roi)
#     #cv2.waitKey(0)
#     text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
#     #print(text)
#     plate_num += text
# print(plate_num)