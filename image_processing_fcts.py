import cv2
import pytesseract
import easyocr
import re 
import difflib
import numpy as np

### Read plate tesseract ocr
def read_plate(image,image_path):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # turn image into black and white
    
    gray = cv2.GaussianBlur(gray, (3,3), 0) # apply GaussianBlur filter to eliminate noise 
    #cv2.imshow("Grayscale befor scale up {}".format(image_path), gray)
    
    scaled_img = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite("{}scaled_image.jpg".format(image_path), scaled_img)
    
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # path of tesseract.exe 

    #text = pytesseract.image_to_string(gray)
    text2 = pytesseract.image_to_string(scaled_img)
    #print("pytesseract text : {}  and text2 : {}".format(text,text2))
    #cv2.waitKey(0)
    return text2.strip()   #, image_path

### Read plate easy ocr
def read_license_plate(image_path):
    reader = easyocr.Reader(['fr'])
    result = reader.readtext(image_path, paragraph="False", allowlist= "ABCDEFGHJKLMNPQRSTUWXYZ0123456789")
    return result[0][1], image_path # return license plate text and the image 

### compare two string
def similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()

### filter format of the plate AA-000-AA doesn't work yet
def filter_plate(licence_text):
    print("filter")    
    if re.match("^[A-Z]{2}-[0-9]{3}-[A-Z]{2}$", licence_text):
        print("valid format ")
    else :
        for index, ch in enumerate(licence_text):
            if index <= 1 or index >=5 and not re.match("^[A-Z]$", licence_text) :
                #print(index, ch)
                licence_texte= licence_text.replace("*",licence_text[index] )
                print (licence_texte)
            #if index > 1 or index <5 and not re.match("^[0-9]$", licence_text) :
              #  print(index, ch)
             #   re.sub("[A-Z]","*",licence_text)
    return licence_text
#######




### image processing after detection
def four_point_transform(image, pts):

	# obtain a consistent order of the points and unpack them
	# individually
	(tl, tr, br, bl) = pts
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def filter_image_by_threshold(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simple thresholding
    ret, binary_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Color-based thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,0,160], dtype=np.uint8)
    upper_white = np.array([255,40,255], dtype=np.uint8)
    color_threshold = cv2.inRange(hsv, lower_white, upper_white)
    
    return binary_threshold, color_threshold

def image_processing(image):
    #print("\n Image processing function\n")
    #image =cv2.imread(image)
    #print("\n 1 lecture\n")
    # B&W filter and color filter selecting only white
    binary, color = filter_image_by_threshold(image)
    #print("\n 2 filtre\n")
    # Find all contours 
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #print("\n  3 contour \n")
    # Sort contours by Area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #print("\n  4 sort  \n",contours)
    # Calculer le rectangle englobant le contour
    rect = cv2.minAreaRect(contours[0])

    # Récupérer les coins du rectangle
    box = cv2.boxPoints(rect)

    # Convertir les coordonnées en entiers
    box = np.int0(box)

    # oder points (top-left, top-right, bottom-right, bottom-left)
    rect = order_points(box)
   # print("\5\n",rect)
    # Draw contour 
    poly = cv2.polylines(image,[box],True,(0,255,255))

    # warped the detection
    warped_image = four_point_transform(image, rect)
    #cv2.namedWindow('Warped', cv2.WINDOW_NORMAL)
    #cv2.imshow("Warped", warped_image)
    #print("fin Image processing")
    return warped_image