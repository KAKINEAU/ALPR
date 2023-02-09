import cv2
"""
# Load image
image = cv2.imread("crop2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours in the image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area (this should be the contour of the license plate)
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour with a rectangle
approx_points = cv2.approxPolyDP(largest_contour, epsilon=0.02*cv2.arcLength(largest_contour, True), closed=True)

# Find the top-left, top-right, bottom-right, and bottom-left points of the rectangle
top_left = approx_points[0][0]
top_right = approx_points[1][0]
bottom_right = approx_points[2][0]
bottom_left = approx_points[3][0]

# Draw the rectangle on the image
cv2.drawContours(image, [approx_points], -1, (0,255,0), 2)

# Show the image with the rectangle drawn
cv2.imshow("License Plate", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Load image
image = cv2.imread("crop2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours in the edges image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store the coordinates of the 4 points
top_left = (0, 0)
top_right = (0, 0)
bottom_left = (0, 0)
bottom_right = (0, 0)

# Iterate over the contours
for c in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(c)

    # Compare the bounding rectangle coordinates to the current top-left, top-right, bottom-left, and bottom-right coordinates
    if x < top_left[0] or (x == top_left[0] and y < top_left[1]):
        top_left = (x, y)
    if x + w > top_right[0] or (x + w == top_right[0] and y < top_right[1]):
        top_right = (x + w, y)
    if x < bottom_left[0] or (x == bottom_left[0] and y + h > bottom_left[1]):
        bottom_left = (x, y + h)
    if x + w > bottom_right[0] or (x + w == bottom_right[0] and y + h > bottom_right[1]):
        bottom_right = (x + w, y + h)

# Draw circles at the 4 points
cv2.circle(image, top_left, 5, (0, 0, 255), -1)
cv2.circle(image, top_right, 5, (0, 0, 255), -1)
cv2.circle(image, bottom_left, 5, (0, 0, 255), -1)
cv2.circle(image, bottom_right, 5, (0, 0, 255), -1)

# Show the image with the circles
cv2.imshow("Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
import numpy as np
"""
def adjust_rectangle(image, xmin, ymin, xmax, ymax):
    # Define the four corner points of the rectangle
    pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    # Define the four corner points of the desired rectangle
    pts2 = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # Apply the perspective transformation to the image
    result = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return result


image = cv2.imread("travers.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

warped= adjust_rectangle(image, "731", "802", "1270", "953")
#print(warped)
#cv2.imshow("Warped", warped)


def detect_license_plate(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Use HoughLinesP to detect lines in the edges image
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    # Iterate over the lines and draw them on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Display the output
    cv2.imshow("License Plate Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread("crop2.jpg")
detect_license_plate(image)
"""

#1 threshold
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


# 2 Canny edges 
"""
def canny_edge(image):
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Find edges that have 4 sides
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    NumberPlateCnt = None
    found = False
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            found = True
            break
    if found:
        cv2.drawContours(image, [NumberPlateCnt], -1, (255, 0, 0), 2)
        return image, NumberPlateCnt
    else:
        return image, None
"""
def canny_edge(img):
    gaus = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(gaus, 50, 150)
    return edges
"""
def contour(image):
    # Find edges that have 4 sides
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    NumberPlateCnt = None
    found = False
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            found = True
            break
    if found:
        cv2.drawContours(image, [NumberPlateCnt], -1, (255, 0, 0), 2)
        return image, NumberPlateCnt
    else:
        return image, None

def contour(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    NumberPlateCnt = None
    found = False
    
    # Check if contour has 4 sides (rectangle)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            found = True
            NumberPlateCnt = approx
            break
    
    if found:
        cv2.drawContours(image, [NumberPlateCnt], -1, (255, 0, 255), 2)
    else:
        # Use convex hull to filter contours
        hull = cv2.convexHull(contours[0])
        cv2.drawContours(image, [hull], -1, (255, 0, 255), 2)
    
    return image
"""
#img = cv2.imread("test2.jpg")
img = cv2.imread("licence.jpg")
#img = cv2.imread("travers.jpg")
binary , color = filter_image_by_threshold(img)
cv2.namedWindow('bin', cv2.WINDOW_NORMAL)
cv2.namedWindow('color', cv2.WINDOW_NORMAL)
cv2.imshow("bin",binary)
cv2.imshow("color",color)

imag  = canny_edge(color)
cv2.namedWindow('canny image', cv2.WINDOW_NORMAL)
cv2.imshow("canny image",imag)

#imag2 =contour(color)
#cv2.imshow("contour with color",imag2)
#cv2.imshow("Numberplatecnt", NumberPlatecnt)

# Trouver les contours dans l'image
contours, _ = cv2.findContours(color, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#contourscanny, _ = cv2.findContours(imag, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# Trier les contours en fonction de leur aire
contours = sorted(contours, key=cv2.contourArea, reverse=True)
#contourscanny = sorted(contourscanny, key=cv2.contourArea, reverse=True)
#print(contours[0])
# Dessiner un contour autour de la plus grande surface de blanc
contourcolor =cv2.drawContours(color, [contours[-1]], 0, 255, 3)
#contourcanny =cv2.drawContours(imag, contourscanny[-1], -1, (0, 0, 255), 2)
# Afficher l'image avec le contour
cv2.namedWindow('Image color avec contour', cv2.WINDOW_NORMAL)
cv2.imshow("Image color avec contour", contourcolor)
#cv2.imshow("Image canny avec contour", contourcanny)

"""
x, y, w, h = cv2.boundingRect(contours[0])
print("Coordonnées du coin supérieur gauche :", x, y)
print("Largeur :", w)
print("Hauteur :", h)
plate = img[y:y+h, x:x+w]
cv2.imshow("plate",plate)
"""


# Calculer le rectangle englobant le contour
rect = cv2.minAreaRect(contours[0])

# Récupérer les coins du rectangle
box = cv2.boxPoints(rect)
# Convertir les coordonnées en entiers
box = np.int0(box)

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


print('box',box)
rect = order_points(box)
poly = cv2.polylines(img,[box],True,(0,255,255))
cv2.imshow("polylines",poly)
print("rect",rect)


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
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


warped = four_point_transform(img, rect)
cv2.namedWindow('Warped', cv2.WINDOW_NORMAL)
cv2.imshow("Warped", warped)


cv2.waitKey(0)

cv2.destroyAllWindows()