import difflib
print(difflib.SequenceMatcher(None, 'FA-235-FB', 'F4-235-FB').ratio())

"""
import pytesseract
import cv2

# Lisez l'image en utilisant OpenCV
image = cv2.imread("20221231-164105.jpg")

# Convertir l'image en noir et blanc (facultatif)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer le filtre GaussianBlur pour éliminer le bruit (facultatif)
gray = cv2.GaussianBlur(gray, (3,3), 0)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Appliquer l'OCR avec PyTesseract
text = pytesseract.image_to_string(gray)

print(text)"""