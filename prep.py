import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

img = cv2.imread('images/keyboard.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 25, 100, apertureSize=3)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#print(contours)

keyContours = []

# Filter relevant contours
for c in contours:

    contourArea = cv2.contourArea(c)

    # Filter contours by size
    if contourArea < 1e3 or 1e5 < contourArea:
        continue

    # Draw the rectangles around the keys
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    keyContours.append([x, y, w, h])

keyContours = sorted(list(set(map(tuple, keyContours))), key=lambda k: [k[1], k[0]])
print(keyContours)

# Draws all countours, including numbers/words/keyboard
#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('images/highlighted_keyboard.png', img)