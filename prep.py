import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

keyCaps = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "BACKSPACE",
           "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "ENTER",
           "A", "S", "D", "F", "G", "H", "J", "K", "L",
           "SHIFT", "Z", "X", "C", "V", "B", "N", "M", ",", ".",
           "SPACE", "BORDER"]



img = cv2.imread('images/keyboardShift.png')
cv2.imwrite('images/preppedKeyboard.png', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 15, 100, apertureSize=3)

contours, _ = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

keyContours = []
borderContour = []

# Filter relevant contours
for c in contours:

    contourArea = cv2.contourArea(c)

    # Filter contours by size
    if contourArea < 1e3:
        continue

    # Detecting the border contour separately 
    if contourArea > 1e5 and not contourArea > 1e7:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        borderContour = [x, y, w, h]
        continue

    if contourArea > 1e5:
        continue

    # Draw the rectangles around the keys
    x, y, w, h = cv2.boundingRect(c)

    overlapping = False

    # Prevents overlapping rectangles for the same key
    for k in keyContours:
        if k[0] <= x and k[1] <= y and k[0] + k[2] >= x + w and k[1] + k[3] >= y + h:
            overlapping = True
            break

    if not overlapping:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        keyContours.append([x, y, w, h])

keyContours = sorted(list(set(map(tuple, keyContours))),
                        key=lambda k: [k[1], k[0]])

# Add the keyboard contour at the end
keyContours.append(borderContour)

# set each to point to the specific key
i = 0
keyCoords = {}
for keyCap in keyCaps:
    keyCoords[keyCap] = [
        (keyContours[i][0], keyContours[i][1]),
        (keyContours[i][0] + keyContours[i][2], keyContours[i][1]),
        (keyContours[i][0] + keyContours[i][2],
            keyContours[i][1] + keyContours[i][3]),
        (keyContours[i][0], keyContours[i][1] + keyContours[i][3])
    ]
    i = i + 1

print(keyContours)
print(' ')
print(keyCoords)

cv2.imwrite('images/highlighted_keyboard.png', img)

with open('prep', 'wb') as prep_file:
    pickle.dump(keyCoords, prep_file)
