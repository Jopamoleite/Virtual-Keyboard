import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def prep():

    img = cv2.imread('images/keyboard.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 15, 100, apertureSize=3)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)

    keyContours = []

    # Filter relevant contours
    for c in contours:

        contourArea = cv2.contourArea(c)

        # Filter contours by size
        if contourArea < 1e3 or 1e5 < contourArea:
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

    # set each to point to the specific key
    keyCoords = {
        "1": [
            (keyContours[0][0], keyContours[0][1]),
            (keyContours[0][0] + keyContours[0][2], keyContours[0][1]),
            (keyContours[0][0] + keyContours[0][2],
             keyContours[0][1] + keyContours[0][3]),
            (keyContours[0][0], keyContours[0][1] + keyContours[0][3])
        ],
        "2": [
            (keyContours[1][0], keyContours[1][1]),
            (keyContours[1][0] + keyContours[1][2], keyContours[0][1]),
            (keyContours[1][0] + keyContours[1][2],
             keyContours[1][1] + keyContours[1][3]),
            (keyContours[1][0], keyContours[1][1] + keyContours[0][3])
        ],
        "3": [
            (keyContours[2][0], keyContours[2][1]),
            (keyContours[2][0] + keyContours[2][2], keyContours[0][1]),
            (keyContours[2][0] + keyContours[2][2],
             keyContours[2][1] + keyContours[2][3]),
            (keyContours[2][0], keyContours[2][1] + keyContours[0][3])
        ],
        "4": [
            (keyContours[3][0], keyContours[3][1]),
            (keyContours[3][0] + keyContours[3][2], keyContours[0][1]),
            (keyContours[3][0] + keyContours[3][2],
             keyContours[3][1] + keyContours[3][3]),
            (keyContours[3][0], keyContours[3][1] + keyContours[0][3])
        ],
        "5": [
            (keyContours[4][0], keyContours[4][1]),
            (keyContours[4][0] + keyContours[4][2], keyContours[0][1]),
            (keyContours[4][0] + keyContours[4][2],
             keyContours[4][1] + keyContours[4][3]),
            (keyContours[4][0], keyContours[4][1] + keyContours[0][3])
        ],
        "6": [
            (keyContours[5][0], keyContours[5][1]),
            (keyContours[5][0] + keyContours[5][2], keyContours[0][1]),
            (keyContours[5][0] + keyContours[5][2],
             keyContours[5][1] + keyContours[5][3]),
            (keyContours[5][0], keyContours[5][1] + keyContours[0][3])
        ],
        "7": [
            (keyContours[6][0], keyContours[6][1]),
            (keyContours[6][0] + keyContours[6][2], keyContours[0][1]),
            (keyContours[6][0] + keyContours[6][2],
             keyContours[6][1] + keyContours[6][3]),
            (keyContours[6][0], keyContours[6][1] + keyContours[0][3])
        ],
        "8": [
            (keyContours[7][0], keyContours[7][1]),
            (keyContours[7][0] + keyContours[7][2], keyContours[0][1]),
            (keyContours[7][0] + keyContours[7][2],
             keyContours[7][1] + keyContours[7][3]),
            (keyContours[7][0], keyContours[7][1] + keyContours[0][3])
        ],
        "9": [
            (keyContours[8][0], keyContours[8][1]),
            (keyContours[8][0] + keyContours[8][2], keyContours[0][1]),
            (keyContours[8][0] + keyContours[8][2],
             keyContours[8][1] + keyContours[8][3]),
            (keyContours[8][0], keyContours[8][1] + keyContours[0][3])
        ],
        "0": [
            (keyContours[9][0], keyContours[9][1]),
            (keyContours[9][0] + keyContours[9][2], keyContours[0][1]),
            (keyContours[9][0] + keyContours[9][2],
             keyContours[9][1] + keyContours[9][3]),
            (keyContours[9][0], keyContours[9][1] + keyContours[0][3])
        ],
        "BACKSPACE": [
            (keyContours[10][0], keyContours[10][1]),
            (keyContours[10][0] + keyContours[10][2], keyContours[0][1]),
            (keyContours[10][0] + keyContours[10][2],
             keyContours[10][1] + keyContours[10][3]),
            (keyContours[10][0], keyContours[10][1] + keyContours[0][3])
        ],
        "Q": [
            (keyContours[11][0], keyContours[11][1]),
            (keyContours[11][0] + keyContours[11][2], keyContours[0][1]),
            (keyContours[11][0] + keyContours[11][2],
             keyContours[11][1] + keyContours[11][3]),
            (keyContours[11][0], keyContours[11][1] + keyContours[0][3])
        ],
        "W": [
            (keyContours[12][0], keyContours[12][1]),
            (keyContours[12][0] + keyContours[12][2], keyContours[0][1]),
            (keyContours[12][0] + keyContours[12][2],
             keyContours[12][1] + keyContours[12][3]),
            (keyContours[12][0], keyContours[12][1] + keyContours[0][3])
        ],
        "E": [
            (keyContours[13][0], keyContours[13][1]),
            (keyContours[13][0] + keyContours[13][2], keyContours[0][1]),
            (keyContours[13][0] + keyContours[13][2],
             keyContours[13][1] + keyContours[13][3]),
            (keyContours[13][0], keyContours[13][1] + keyContours[0][3])
        ],
        "R": [
            (keyContours[14][0], keyContours[14][1]),
            (keyContours[14][0] + keyContours[14][2], keyContours[0][1]),
            (keyContours[14][0] + keyContours[14][2],
             keyContours[14][1] + keyContours[14][3]),
            (keyContours[14][0], keyContours[14][1] + keyContours[0][3])
        ],
        "T": [
            (keyContours[15][0], keyContours[15][1]),
            (keyContours[15][0] + keyContours[15][2], keyContours[0][1]),
            (keyContours[15][0] + keyContours[15][2],
             keyContours[15][1] + keyContours[15][3]),
            (keyContours[15][0], keyContours[15][1] + keyContours[0][3])
        ],
        "Y": [
            (keyContours[16][0], keyContours[16][1]),
            (keyContours[16][0] + keyContours[16][2], keyContours[0][1]),
            (keyContours[16][0] + keyContours[16][2],
             keyContours[16][1] + keyContours[16][3]),
            (keyContours[16][0], keyContours[16][1] + keyContours[0][3])
        ],
        "U": [
            (keyContours[17][0], keyContours[17][1]),
            (keyContours[17][0] + keyContours[17][2], keyContours[0][1]),
            (keyContours[17][0] + keyContours[17][2],
             keyContours[17][1] + keyContours[17][3]),
            (keyContours[17][0], keyContours[17][1] + keyContours[0][3])
        ],
        "I": [
            (keyContours[18][0], keyContours[18][1]),
            (keyContours[18][0] + keyContours[18][2], keyContours[0][1]),
            (keyContours[18][0] + keyContours[18][2],
             keyContours[18][1] + keyContours[18][3]),
            (keyContours[18][0], keyContours[18][1] + keyContours[0][3])
        ],
        "O": [
            (keyContours[19][0], keyContours[19][1]),
            (keyContours[19][0] + keyContours[19][2], keyContours[0][1]),
            (keyContours[19][0] + keyContours[19][2],
             keyContours[19][1] + keyContours[19][3]),
            (keyContours[19][0], keyContours[19][1] + keyContours[0][3])
        ],
        "P": [
            (keyContours[20][0], keyContours[20][1]),
            (keyContours[20][0] + keyContours[20][2], keyContours[0][1]),
            (keyContours[20][0] + keyContours[20][2],
             keyContours[20][1] + keyContours[20][3]),
            (keyContours[20][0], keyContours[20][1] + keyContours[0][3])
        ],
        "ENTER": [
            (keyContours[21][0], keyContours[21][1]),
            (keyContours[21][0] + keyContours[21][2], keyContours[0][1]),
            (keyContours[21][0] + keyContours[21][2],
             keyContours[21][1] + keyContours[21][3]),
            (keyContours[21][0], keyContours[21][1] + keyContours[0][3])
        ],
        "A": [
            (keyContours[22][0], keyContours[22][1]),
            (keyContours[22][0] + keyContours[22][2], keyContours[0][1]),
            (keyContours[22][0] + keyContours[22][2],
             keyContours[22][1] + keyContours[22][3]),
            (keyContours[22][0], keyContours[22][1] + keyContours[0][3])
        ],
        "S": [
            (keyContours[23][0], keyContours[23][1]),
            (keyContours[23][0] + keyContours[23][2], keyContours[0][1]),
            (keyContours[23][0] + keyContours[23][2],
             keyContours[23][1] + keyContours[23][3]),
            (keyContours[23][0], keyContours[23][1] + keyContours[0][3])
        ],
        "D": [
            (keyContours[24][0], keyContours[24][1]),
            (keyContours[24][0] + keyContours[24][2], keyContours[0][1]),
            (keyContours[24][0] + keyContours[24][2],
             keyContours[24][1] + keyContours[24][3]),
            (keyContours[24][0], keyContours[24][1] + keyContours[0][3])
        ],
        "F": [
            (keyContours[25][0], keyContours[25][1]),
            (keyContours[25][0] + keyContours[25][2], keyContours[0][1]),
            (keyContours[25][0] + keyContours[25][2],
             keyContours[25][1] + keyContours[25][3]),
            (keyContours[25][0], keyContours[25][1] + keyContours[0][3])
        ],
        "G": [
            (keyContours[26][0], keyContours[26][1]),
            (keyContours[26][0] + keyContours[26][2], keyContours[0][1]),
            (keyContours[26][0] + keyContours[26][2],
             keyContours[26][1] + keyContours[26][3]),
            (keyContours[26][0], keyContours[26][1] + keyContours[0][3])
        ],
        "H": [
            (keyContours[27][0], keyContours[27][1]),
            (keyContours[27][0] + keyContours[27][2], keyContours[0][1]),
            (keyContours[27][0] + keyContours[27][2],
             keyContours[27][1] + keyContours[27][3]),
            (keyContours[27][0], keyContours[27][1] + keyContours[0][3])
        ],
        "J": [
            (keyContours[28][0], keyContours[28][1]),
            (keyContours[28][0] + keyContours[28][2], keyContours[0][1]),
            (keyContours[28][0] + keyContours[28][2],
             keyContours[28][1] + keyContours[28][3]),
            (keyContours[28][0], keyContours[28][1] + keyContours[0][3])
        ],
        "K": [
            (keyContours[29][0], keyContours[29][1]),
            (keyContours[29][0] + keyContours[29][2], keyContours[0][1]),
            (keyContours[29][0] + keyContours[29][2],
             keyContours[29][1] + keyContours[29][3]),
            (keyContours[29][0], keyContours[29][1] + keyContours[0][3])
        ],
        "L": [
            (keyContours[30][0], keyContours[30][1]),
            (keyContours[30][0] + keyContours[30][2], keyContours[0][1]),
            (keyContours[30][0] + keyContours[30][2],
             keyContours[30][1] + keyContours[30][3]),
            (keyContours[30][0], keyContours[30][1] + keyContours[0][3])
        ],
        "Z": [
            (keyContours[31][0], keyContours[31][1]),
            (keyContours[31][0] + keyContours[31][2], keyContours[0][1]),
            (keyContours[31][0] + keyContours[31][2],
             keyContours[31][1] + keyContours[31][3]),
            (keyContours[31][0], keyContours[31][1] + keyContours[0][3])
        ],
        "X": [
            (keyContours[32][0], keyContours[32][1]),
            (keyContours[32][0] + keyContours[32][2], keyContours[0][1]),
            (keyContours[32][0] + keyContours[32][2],
             keyContours[32][1] + keyContours[32][3]),
            (keyContours[32][0], keyContours[32][1] + keyContours[0][3])
        ],
        "C": [
            (keyContours[33][0], keyContours[33][1]),
            (keyContours[33][0] + keyContours[33][2], keyContours[0][1]),
            (keyContours[33][0] + keyContours[33][2],
             keyContours[33][1] + keyContours[33][3]),
            (keyContours[33][0], keyContours[33][1] + keyContours[0][3])
        ],
        "V": [
            (keyContours[34][0], keyContours[34][1]),
            (keyContours[34][0] + keyContours[34][2], keyContours[0][1]),
            (keyContours[34][0] + keyContours[34][2],
             keyContours[34][1] + keyContours[34][3]),
            (keyContours[34][0], keyContours[34][1] + keyContours[0][3])
        ],
        "B": [
            (keyContours[35][0], keyContours[36][1]),
            (keyContours[35][0] + keyContours[36][2], keyContours[0][1]),
            (keyContours[35][0] + keyContours[36][2],
             keyContours[35][1] + keyContours[36][3]),
            (keyContours[35][0], keyContours[36][1] + keyContours[0][3])
        ],
        "N": [
            (keyContours[36][0], keyContours[36][1]),
            (keyContours[36][0] + keyContours[36][2], keyContours[0][1]),
            (keyContours[36][0] + keyContours[36][2],
             keyContours[36][1] + keyContours[36][3]),
            (keyContours[36][0], keyContours[36][1] + keyContours[0][3])
        ],
        "M": [
            (keyContours[37][0], keyContours[37][1]),
            (keyContours[37][0] + keyContours[37][2], keyContours[0][1]),
            (keyContours[37][0] + keyContours[37][2],
             keyContours[37][1] + keyContours[37][3]),
            (keyContours[37][0], keyContours[37][1] + keyContours[0][3])
        ],
        ",": [
            (keyContours[38][0], keyContours[38][1]),
            (keyContours[38][0] + keyContours[38][2], keyContours[0][1]),
            (keyContours[38][0] + keyContours[38][2],
             keyContours[38][1] + keyContours[38][3]),
            (keyContours[38][0], keyContours[38][1] + keyContours[0][3])
        ],
        ".": [
            (keyContours[39][0], keyContours[39][1]),
            (keyContours[39][0] + keyContours[39][2], keyContours[0][1]),
            (keyContours[39][0] + keyContours[39][2],
             keyContours[39][1] + keyContours[39][3]),
            (keyContours[39][0], keyContours[39][1] + keyContours[0][3])
        ],
        "SPACE": [
            (keyContours[40][0], keyContours[40][1]),
            (keyContours[40][0] + keyContours[40][2], keyContours[0][1]),
            (keyContours[40][0] + keyContours[40][2],
             keyContours[40][1] + keyContours[40][3]),
            (keyContours[40][0], keyContours[40][1] + keyContours[0][3])
        ],

    }

    print(keyContours)
    print(' ')
    print(keyCoords)

    # Draws all countours, including numbers/words/keyboard
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    #cv2.imshow('image', img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite('images/highlighted_keyboard.png', img)

    return keyCoords
