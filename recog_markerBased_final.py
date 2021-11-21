# Import essential libraries
from typing import Counter
from numpy.core.defchararray import equal
from numpy.core.fromnumeric import size
import requests
import pickle
import argparse
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from playsound import playsound
import math

MIN_FRAMES_REQUIRED = 10

with open('prep', 'rb') as prep_file:
    keyCoords = pickle.load(prep_file)

keyRealCoords = {}
frameCount = 0

# function to plot image histogram, was utilitary to choose binarization threshold value


def plot_img_histogram(frame):
    imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # smoothing histogram
    kernel = np.ones((5, 5), np.float32)/25
    dst = cv2.filter2D(imgGrey, -1, kernel)
    cv2.imshow('blured', dst)
    histr = cv2.calcHist([dst], [0], None, [256], [0, 256])
    plt.plot(histr)
    plt.show()
    return


def binaryThresholding(frame):
    imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # OTSU method works on bad and good light contions but preserves the blacks from the marker, making it easier for the detection of the keys
    blur = cv2.GaussianBlur(imgGrey, (5, 5), 0)
    ret3, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return th2


def blobDetection(binaryImg):

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--connectivity", type=int, default=4,
                    help="connectivity for connected component analysis")
    args = vars(ap.parse_args())

    output = cv2.connectedComponentsWithStats(
        binaryImg, args["connectivity"], cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    mask = np.zeros(binaryImg.shape, dtype="uint8")

    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        # heuristic to detect the key:
        # - if the a big white rectangle is detected, means we probably found the paper sheet white border containing the keyboard
        # - loop again on the numLabels, and check if the coordinates of the labels found are inside the white rectangle and if the area is smaller
        # - If so, that means a key was found
        # Store that key into on an array so that it could later be matched with the coordinates stored of the prep program

        keepWidth = w > 0 and w < 20000
        keepHeight = h > 50 and h < 500
        keepArea = area > 10000 and area < 100000

        # used to count keys, small squares
        c = 0

        if all((keepWidth, keepArea)):
            # check if there are small rectangles inside a bigger one
            for key in range(0, numLabels):

                xKey = stats[key, cv2.CC_STAT_LEFT]
                yKey = stats[key, cv2.CC_STAT_TOP]
                wKey = stats[key, cv2.CC_STAT_WIDTH]
                hKey = stats[key, cv2.CC_STAT_HEIGHT]
                areaKey = stats[key, cv2.CC_STAT_AREA]
                (cXKey, cYKey) = centroids[key]

                insideWidth = xKey >= x and xKey <= x + w
                insideHeight = yKey >= y and yKey <= y + h
                insideArea = areaKey < area

                # if inside the bigger square increment
                if all((insideHeight, insideWidth, insideArea)):
                    c = c+1

                # if the square has more than 20 squares inside it is a good candidate to be the marker border
                if c > 20:
                    componentMask = (labels == i).astype("uint8") * 255
                    mask = cv2.bitwise_or(mask, componentMask)

    return mask


def detect_contours(mask, frame):
    markerPoints = []
    # show our output image and connected component mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   # remove outer countours
    rect = []
    if(contours):

        font = cv2.FONT_HERSHEY_COMPLEX
        for cnt in contours:

            approx = cv2.approxPolyDP(
                cnt, 0.009 * cv2.arcLength(cnt, True), True)

            # draws boundary of contours.
            cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)

            # Used to flatted the array containing
            # the co-ordinates of the vertices.
            n = approx.ravel()
            i = 0

            points = []

            for j in n:
                if(i % 2 == 0):
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y)

                    cv2.putText(frame, string, (x, y), font, 0.5, (0, 255, 0))

                    points.append((x, y))
                i = i + 1
            rect.append(points)

        # select rectangle with the smallest area
        markerPoints = rect[0]
        for i in rect:
            # calculate rect area
            currArea = cv2.contourArea(np.array(markerPoints))
            newArea = cv2.contourArea(np.array(i))

            if newArea < currArea:
                markerPoints = i
    frame = cv2.circle(frame, (200, 200), 2, (0, 0, 255), 2)
    return markerPoints


def detect_marker(frame):
    markerPoints = []
    th = binaryThresholding(frame)
    mask = blobDetection(th)
    markerPoints = detect_contours(mask, frame)
    return markerPoints


def calculate_homography(keyCoords, markerPoints):

    if markerPoints == None:
        return

    # get homography matrix
    M, mask = cv2.findHomography(
        np.array(keyCoords["BORDER"]), np.array(markerPoints), cv2.RANSAC, 5.0)

    # for each key calculate each coordinates on the image using the homography matrix
    for key, value in keyCoords.items():
        pts = np.float32(np.array(value)).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        keyRealCoords[key] = np.int32(dst)

    return keyRealCoords


def make_sound():
    playsound('note.mp3', block=False)


def get_key_being_pressed(x, y, frame, frameCount):

    ret = False
    for key, value in keyRealCoords.items():

        if key == "BORDER":
            break

        if (x > value.item(6) and x < value.item(2) and y > value.item(3) and y < value.item(7)):
            ret = key
    img1 = cv2.circle(frame, (200, 200), 2, (0, 0, 255), 2)  # TODO remove this

    if ret != False:
        frameCount = frameCount + 1
        if frameCount == MIN_FRAMES_REQUIRED:
            make_sound()
        if frameCount > MIN_FRAMES_REQUIRED:
            img2 = cv2.polylines(
                img1, [keyRealCoords[ret]], True, 255, 3, cv2.LINE_AA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, ret, (0, 50), font,
                        2, (0, 255, 0), 2, cv2.LINE_AA)
            return img2, frameCount
        return frame, frameCount

    frameCount = 0
    return frame, frameCount
