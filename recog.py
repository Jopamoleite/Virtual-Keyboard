# Import essential libraries
from typing import Counter
import requests
import argparse
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

from prep import prep


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

# thresh 127
# maxval 255

# TODO: use dynamic thresholding


def binaryThresholding(frame):
    imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # This is not sufcient as we would need to make hard math reading the images histogram to adapt to light
    # ret, th1 = cv2.threshold(imgGrey, 127, 255, cv2.THRESH_BINARY)

    # this method works to on all ilumination conditions, but does not conserve blacks it works fine as a line detector
    th1 = cv2.adaptiveThreshold(
        imgGrey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)

    # OTSU method works on bad and good light contions but preserves the blacks from the marker, making it easit for the detection of the keys
    blur = cv2.GaussianBlur(imgGrey, (5, 5), 0)
    ret3, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plot img histogram to analyse the thrshold to use
    return th2

# binaryImg - black and white image to find blobs
# connectivity - connectivity can be 4 or 8
# objectColor - image of the objects can be 0 if black or 255 if white
# returns - array with blobs detected


def blobDetection(binaryImg):

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--connectivity", type=int, default=4,
                    help="connectivity for connected component analysis")
    args = vars(ap.parse_args())

    # since the algorithm is detecting only white blobs
    # reverse = cv2.bitwise_not(binaryImg)
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
        # - if the a bug white rectangle is detected, means we probably found the paper sheet white border containing teh keyboard
        # - loop again on the numLabels, and check if the coordinates of the labels found are inside the white rectangle and if the area is smaller
        # - If so, that means a key was found
        # Store that key info on an array so that it could later be matched with the coordinates stored of the prep program

        keepWidth = w > 0 and w < 20000
        keepHeight = h > 50 and h < 500
        keepArea = area > 10000 and area < 100000

        c = 0
        # TODO needs refinement
        if all((keepWidth, keepArea)):
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

                if all((insideHeight, insideWidth, insideArea)):
                    c = c+1
                    #print("[INFO] keeping connected component '{}'".format(key))
                    #componentMask = (labels == key).astype("uint8") * 255
                    #mask = cv2.bitwise_or(mask, componentMask)

                if c > 20:
                    print("[INFO] keeping connected component '{}'".format(i))
                    componentMask = (labels == i).astype("uint8") * 255
                    mask = cv2.bitwise_or(mask, componentMask)

    return mask


def detect_contours(mask, frame):

    # show our output image and connected component mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # remove outer countours
    if(contours):
        contour = contours[0]
        for i in contours:
            if cv2.contourArea(i, True) > cv2.contourArea(contour, True):
                contour = i

    # probably get the rectabgle

    x = 1000000000
    y = 1000000000
    w = 0
    h = 0
    # get points
    for i in contour:
        if i[0][0] < x:
            x = i[0][0]
        if i[0][1] < y:
            y = i[0][1]
        if i[0][0] > w:
            w = i[0][0]
        if i[0][1] > h:
            h = i[0][1]

    # marker coords limit
    markerLimitPoints = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    print(markerLimitPoints)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow("Connected Component", frame)
    # cv2.waitKey(0)


def detect_marker(frame):
    th = binaryThresholding(frame)
    mask = blobDetection(th)
    detect_contours(mask, frame)
    return

# MAIN


keyCoords = prep()
# Replace the below URL with your own. Droidcam keep '/video'
url = "http://192.168.1.24:4747/video"


# While loop to continuously fetching data from the Url
vid = cv2.VideoCapture(url)
lastFrame = ' '

while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    lastFrame = frame

    # Detect keyboard
    detect_marker(frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# plot_img_histogram(lastrame)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
