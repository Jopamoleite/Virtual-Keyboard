# Import essential libraries
import requests
import argparse
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt


def plot_img_histogram(frame):
    imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histr = cv2.calcHist([imgGrey], [0], None, [256], [0, 256])
    plt.plot(histr)
    plt.show()
    return

# thresh 127
# maxval 255

# TODO: use dynamic thresholding


def binaryThresholding(frame):
    imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(imgGrey, 127, 255, cv2.THRESH_BINARY)
    # plot img histogram to analyse the thrshold to use
    return thresh1

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

        # ensure the width, height, and area are all neither too small
        # nor too big
        keepWidth = w > 0 and w < 1500
        keepHeight = h > 0 and h < 1500
        keepArea = area > 0 and area < 1500

    # ensure the connected component we are examining passes all
        # three test
    #  construct a mask for the current connected component and
        # then take the bitwise OR with the mask
        if all((keepWidth, keepHeight, keepArea)):
            print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            print(componentMask)
            mask = cv2.bitwise_or(mask, componentMask)

    # show our output image and connected component mask
    cv2.imshow("Connected Component", mask)
    cv2.waitKey(0)

    return


def detect_marker(frame):
    binaryThresholding(frame)
    return


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
    threshFrame = binaryThresholding(frame)

    histr = cv2.calcHist([frame], [0], None, [256], [0, 256])
    # Display the resulting frame
    # cv2.imshow('frame', frame) ##REVERT o see colored image
    cv2.imshow('frame', threshFrame)

    # Histogram hand in the frame

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

blobDetection(threshFrame)
# plot_img_histogram(lastFrame)
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
