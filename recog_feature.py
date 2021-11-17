import numpy as np
from prep import prep
import cv2 as cv
from matplotlib import pyplot as plt

keyCoords = prep()
keyRealCoords = {}


# Initiate SIFT detector

img1 = cv.imread('images/keyboard.png', 0)  # queryImage
#sift = cv.SIFT_create()
#fast = cv.FastFeatureDetector_create()
orb = cv.ORB_create()
#kp1, des1 = sift.detectAndCompute(img1, None)
kp1 = orb.detect(img1, None)
kp1, des1 = orb.compute(img1, kp1)


# surf = cv.surf(400)


def get_key_being_pressed(x, y):

    ret = False
    for key, value in keyRealCoords.items():

        # print(value)
        if key == "BORDER":
            break

        if (x > value.item(6) and x < value.item(2) and y > value.item(3) and y < value.item(7)):
            ret = key

    print(ret)


def feature_detection(frame):

    MIN_MATCH_COUNT = 10

    img2 = frame

    # find the keypoints and descriptors with SIFT
    #kp2, des2 = sift.detectAndCompute(img2, None)
    kp2 = orb.detect(img2, None)
    kp2, des2 = orb.compute(img2, kp2)
    # kp2, des2 = surf.detectAndCompute(img2, None)

    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #search_params = dict(checks=50)
    #flann = cv.FlannBasedMatcher(index_params, search_params)

    #matches = flann.knnMatch(des1, des2, k=2)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
    matches = bf.match(des1, des2)
# Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # create BFMatcher object
    #bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
    #matches = bf.knnMatch(des1, des2, k=20)
# Sort them in the order of their distance.
    #matches = sorted(matches, key=lambda x: x.distance)
    # store all the good matches as per Lowe's ratio test.

    # store all the good matches as per Lowe's ratio test.

    for match in matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt

    good = []

    # for m in matches:
    #    if m.distance < 75:
    #        good.append(m)
#
    # if len(good) > MIN_MATCH_COUNT:
    #    src_pts = np.float32(
    #        [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #    dst_pts = np.float32(
    #        [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
    #    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#
    #    if np.array(M).size == 1:
    #        return
#
    #    for key, value in keyCoords.items():
    #        pts = np.float32(np.array(value)).reshape(-1, 1, 2)
    #        dst = cv.perspectiveTransform(pts, M)
    #        img2 = cv.polylines(img2, [np.int32(dst)],
    #                            True, 255, 3, cv.LINE_AA)
    #        keyRealCoords[key] = np.int32(dst)

    # print(keyRealCoords)
    img2 = cv.circle(img2, (200, 200), 2, (0, 0, 255), 2)
    img2 = cv.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
    cv.imshow("features", img2)


# MAIN
# Replace the below URL with your own. Droidcam keep '/video'
url = "http://192.168.1.24:4747/video"
vid = cv.VideoCapture(url)
# detect_key(keyCoords, 1, 1)

while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    #b = cv.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    feature_detection(frame)
    # cv.imshow("features", frame)
    # Detect keyboard
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    get_key_being_pressed(200, 200)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
