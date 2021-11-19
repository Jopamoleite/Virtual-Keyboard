import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from playsound import playsound

MIN_FRAMES_REQUIRED = 10
# GLOBAL VARS
keyRealCoords = {}
frameCount = 0
frameCount2 = 0
frameCountDetector = 0

# Initiate SIFT detector

img1 = cv.imread('images/preppedKeyboard.png', 0)  # queryImage
sift = cv.SIFT_create()
orb = cv.ORB_create()
fast = cv.FastFeatureDetector_create()

kp1, des1 = sift.detectAndCompute(img1, None)

# kp1 = sift.detect(img1, None)
# kp1, des1 = sift.compute(img1, kp1)


# surf = cv.surf(400)

def make_sound():
    playsound('note.mp3', block=False)


def get_key_being_pressed(x, y, frame, frameCount, keyRealCoords):

    ret = False
    for key, value in keyRealCoords.items():
        # print(value)
        if key == "BORDER":
            break

        if (x > value.item(6) and x < value.item(2) and y > value.item(3) and y < value.item(7)):
            ret = key
            break
    img1 = cv.circle(frame, (x, y), 2, (0, 0, 255),
                     2)  # TODO remove this
    if ret != False:
        frameCount = frameCount + 1
        if frameCount == MIN_FRAMES_REQUIRED:
            make_sound()
        if frameCount > MIN_FRAMES_REQUIRED:
            img2 = cv.polylines(
                img1, [keyRealCoords[ret]], True, 255, 3, cv.LINE_AA)
            return img2, frameCount, ret
        return frame, frameCount, ""

    frameCount = 0
    return frame, frameCount, ""

def draw_key(x, y, frame, key):
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, key, (x, y + 50), font,
                2, (0, 255, 0), 2, cv.LINE_AA)



def feature_detection(frame, keyCoords, oldKeyRealCoords):

    MIN_MATCH_COUNT = 10

    img2 = frame
    M = None
    kp2, des2 = sift.detectAndCompute(img2, None)

    # kp2 = sift.detect(img2, None)
    # kp2, des2 = sift.compute(img2, kp2)

    #bf = cv.BFMatcher()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return oldKeyRealCoords
    # matches = bf.knnMatch(des1, des2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        if M is None:
            return None

        for key, value in keyCoords.items():
            pts = np.float32(np.array(value)).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            # img2 = cv.polylines(img2, [np.int32(dst)],
            # True, 255, 3, cv.LINE_AA)
            keyRealCoords[key] = np.int32(dst)

        return keyRealCoords

    return oldKeyRealCoords
