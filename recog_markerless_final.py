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

def make_sound():
    playsound('note.mp3', block=False)


# Handles hovering finger over key
def get_key_being_pressed(x, y, frame, frameCount, keyRealCoords):

    ret = False
    # Checks if finger position is inside a key
    for key, value in keyRealCoords.items():
        if key == "BORDER":
            break

        if (x > value.item(6) and x < value.item(2) and y > value.item(3) and y < value.item(7)):
            ret = key
            break

    # Draws a circle on the finger pixel
    img1 = cv.circle(frame, (x, y), 2, (0, 0, 255),
                     2)

    # Checks how many frames the finger has been inside this key, if over MIN_FRAMES_REQUIRED it displays an audiovisual queue
    if ret != False:
        frameCount = frameCount + 1
        if frameCount == MIN_FRAMES_REQUIRED:
            make_sound()
        if frameCount > MIN_FRAMES_REQUIRED:
            img2 = cv.polylines(
                img1, [keyRealCoords[ret]], True, 255, 3, cv.LINE_AA)
            return img2, frameCount, ret
        return frame, frameCount, ""

    # If the finger wasn't hovering any key, we reset the frame count
    frameCount -= 1
    return frame, frameCount, ""

# Draws the key on position (x, y+50) on the desired frame
def draw_key(x, y, frame, key):
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, key, (x, y + 50), font,
                2, (0, 255, 0), 2, cv.LINE_AA)

# Detects features on the current frame and returns the pixel coordinates corresponding to the keys in the current frame
def feature_detection(frame, keyCoords, oldKeyRealCoords):

    # Minimum number of matches that must be found to establish homography, must be atleast 4
    MIN_MATCH_COUNT = 10

    img2 = frame
    M = None
    # Use SIFT to both detect keypoints and compute the descriptors
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use FLANN to match the original image's descriptors with the current frame
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return oldKeyRealCoords
    good = []

    # Use Lowe's ratio to test if matches are good
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # If there are enough good matches, we'll calculate the homography matrix and find the coordinates of the keys we need
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
            # Highlights the key contours
            #img2 = cv.polylines(img2, [np.int32(dst)],
            #    True, 255, 3, cv.LINE_AA)
            #cv.imshow("board", img2)
            keyRealCoords[key] = np.int32(dst)

        return keyRealCoords

    return oldKeyRealCoords
