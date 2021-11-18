#!/usr/bin/python
import cv2
import numpy as np
from prep import prep
# from hand_tracker import sample_hand_pixels, calculate_hand_histogram, calculate_hand_mask, calculate_mask_contours, find_farthest_defect, get_contour_tip
from recog_markerless_final import feature_detection, get_key_being_pressed
from recog_markerBased_final import detect_marker, calculate_homography

URL = "http://192.168.1.24:4747/video"
MARKER_TYPE = 0  # 0 - Marker Based / 1 - Markerless

hand_pixels = None
hand_hist = None
hand_sampler_size = 0
sampler_x1, sampler_x2, sampler_y1, sampler_y2 = 0, 0, 0, 0

database = []
keyCoords = {}
frameCount = 0
frameCount2 = 0
frameCountDetector = 0

vid = cv2.VideoCapture(URL)
# first run preparation program calibrating the hand color
database = prep()


# keyboard detection


while(True):

    ret, frame = vid.read()

    # detect marker
    if MARKER_TYPE == 0:
        markerPoints = detect_marker(frame)

        if len(markerPoints) == 4:
            keyCoords = calculate_homography(database, markerPoints)
    else:
        frameCountDetector = frameCountDetector + 1
        if frameCountDetector == 1:
            keyCoords = feature_detection(frame, database, keyCoords)
            frameCountDetector = 0

    # detect finger

    if keyCoords != {}:
        frame, frameCount = get_key_being_pressed(
            200, 200, frame, frameCount, keyCoords)
        frame, frameCount2 = get_key_being_pressed(
            100, 300, frame, frameCount2, keyCoords)

    cv2.imshow("features", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
