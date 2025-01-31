#!/usr/bin/python
import cv2
import numpy as np
import pickle
#from hand_tracker_final import sample_hand_pixels, calculate_hand_histogram, calculate_hand_mask, calculate_mask_contours, find_farthest_defect, get_contour_tip
from recog_markerless_final import feature_detection, get_key_being_pressed, draw_key
from recog_markerBased_final import detect_marker, calculate_homography
from hand_tracker_module_final import hand_sampler, calculate_hand_contours, find_fingertips

URL = "http://192.168.1.242:4747/video"
MARKER_TYPE = 1  # 0 - Marker Based / 1 - Markerless

# Finger tracking variables
hand_pixels = None
hand_hist = None
hand_sampler_size = 0
sampler_x1, sampler_x2, sampler_y1, sampler_y2 = 0, 0, 0, 0

# Key press related variables
database = []
keyCoords = {}
frameCount = 0
frameCount2 = 0
frameCountDetector = 0

# Default placeholder values for fingers, used for testing
fingerX1 = 600
fingerY1 = 200
fingerX2 = 100
fingerY2 = 300

vid = cv2.VideoCapture(1)
# first run preparation program calibrating the hand color
with open('prep', 'rb') as prep_file:
    database = pickle.load(prep_file)
 
# hand prep
hand_hist = hand_sampler(vid)

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
    hand_contours = calculate_hand_contours(frame, hand_hist)
    finger1, finger2 = find_fingertips(frame, hand_contours)

    if finger1 is not None:
        fingerX1 = finger1[0]
        fingerY1 = finger1[1]
    if finger2 is not None:
        fingerX2 = finger2[0]
        fingerY2 = finger2[1]

    # Detect key being pressed
    if keyCoords != {} and keyCoords is not None:
        frame, frameCount, key = get_key_being_pressed(
            fingerX1, fingerY1, frame, frameCount, keyCoords)
        frame, frameCount2, key2 = get_key_being_pressed(
            fingerX2, fingerY2, frame, frameCount2, keyCoords)

        # Handle pressed keys and deal with edge cases (SHIFT)
        if not (len(key) == 0 and len(key2) == 0):
            if len(key) == 0 or len(key2) == 0:
                draw_key(0, 0, frame, key if len(key) > len(key2) else key2)
            else:
                if key != key2:
                    if key == "SHIFT" and len(key2) == 1 and key2.isalpha():
                        draw_key(0, 0, frame, key2.lower())
                    else:
                        if key2 == "SHIFT" and len(key) == 1 and key.isalpha():
                            draw_key(0, 0, frame, key.lower())
                        else:
                            draw_key(0, 0, frame, key)
                            draw_key(0, 75, frame, key2)


    cv2.imshow("features", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
