import cv2
import numpy as np

hand_sampler_size, sampler_x1, sampler_x2, sampler_y1, sampler_y2 = 0, 0, 0, 0, 0

def sample_hand_pixels(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv_frame[:,:,0] = cv2.equalizeHist(hsv_frame[:,:,0])
    #hsv_frame[:,:,1] = cv2.equalizeHist(hsv_frame[:,:,1])
    #hsv_frame[:,:,2] = cv2.equalizeHist(hsv_frame[:,:,2])

    region_of_interest = np.zeros(
        [hand_sampler_size, hand_sampler_size, 3], dtype=hsv_frame.dtype)
    region_of_interest[0:hand_sampler_size,
                       0:hand_sampler_size] = hsv_frame[sampler_y1:sampler_y2, sampler_x1:sampler_x2]

    return region_of_interest


def calculate_hand_histogram(pixels):
    aux_hist = cv2.calcHist([pixels], [0, 1], None, [
                            180, 256], [0, 180, 0, 256])
    return cv2.normalize(aux_hist, aux_hist, 0, 255, cv2.NORM_MINMAX)


def calculate_hand_mask(frame, hand_hist):
    back_project = cv2.calcBackProject(
        [frame], [0, 1], hand_hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(back_project, -1, disc, back_project)
    cv2.imshow('back', back_project)

    _, mask = cv2.threshold(back_project, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('mask', mask)

    blured = cv2.blur(mask, (2, 2))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closed = cv2.morphologyEx(blured, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilated = cv2.dilate(closed, dilate_kernel, iterations=1)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(blured, erode_kernel, iterations=1)

    cv2.imshow('closed', closed)
    cv2.imshow('dilated', dilated)
    cv2.imshow('eroded', eroded)
    return eroded


def calculate_mask_contours(mask):
    contours, _ = cv2.findContours(mask, 1, 2)
    return contours


def calculate_center(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return cx, cy


def find_farthest_defect(defects, contour, center):
    if defects is None:
        return None

    farthest_distance = 0
    farthest_point = None

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        distance = np.linalg.norm(center - start)
        if distance > farthest_distance:
            farthest_distance = distance
            farthest_point = start

        distance = np.linalg.norm(center - end)
        if distance > farthest_distance:
            farthest_distance = distance
            farthest_point = end

        distance = np.linalg.norm(center - far)
        if distance > farthest_distance:
            farthest_distance = distance
            farthest_point = far

    return farthest_point


def get_contour_tip(frame, contour):
    cx, cy = calculate_center(contour)

    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 2, (0, 0, 0), -1)

    hull = cv2.convexHull(contour, returnPoints=False)
    cv2.drawContours(frame, contour[hull], -1, (0, 255, 255), 3)

    try:
        defects = cv2.convexityDefects(contour, hull)
        return find_farthest_defect(defects, contour, np.array([cx, cy]))
    except cv2.error:
        return None


def calculate_hand_contours(frame, hand_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv_frame[:,:,0] = cv2.equalizeHist(hsv_frame[:,:,0])
    #hsv_frame[:,:,1] = cv2.equalizeHist(hsv_frame[:,:,1])
    #hsv_frame[:,:,1] = cv2.equalizeHist(hsv_frame[:,:,2])

    mask = calculate_hand_mask(hsv_frame, hand_hist)

    return calculate_mask_contours(mask)


def find_fingertips(frame, contours):
    point_1 = None
    point_2 = None

    contour_1 = None
    contour_2 = None

    area_1 = 0
    area_2 = 0

    for contour in contours:
        aux_area = cv2.contourArea(contour)

        if contour_1 is None or aux_area > area_1:
            contour_2 = contour_1
            contour_1 = contour
            area_2 = area_1
            area_1 = aux_area
            continue

        if contour_2 is None or aux_area > area_2:
            contour_2 = contour
            area_2 = aux_area

    if contour_1 is not None:
        point_1 = get_contour_tip(frame, contour_1)

    if contour_2 is not None:
        point_2 = get_contour_tip(frame, contour_2)

    if point_1 is not None:
        cv2.circle(frame, point_1, 5, [0, 0, 255], -1)
    if point_2 is not None:
        cv2.circle(frame, point_2, 5, [0, 0, 255], -1)

    return point_1, point_2


def calculate_sampler_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _= cv2.findContours(mask, 1, 2)

    max_area = 0
    max_contour = None
    for contour in contours:
        aux_area = cv2.contourArea(contour)

        if max_contour is None or aux_area > max_area:
            max_contour = contour
            max_area = aux_area

    sampler_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(sampler_mask, [max_contour], -1, (255), -1)

    cv2.imshow('aux', sampler_mask)

    return sampler_mask


def get_hand_pixels(frame, sampler_mask):
    masked = cv2.bitwise_and(frame, frame, mask=sampler_mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    _, hand_mask = cv2.threshold(gray_eq, 235, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hand_pixels = cv2.bitwise_and(masked, masked, mask=hand_mask)

    cv2.imshow('gray', gray_eq)
    cv2.imshow('frame', hand_pixels)

    return hand_pixels, hand_mask


def hand_sampler(cap):
    sampler_mask = None
    hand_pixels = None
    hand_mask = None
    hand_hist = None

    step = 0

    while(True):
        _, frame = cap.read()

        if step == 3:
            hand_hsv = cv2.cvtColor(hand_pixels, cv2.COLOR_BGR2HSV)
            hand_hist = cv2.calcHist([hand_hsv], [0, 1], hand_mask, [180, 256], [0, 180, 0, 256])

            contours = calculate_hand_contours(frame, hand_hist)
            cv2.drawContours(frame, contours, -1, (255, 255, 0), 3)

            point_1, point_2 = find_fingertips(frame, contours)

        cv2.imshow('frame', frame)

        if step == 1:
            sampler_mask = calculate_sampler_mask(frame)
        if step == 2:
            hand_pixels, hand_mask = get_hand_pixels(frame, sampler_mask)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or step > 3:
            break
        if key & 0xFF == ord('s') and step < 4:
            step += 1
        if key & 0xFF == ord('c') and step > 0:
            step -= 1

    # Destroy all the windows
    cv2.destroyAllWindows()

    return hand_hist


def test():
    cap = cv2.VideoCapture(1)

    hand_sampler(cap)

    cap.release()

test()