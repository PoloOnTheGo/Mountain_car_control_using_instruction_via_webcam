# Importing the libraries
import cv2
import numpy as np
from webcam_gesture_recognition import folder_management as fm

mode = ''


def printing_count_details_onscreen(frame):
    cv2.putText(frame, "MODE : " + mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "left(0): " + fm.get_image_count('0'), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "no push(1): " + fm.get_image_count('1'), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "right(2) : " + fm.get_image_count('2'), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)


def roi_generation(frame):
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI (The increment/decrement by 1 is to compensate for the bounding box)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))
    return roi


def preprocess_image(frame):
    # -------PRIOR CLEANUP OF BACKGROUND WITH THRESHOLD-------#
    # th_lb,th_ub = getThreshTrackBar()
    th_lb, th_ub = (141, 255)
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, th_lb, th_ub, cv2.THRESH_BINARY_INV)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # -------FINAL EXTRACTION OF HAND FROM IMAGE---------#
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((21, 21), np.uint8)
    # l_b,u_b = getHSVTrackbarValues()
    l_b, u_b = (np.array([0, 58, 71]), np.array([83, 225, 255]))
    mask = cv2.inRange(frame, l_b, u_b)
    mask = cv2.dilate(mask, kernel)
    cv2.imshow("mask", mask)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    cv2.imshow("processed", frame)

    # -------prepare for processing-------#
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (15, 15), 0)
    frame = cv2.resize(frame, (64, 64))
    return frame


def process_video(cap):
    ret, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Printing the count in each set to the screen
    printing_count_details_onscreen(frame)

    # creating region of interest
    processed = roi_generation(frame)
    
    # processing roi
    # processed = preprocess_image(target_frame)

    # show video feed
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", processed)

    return processed


def save_processed_img(processed, interrupt):
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(fm.get_image_name('0'), processed)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(fm.get_image_name('1'), processed)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(fm.get_image_name('2'), processed)


def capture_process_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        processed = process_video(cap)
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:  # esc key
            break
        save_processed_img(processed, interrupt)
    cap.release()
    cv2.destroyAllWindows()


def collect_data(data_mode):
    global mode
    mode = data_mode
    # Create the directory structure
    fm.create_directories()

    # get working directory
    fm.set_directory(mode)

    # capture video feed
    capture_process_video()


collect_data('train')
collect_data('test')
collect_data('validation')
