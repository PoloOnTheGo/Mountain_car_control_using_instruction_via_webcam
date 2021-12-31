import numpy as np
import tensorflow as tf
import cv2
import pyautogui
from keras_preprocessing.image import ImageDataGenerator

from webcam_gesture_recognition import image_processing_util as ipu

'''
-------PROGRAM STATES-------
0 - LEFT
1 - NO PUSH 
2 - RIGHT
------------END-------------
'''

'''
--------Pseudo code---------
1. load_model
2. video_processing
    a. open video 
    b. image_processing
    c. interpreting_result_using_model
    d. print_output_on_screen
    e. close_video
------------end------------- 
'''


def load_model():
    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    cnn = tf.keras.models.model_from_json(model_json)
    # load weights into new model
    cnn.load_weights("model-bw.h5")
    print("Loaded model from disk")
    return cnn


def interpreting_result_using_model(actual_frame, input_frame):
    prev_result = -1
    result_cnt = 0
    decision_threshold = 1
    modified_frame = input_frame.reshape(1, 64, 64, 3)
    output = model.predict(modified_frame)[0]
    result = list(output).index(max(output))
    if result == prev_result:
        result_cnt += 1
    else:
        result_cnt = 1
    prev_result = result
    if result_cnt == decision_threshold:
        print_output_on_screen(actual_frame, str(result))
        result_cnt = 0

    # if result == prev_result:
    #     result_cnt += 1
    # else:
    #     result_cnt = 1
    # prev_result = result
    # if result_cnt == decision_threshold:
    #     executeTask(result)
    #     result_cnt = 0

    # return output_action


def print_output_on_screen(frame, action):
    cv2.putText(frame, "OUTPUT : ", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    if action == '0':
        action_text = "LEFT"
    elif action == '1':
        action_text = "NO PUSH"
    elif action == '2':
        action_text = "RIGHT"
    else:
        action_text = "UNIDENTIFIED"
    print(action_text)
    cv2.putText(frame, "ACTION : " + action_text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)


def video_processing():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        actual_frame, processed_frame = ipu.process_video(cap, 'prediction', '')
        interpreting_result_using_model(actual_frame, processed_frame)
        # cv2.imshow("cam", processed_frame)
        # # cv2.waitKey(10)
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:  # esc key
            break
        # print_output_on_screen(processed_frame, output)
    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------PROGRAM START-----------------------------------------#
model = load_model()
video_processing()


# def empty(e):
#     pass
#
#
# def createHSVTrackBar():
#     cv2.namedWindow("trackbar")
#     cv2.createTrackbar("lh", "trackbar", 0, 255, empty)
#     cv2.createTrackbar("uh", "trackbar", 0, 255, empty)
#     cv2.createTrackbar("ls", "trackbar", 0, 255, empty)
#     cv2.createTrackbar("us", "trackbar", 0, 255, empty)
#     cv2.createTrackbar("lv", "trackbar", 0, 255, empty)
#     cv2.createTrackbar("uv", "trackbar", 0, 255, empty)
#
#
# def getHSVTrackbarValues():
#     lh = cv2.getTrackbarPos("lh", "trackbar")
#     uh = cv2.getTrackbarPos("uh", "trackbar")
#     ls = cv2.getTrackbarPos("ls", "trackbar")
#     us = cv2.getTrackbarPos("us", "trackbar")
#     lv = cv2.getTrackbarPos("lv", "trackbar")
#     uv = cv2.getTrackbarPos("uv", "trackbar")
#     l_b = np.array([lh, ls, lv])
#     u_b = np.array([uh, us, uv])
#     return l_b, u_b
#
#
# def createThreshTrackBar():
#     cv2.namedWindow("trackbar")
#     cv2.createTrackbar("lb", "trackbar", 0, 255, empty)
#     cv2.createTrackbar("ub", "trackbar", 0, 255, empty)
#
#
# def getThreshTrackBar():
#     lb = cv2.getTrackbarPos("lb", "trackbar")
#     ub = cv2.getTrackbarPos("ub", "trackbar")
#     return lb, ub


def executeTask(task):
    '''--------------------------
    0 - CLOSED PALM
    1 - DOWN
    2 - LEFT
    3 - RIGHT
    4 - UP
    5 - NONE
    6 - OPEN PALM
    7 - THUMBS DOWN
    --------------------------'''
    global program_state
    task = int(task)
    if task == 6:
        pyautogui.press("space")
    elif task == 1:
        pyautogui.press("down")

# def preprocess_image(frame):
#     # -------PRIOR CLEANUP OF BACKGROUND WITH THRESHOLD-------#
#     th_lb, th_ub = (180, 255)
#     # th_lb,th_ub = getThreshTrackBar()
#     mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(mask, th_lb, th_ub, cv2.THRESH_BINARY_INV)
#     frame = cv2.bitwise_and(frame, frame, mask=mask)
#
#     # -------FINAL EXTRACTION OF HAND FROM IMAGE---------#
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     kernel = np.ones((21, 21), np.uint8)
#     l_b, u_b = (np.array([0, 63, 145]), np.array([248, 97, 255]))
#     # l_b, u_b = getHSVTrackbarValues()
#     mask = cv2.inRange(frame, l_b, u_b)
#     mask = cv2.dilate(mask, kernel)
#     frame = cv2.bitwise_and(frame, frame, mask=mask)
#     cv2.imshow("processed", frame)
#     frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
#
#     # -------prepare for processing-------#
#     # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow("frame", frame)
#     frame = cv2.GaussianBlur(frame, (15, 15), 0)
#     frame = cv2.resize(frame, (64, 64))
#     cv2.imshow("actual", frame)
#     frame = np.reshape(frame, (64, 64, 3))
#     frame = np.expand_dims(frame, axis=0)
#
#     return frame


# def process_frame(target_frame):
#     input = preprocess_image(target_frame)
#     output = model.predict(input)[0]
#     print(output)
#     class_result = list(output).index(max(output))
#     return str(class_result)
#
#
# def background_process():
#     global program_state
#     prev_result = -1
#     result_cnt = 0
#     decision_threshold = 1
#
#     while cap.isOpened():
#
#         # ---------EXTRACT FRAME OF INTEREST---------#
#         ret, frame = cap.read()
#         frame = cv2.flip(frame, 1)
#         # pt1 = (300,180)
#         # pt2 = (550,420)
#         # frame = cv2.rectangle(frame,pt1,pt2,(255,0,0),2)
#         # target_frame = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:]
#
#         x1 = int(0.5 * frame.shape[1])
#         y1 = 10
#         x2 = frame.shape[1] - 10
#         y2 = int(0.5 * frame.shape[1])
#         # Drawing the ROI (The increment/decrement by 1 is to compensate for the bounding box)
#         cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
#         # Extracting the ROI
#         target_frame = frame[y1:y2, x1:x2]
#         # roi = cv2.resize(roi, (64, 64))
#
#         # ---------------CALCULATE RESULT AND REACT----------------#
#         result = process_frame(target_frame)
#         if result == prev_result:
#             result_cnt += 1
#         else:
#             result_cnt = 1
#         prev_result = result
#         if result_cnt == decision_threshold:
#             executeTask(result)
#             result_cnt = 0
#
#         # -----------------DISPLAY CAMERA------------------#
#         frame = cv2.putText(frame, result, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
#         cv2.imshow("cam", frame)
#         cv2.waitKey(1)
#
# # print("MODEL INITIALIZED")
# # cap = cv2.VideoCapture(0)
# #
# # createHSVTrackBar()
# # createThreshTrackBar()
# #
# # background_process()
#
# # # Loading the model
# # from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# #
# # json_file = open("model-bw.json", "r")
# # model_json = json_file.read()
# # json_file.close()
# # cnn = tf.keras.models.model_from_json(model_json)
# # # load weights into new model
# # cnn.load_weights("model-bw.h5")
# # print("Loaded model from disk")
# #
# # train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# # training_set = train_datagen.flow_from_directory('dataset1/train', target_size=(64, 64), batch_size=32,
# #                                                  class_mode='categorical')
# #
# # test_image = image.load_img('dataset1/train/0/491.jpg', target_size=(64, 64))
# # test_image = image.img_to_array(test_image)
# # test_image = np.expand_dims(test_image, axis=0)
# # result = cnn.predict(test_image)
# # training_set.class_indices
# # if result[0][0] == 0:
# #     prediction = 'push_left'
# # if result[0][0] == 1:
# #     prediction = 'no_push'
# # else:
# #     prediction = 'push_right'
# #
# # print(prediction)
# #
# # # import numpy as np
# # # import tensorflow as tf
# # # # from webcam_gesture_recognition import dataset_creation as dc
# # # import operator
# # # import cv2
# # # import sys, os
# # #
# # #
# # # def roi_generation(frame):
# # #     x1 = int(0.5 * frame.shape[1])
# # #     y1 = 10
# # #     x2 = frame.shape[1] - 10
# # #     y2 = int(0.5 * frame.shape[1])
# # #     # Drawing the ROI (The increment/decrement by 1 is to compensate for the bounding box)
# # #     cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
# # #     # Extracting the ROI
# # #     roi = frame[y1:y2, x1:x2]
# # #     roi = cv2.resize(roi, (64, 64))
# # #     return roi
# # #
# # #
# # # # Loading the model
# # # json_file = open("model-bw.json", "r")
# # # model_json = json_file.read()
# # # json_file.close()
# # # loaded_model = tf.keras.models.model_from_json(model_json)
# # # # load weights into new model
# # # loaded_model.load_weights("model-bw.h5")
# # # print("Loaded model from disk")
# # #
# # #
# # # # Category dictionary
# # # categories = {0: 'Push_Left', 1: 'No_Push', 2: 'Push_Right'}
# # #
# # # cap = cv2.VideoCapture(0)
# # # while cap.isOpened():
# # #     _, frame = cap.read()
# # #     # Simulating mirror image
# # #     frame = cv2.flip(frame, 1)
# # #
# # #     # creating region of interest
# # #     processed = roi_generation(frame)
# # #
# # #     # # Got this from collect-data.py
# # #     # # Coordinates of the ROI
# # #     # x1 = int(0.5*frame.shape[1])
# # #     # y1 = 10
# # #     # x2 = frame.shape[1]-10
# # #     # y2 = int(0.5*frame.shape[1])
# # #     # # Drawing the ROI
# # #     # # The increment/decrement by 1 is to compensate for the bounding box
# # #     # cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
# # #     # # Extracting the ROI
# # #     # roi = frame[y1:y2, x1:x2]
# # #     #
# # #     # # Resizing the ROI so it can be fed to the model for prediction
# # #     # roi = cv2.resize(roi, (64, 64))
# # #     # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# # #     # _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
# # #     cv2.imshow("test", processed)
# # #     # Batch of 1
# # #     result = loaded_model.predict(processed.reshape(1, 64, 64, 3))
# # #     prediction = {'Push_Left': result[0][0],
# # #                   'No_Push': result[0][1],
# # #                   'Push_Right': result[0][2]}
# # #     # # Sorting based on top prediction
# # #     prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
# # #     #
# # #     # Displaying the predictions
# # #     cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
# # #     cv2.imshow("Frame", frame)
# # #
# # #     interrupt = cv2.waitKey(10)
# # #     if interrupt & 0xFF == 27: # esc key
# # #         break
# # #
# # #
# # # cap.release()
# # # cv2.destroyAllWindows()
