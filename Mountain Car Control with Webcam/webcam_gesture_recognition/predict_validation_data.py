import numpy as np
import tensorflow as tf
import cv2
import pyautogui
from keras_preprocessing.image import ImageDataGenerator
import pyglet

from agent import agent_prototype
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


def interpreting_result_using_model(model, actual_frame, input_frame):
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

    return result


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


def video_processing(model, env_state, total_reward):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        actual_frame, processed_frame = ipu.process_video(cap, 'prediction', '')
        env.render()
        print(env_state)
        action = interpreting_result_using_model(model, actual_frame, processed_frame)
        print(action)
        env_state, reward, done, info = env.step(action)
        total_reward += reward
        if done or env_state[0] >= 0.5:
            break
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:  # esc key
            break
        # print_output_on_screen(processed_frame, output)
    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------PROGRAM START-----------------------------------------#

env = agent_prototype.initialize_environment('MountainCar-v0')
loaded_model = load_model()
tot_reward = 0
for i_episode in range(3):
    state = env.reset()
    video_processing(loaded_model, state, tot_reward)
    print('Episode {} Average Reward: {}'.format(i_episode+1, tot_reward))

