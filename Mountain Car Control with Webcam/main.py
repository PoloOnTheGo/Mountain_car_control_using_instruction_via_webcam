import tensorflow as tf
import cv2
import open_ai_env_agent
import image_processing_util as ipu

'''
-------PROGRAM STATES-------
0 - LEFT
1 - NO PUSH 
2 - RIGHT
------------END-------------
'''

'''
--------Pseudo code---------
1. initializing the environment
2. load_model
3. loop for each episode
    a. resetting env
    b.video_processing
        i. open video 
        ii. image_processing
        iii. interpreting_result_using_model
        iv. print_output_on_screen
        v. close_video
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


def interpreting_result_using_model(model, input_frame):
    modified_frame = input_frame.reshape(1, 64, 64, 3)
    output = model.predict(modified_frame)[0]
    result = list(output).index(max(output))
    print_output(str(result))
    return result


def print_output(action):
    if action == '0':
        action_text = "LEFT"
    elif action == '1':
        action_text = "NO PUSH"
    elif action == '2':
        action_text = "RIGHT"
    else:
        action_text = "UNIDENTIFIED"
    print(action_text)


def video_processing(model, env_state):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        processed_frame = ipu.process_video(cap, 'prediction', '')
        env.render()
        print(env_state)
        action = interpreting_result_using_model(model, processed_frame)
        print(action)
        env_state, reward, done, info = env.step(action)
        if done or env_state[0] >= 0.5:
            break
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:  # esc key
            break
    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------PROGRAM START-----------------------------------------#

env = open_ai_env_agent.initialize_environment('MountainCar-v0')
loaded_model = load_model()
for i_episode in range(3):
    state = env.reset()
    video_processing(loaded_model, state)
    print('Episode {}'.format(i_episode+1))

