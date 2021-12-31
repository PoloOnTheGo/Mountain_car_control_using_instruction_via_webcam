# Importing the libraries
import cv2
from webcam_gesture_recognition import folder_management as fm
from webcam_gesture_recognition import image_processing_util as ipu


def save_processed_img(processed, interrupt):
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(fm.get_image_name('0'), processed)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(fm.get_image_name('1'), processed)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(fm.get_image_name('2'), processed)


def capture_process_video(mode):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, processed = ipu.process_video(cap, 'data_collect', mode)
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:  # esc key
            break
        save_processed_img(processed, interrupt)
    cap.release()
    cv2.destroyAllWindows()


def collect_data(data_mode):
    mode = data_mode
    # Create the directory structure
    fm.create_directories()

    # get working directory
    fm.set_directory(mode)

    # capture video feed
    capture_process_video(mode)


collect_data('train')
collect_data('test')
collect_data('validation')
