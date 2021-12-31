import os

sub_dir = ['0', '1', '2']  # The list of sub-directories
root_dir = 'dataset1'
main_dir = ['train', 'test', 'validation']  # Name of the sub-directories
# act_no_fld_name_map = {'0': '0', '1': '1', '2': '2'}
working_directory: str = ''


def create_directories():
    for i in main_dir:
        for j in sub_dir:
            dir_name = root_dir + '/' + i + '/' + j

            # Create target Directory if don't exist
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print("Directory ", dir_name, " Created ")


def set_directory(mode):
    global working_directory
    working_directory = root_dir + '/' + mode + '/'
    return working_directory


def get_image_count(act_no):
    return str(len(os.listdir(working_directory + act_no)))


def get_image_name(act_no):
    return working_directory + act_no + '/' + get_image_count(act_no) + '.jpg'
