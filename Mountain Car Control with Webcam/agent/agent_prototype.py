import gym

# env = ''


def initialize_environment(env_name):
    # global env
    # msg = 'Environment {} already initialized.'.format(env_name)
    # if env != '':
    env = gym.make(env_name)
    # msg = 'Environment {} initialized.'.format(env_name)
    # print(msg)
    # print(env.id)
    return env
