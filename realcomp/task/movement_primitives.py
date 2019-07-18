import numpy as np

def init_position():
    return np.zeros(9)


def hand_on_table(x):
    action = np.zeros(9)
    angle_from_x = x * np.pi / 4
    action[0] = x
    action[1] = np.pi / 2
    return action


def loop(x):
    while True:
        for _ in range(3):
            yield init_position()
        noise = (np.random.random() - 0.5) / 5
        curr_x = x + noise
        yield hand_on_table(-curr_x)
        yield hand_on_table(curr_x)
        yield init_position()
    
