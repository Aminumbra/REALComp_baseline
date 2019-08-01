import sys
import time

try:
    import tkinter.simpledialog
except ImportError:
    HASGUI = False
else:
    HASGUI = True

import os
import tensorboardX
import torch

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
print("DEBUG: ", DEBUG)

experiment_name = "DEBUG:" + time.strftime("%Y_%m_%d-%H_%M_%S")

response = None
# if not DEBUG and HASGUI:
#     try:
#         root = tkinter.Tk()
#         response = tkinter.simpledialog.askstring("comment", "comment")
#         root.destroy()
#     except tkinter.TclError as _:
#         pass

if len(sys.argv) > 1:
    response = f'{sys.argv[1]}:{time.strftime("%Y_%m_%d-%H_%M_%S")}{os.getpid()}'

if len(sys.argv) > 2:
    model_to_load = f'{sys.argv[2]}'
else:
    model_to_load = ''

if response is not None:
    experiment_name = response

print('EXPERIMENT:', experiment_name)

tensorboard = tensorboardX.SummaryWriter(os.path.join('runs', experiment_name), flush_secs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING {device}")

render = False
noop_steps = 2
frames_per_action = 1
intrinsic_frames = 8000
# render_but_no_render = True
enjoy = True
wtcheat = True
save_every = False
extrinsic_trials = 10
num_envs = 4
observations_to_stack = 1
pre_train_cnn = False
reset_on_touch = False
random_reset = None#"random"
random_goal = "random"
actions_per_episode = 1
image_shape = (144, 72)
lr = 3e-4
ludicrous_speed = True

if DEBUG:
    noop_steps = 1
