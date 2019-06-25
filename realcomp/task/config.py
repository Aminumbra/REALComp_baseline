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
if not DEBUG and HASGUI:
    try:
        root = tkinter.Tk()
        response = tkinter.simpledialog.askstring("comment", "comment")
        root.destroy()
    except tkinter.TclError as _:
        pass

if len(sys.argv) > 1:
    response = f'{sys.argv[1]}:{time.strftime("%Y_%m_%d-%H_%M_%S")}{os.getpid()}'

if response is not None:
    experiment_name = response

print('EXPERIMENT:', experiment_name)

tensorboard = tensorboardX.SummaryWriter(os.path.join('runs', experiment_name), flush_secs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING {device}")

render = False
noop_steps = 120
frames_per_action = 30
intrinsic_frames = 10000
render_but_no_render = True
wtcheat = True
