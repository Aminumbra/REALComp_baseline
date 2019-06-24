from tensorboardX import SummaryWriter
from random import random


def writer(logs_dir):
    return SummaryWriter(comment=logs_dir)
