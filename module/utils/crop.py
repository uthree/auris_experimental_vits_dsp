import random
import torch

def decide_crop_range(max_length=500, frames=50):
    left = random.randint(0, max_length-frames)
    right = left + frames
    return (left, right)


def crop_features(z, area):
    left, right = area[0], area[1]
    return z[:, :, left:right]


def crop_waveform(wf, area, frame_size):
    left, right = area[0], area[1]
    return wf[:, left*frame_size:right*frame_size]

