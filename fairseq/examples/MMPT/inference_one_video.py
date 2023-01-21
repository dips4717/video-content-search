import torch as th
import pandas as pd
import os
import numpy as np
import ffmpeg
import random
from torch.utils.data import Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import numpy as np
from mmpt.models import MMPTModel
import sys
import argparse
import pickle
import json
sys.path.append('scripts/video_feature_extractor')
from preprocessing import Preprocessing

# Parameters
framerate = 30
size = 224
centercrop = True
hflip = False


def _get_video_dim(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams']
                         if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return height, width

def _get_output_dim(height, width):
    """
    keep the shorter side be `.size`, strech the other.
    """
    if height >= width:
        return int(height * size / width), size
    else:
        return size, int(width * size / height)

def _run(cmd, output_file):
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )
    if centercrop and isinstance(size, int):
        height, width = size, size
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    video = th.from_numpy(video.astype('float32'))
    
    return video.permute(0, 3, 1, 2)

meta = json.load(open('runs/vggsoundfeats/meta_plan.json', 'r'))
input_video_paths = meta['video_path']


for i in range (16357,16360):
    
    video_path = input_video_paths[i]
    output_file = None
    
    h, w = _get_video_dim(video_path)
    height, width = _get_output_dim(h, w)
    cmd = (ffmpeg
            .input(video_path)
            .filter('fps', fps=framerate)
            .filter('scale', width, height)
            )
    if hflip:
        cmd = cmd.filter('hflip')
    
    if centercrop:
        x = int((width - size) / 2.0)
        y = int((height - size) / 2.0)
        cmd = cmd.crop(x, y, size, size)
                      
    video = _run(cmd, output_file)
    print(video.shape)
    
    input ('Enter to continue')


#%% Model 
# model, tokenizer, aligner = MMPTModel.from_pretrained(
#     "projects/retri/videoclip/how2.yaml")
# model.eval()
# model=model.cuda()
# preprocess = Preprocessing('s3d')


# Forward Pass
# if len(video.shape) > 3:
#     video = video.squeeze()

# video = preprocess(video)
# video = video.permute (0,2,3,4,1)
# video = video.unsqueeze(dim=0)
# #print(video.shape)
# caps, cmasks = aligner._build_text_seq(tokenizer("I am singing a song in the party", add_special_tokens=False)["input_ids"])
# caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
# video, caps, cmasks = video.cuda(), caps.cuda(), cmasks.cuda()

# output = model(video, caps, cmasks, return_score=True, video_featonly=False )


