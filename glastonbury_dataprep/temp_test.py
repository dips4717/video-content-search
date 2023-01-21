#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 05:49:55 2022

@author: dipu
"""


#%% Is exists and remove folder
import os
import glob

output_dir = '/mnt/amber/scratch/Dipu/glastonbury/clips_ada4_lumaFalse'
input_dir = '/mnt/amber/scratch/Dipu/glastonbury/videos'


def get_save_path(test_video_file):
    prefn, fn = os.path.split(test_video_file)
    fn = fn.split('.')[0]
    save_path = f'{output_dir}/{fn}/'
    return save_path

vid_list= glob.glob(f'{input_dir}/*.mp4')
vid_dir = [get_save_path(x) for x in vid_list]
os.path.exists(vid_dir[1180])

for pth in vid_dir[3000:3190]:
    os.system(f'rm -rf {pth}')


#%% Combine dictionaries
from utils import pickle_load, pickle_save

dict1 = pickle_load('duration_dict__0_100000_final.pkl')
dict2 = pickle_load('duration_dict__100000_200000_final.pkl')
dict3 = pickle_load('duration_dict__200000_300000_final.pkl')
dict4 = pickle_load('duration_dict__300000_400000_final.pkl')
dict5 = pickle_load('duration_dict__400000_600000_final.pkl')

dict_all = {**dict1, **dict2, **dict3, **dict4, **dict5}

pickle_save(dict_all, 'ada4_clips_duration.pkl')

#%% Combine videos that are greater than 15 frames and less than 25 frames, and chunked videos 
import pandas as pd
from utils import pickle_load,pickle_save
dict1 = pickle_load('ann/ada4_clips_duration.pkl')
df = pd.DataFrame({
        "fnames": list(dict1.keys()),
        "durations": list(dict1.values())
        })


df_valid = df[df['durations'].apply(lambda x:len(x) == 4)]
df_okay = df_valid [df_valid['durations'].apply(lambda x: x[0]>15 and x[0]<=750)]
chunked_videos = pickle_load('ann/chunked_ann.pkl')

chunked_lists = []
for fns in chunked_videos.values():
    chunked_lists+=fns



# chunked_lists = list(chunked_videos.values())
# chunked_lists = [x for sublist in chunked_lists for x in sublist]
chunked_lists = ['/mnt/amber/scratch/Dipu/glastonbury/' + x for x in chunked_lists]
okay_list = df_okay.fnames.to_list()

final_list_25sec = okay_list + chunked_lists

pickle_save(final_list_25sec, 'ann/final_clips_25sec.pkl')
final_list = pickle_load('ann/final_clips_25sec.pkl')

#%% Figuring out discrepancy in chunked videos 
import os

root = 'chunked_videos' 
all_files = [] 
for path, subdirs, files in os.walk(root):
    for name in files:
        all_files.append(os.path.join(path, name))
all_vids = [x for x in all_files if x [-3:]== 'mp4']


chunked_videos = pickle_load('ann/chunked_ann.pkl')
chunked_list = []
for fns in chunked_videos.values():
    chunked_list+=fns


#%% Test feats saved 
from utils import pickle_load

feats = pickle_load()