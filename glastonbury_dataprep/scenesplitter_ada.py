#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:32:09 2022
PySceneDetect API Test Script
Usage: Run as part of standard test suite, but can be manually invoked
by passing the video file to perform scene detection on as an argument
to this file, e.g. `python api_test.py SOME_VIDEO.mp4`
@author: dipu
"""

from __future__ import print_function
import os
import sys
import scenedetect
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect import StatsManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
import pandas as pd
from multiprocessing import Pool, Value
import  argparse 
import glob


parser = argparse.ArgumentParser()
# parser.add_argument('--part', type=int, default=1)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=5000)
parser.add_argument('--machine', type=str, default='amber')
args = parser.parse_args()


if args.machine == 'amber':
    input_dir = '/mnt/amber/scratch/Dipu/glastonbury/videos'
    output_dir = '/mnt/amber/scratch/Dipu/glastonbury/clips_ada4_lumaFalse'
else:
    input_dir = '/user/HS103/dm0051/amber_scratch/glastonbury/videos'
    output_dir= '/user/HS103/dm0051/amber_scratch/glastonbury/clips_ada4_lumaFalse_test'


vid_list= glob.glob(f'{input_dir}/*.mp4')

# if args.part < 3: 
#     vid2process  = vid_list[(args.part-1)*2000 : args.part*2000 ]
# else:
#     vid2process  = vid_list[(args.part-1)*2000 : ]
vid2process  = vid_list[args.start : args.end]

class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value



def test_api(test_video_file):
    counter.add(1)
    
    prefn, fn = os.path.split(test_video_file)
    fn = fn.split('.')[0]
    save_path = f'{output_dir}/{fn}/'
    if not os.path.exists(save_path) or (len(os.listdir(save_path))  < 2):
        os.system (f'rm -rf {save_path}') 
        os.makedirs(save_path)      
    else:
        print('Already exists!')
        return None


    video_manager = VideoManager([test_video_file])
    video_manager.set_downscale_factor()
    video_manager.start()

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(AdaptiveDetector(video_manager, adaptive_threshold=4.0))

    try:       
        scene_manager.detect_scenes(frame_source=video_manager)  # Perform scene detection on video_manager.
        scene_list = scene_manager.get_scene_list()   #Obtain list of detected scenes.

        

        split_video_ffmpeg([test_video_file], scene_list, f'{save_path}{fn}-scene$SCENE_NUMBER.mp4', '', suppress_output=True )

        start_t = [x[0].get_timecode() for x in scene_list]
        end_t =   [x[1].get_timecode() for x in scene_list]
        start_f = [x[0].get_frames() for x in scene_list]
        end_f = [x[1].get_frames() for x in scene_list]
        
        df = pd.DataFrame({"start_time": start_t,
              "end_time": end_t,
              "start_frame": start_f,
              "end_frame": end_f})
        
        df.to_csv(f'{save_path}{fn}_scene_list.csv')
        
        if  counter.value % 10 == 0:
            print('{} / {}'.format(counter.value, len(vid2process)))
        
            
    finally:
        video_manager.release()
    
    return None

counter = Counter()
p = Pool(40)
aa = p.map(test_api, vid2process)



print ("DONE!!!!")


# python scenesplitter_ada.py --machine 'amber' --part 1
# python scenesplitter_ada.py --machine 'amber' --part 2
# python scenesplitter_ada.py --machine 'amber' --part 3


# amber
# python scenesplitter_ada.py --machine 'amber' --start 0 --end 1300
# python scenesplitter_ada.py --machine 'amber' --start 1300 --end 1700
# python scenesplitter_ada.py --machine 'amber' --start 1700 --end 5000



