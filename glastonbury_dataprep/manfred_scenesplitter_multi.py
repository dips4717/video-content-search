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
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import pandas as pd
from multiprocessing import Pool, Value

import glob
input_dir = '/mnt/amber/scratch/Dipu/glastonbury/videos'
vid_list= glob.glob(f'{input_dir}/*.mp4')
output_dir = '/mnt/amber/scratch/Dipu/glastonbury/clips'

# vid2process  = vid_list[1000:2000]
vid2process  = vid_list[4000:]

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
    
    video_manager = VideoManager([test_video_file])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=27))

    try:       
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)  # Perform scene detection on video_manager.
        scene_list = scene_manager.get_scene_list()   #Obtain list of detected scenes.
        
        prefn, fn = os.path.split(test_video_file)
        fn = fn.split('.')[0]
        save_path = f'{output_dir}/{fn}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)      
        
        split_video_ffmpeg([test_video_file], scene_list, f'{save_path}{fn}-scene$SCENE_NUMBER.mp4', '', suppress_output=True )
        
        # print('List of scenes obtained:')
        # for i, scene in enumerate(scene_list):
        #     print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        #         i+1,
        #         scene[0].get_timecode(), scene[0].get_frames(),
        #         scene[1].get_timecode(), scene[1].get_frames(),))
        
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
p = Pool(20)
aa = p.map(test_api, vid2process)

print ("DONE!!!!")
