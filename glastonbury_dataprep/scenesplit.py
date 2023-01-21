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
from scenedetect.detectors import ContentDetector,  ThresholdDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg

import pandas as pd


output_dir = 'splitouts_2'
detector = ['con', 'thres', 'ada']
threshold = 35
adaptive_threshold = 4
algo = detector[2]
luma_only = True

fn = 'videos/1Cf6iW6_Xoc.mp4'
# fn = 'videos/0go_RY8M-rI.mp4'
# fn = 'videos/goldeneye.mp4'
# fn = 'videos/-7cjH1nPFDc.mp4'


def test_api(test_video_file):
    prefn, fn = os.path.split(test_video_file)
    fn = fn.split('.')[0]
    
    video_manager = VideoManager([test_video_file])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
        
    
    if algo == 'con':
        Detector = ContentDetector
    elif algo == 'thres':
        Detector = ThresholdDetector
    elif  algo == 'ada':
        Detector = AdaptiveDetector
       
    try:       
        video_manager.set_downscale_factor()
        video_manager.start()
        
        if algo != 'ada':    
            scene_manager.add_detector(Detector(threshold=threshold, luma_only=luma_only))
            save_path = f'{output_dir}/{fn}_{algo}_{threshold}_luma{int(luma_only)}/'
        else:
            scene_manager.add_detector(Detector(video_manager, adaptive_threshold= adaptive_threshold, luma_only=luma_only))
            save_path = f'{output_dir}/{fn}_{algo}_{adaptive_threshold}_luma{int(luma_only)}/'

        
        scene_manager.detect_scenes(frame_source=video_manager)  # Perform scene detection on video_manager.
        scene_list = scene_manager.get_scene_list()   #Obtain list of detected scenes.
        
       
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)      
        
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
        
            
    finally:
        video_manager.release()



test_video = fn
test_api(test_video)

