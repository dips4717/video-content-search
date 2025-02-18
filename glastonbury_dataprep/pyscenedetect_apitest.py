""" PySceneDetect API Test Script
Usage: Run as part of standard test suite, but can be manually invoked
by passing the video file to perform scene detection on as an argument
to this file, e.g. `python api_test.py SOME_VIDEO.mp4`
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

STATS_FILE_PATH = 'api_test_statsfile.csv'

output_dir = 'splitouts'
os.system(f'rm -rf {output_dir}')
os.system(f'mkdir {output_dir}')


def test_api(test_video_file):
    # (str) -> None
    """ Test overall PySceneDetect API functionality.
    Can be considered a high level integration/black-box test.
    """

    print("Running PySceneDetect API test...")
    print("PySceneDetect version being used: %s" % str(scenedetect.__version__))

    # Create a video_manager point to video file testvideo.mp4. Note that multiple
    # videos can be appended by simply specifying more file paths in the list
    # passed to the VideoManager constructor. Note that appending multiple videos
    # requires that they all have the same frame size, and optionally, framerate.
    video_manager = VideoManager([test_video_file])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector(threshold=27))
    base_timecode = video_manager.get_base_timecode()

    try:
        # If stats file exists, load it.
        if os.path.exists(STATS_FILE_PATH):
            # Read stats from CSV file opened in read mode:
            with open(STATS_FILE_PATH, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file)

        start_time = base_timecode + 20     # 00:00:00.667
        end_time = base_timecode + 20.0     # 00:00:20.000
        # Set video_manager duration to read frames from 00:00:00 to 00:00:20.
        #video_manager.set_duration(start_time=start_time, end_time=end_time)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()
        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list()
        # Like FrameTimecodes, each scene in the scene_list can be sorted if the
        # list of scenes becomes unsorted.
        
        fn = test_video_file.split('.')[0]
        save_path = f'{output_dir}/{fn}'
        
        split_video_ffmpeg([test_video_file], scene_list, f'{save_path}-scene$SCENE_NUMBER.mp4', '' )
        
        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i+1,
                scene[0].get_timecode(), scene[0].get_frames(),
                scene[1].get_timecode(), scene[1].get_frames(),))

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(STATS_FILE_PATH, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()


# Support running as a stand-alone file.
if __name__ == "__main__":
    test_video = 'goldeneye.mp4'
    test_api(test_video)
    
    # if len(sys.argv) < 2:
    #     print('Usage: %s [TEST_VIDEO_FILE]' % sys.argv[0])
    # else:
    #     test_api(sys.argv[1])