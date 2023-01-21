# REad video frames

import os
import numpy as np
import ffmpeg


def _get_video_dim(self, video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams']
                            if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return height, width

def _get_output_dim(self, h, w):
    if isinstance(self.size, tuple) and len(self.size) == 2:
        return self.size
    elif h >= w:
        return int(h * self.size / w), self.size
    else:
        return self.size, int(w * self.size / h)


#%% 
# video_path = 
# output_file = 

# h, w = _get_video_dim(video_path)
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# height, width = _get_output_dim(h, w)

# cmd = (
#     ffmpeg.input(video_path)
#     .filter('fps', fps=30)
#     .filter('scale', width, height))

# hflip=True
# centercrop=True

# if hflip:
#     cmd = cmd.filter('hflip')

# if centercrop:
#     x = int((width - self.size) / 2.0)
#     y = int((height - self.size) / 2.0)
#     cmd = cmd.crop(x, y, self.size, self.size)
# video = self._run(cmd, output_file)