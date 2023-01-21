import json

from nbformat import read
import ffmpeg
import numpy as np
from multiprocessing import Pool, Value
from collections import defaultdict
from utils import pickle_load, pickle_save
import os 
import argparse

import os 

root = '/mnt/amber/scratch/Dipu/glastonbury/clips_ada4_lumaFalse' 

# all_files = [] 
# for path, subdirs, files in os.walk(root):
#     for name in files:
#         all_files.append(os.path.join(path, name))
# all_vids = [x for x in all_files if x [-3:]== 'mp4']

# with open ('video_clips.txt', 'w') as f:
#     f.write('\n'.join(all_vids))

# video_clips_pkl = pickle_save(all_vids, 'video_clips.pkl')

parser = argparse.ArgumentParser()
# parser.add_argument('--part', type=int, default=1)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=5000)
parser.add_argument('--machine', type=str, default='amber')
args = parser.parse_args()
                         
class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value
      
import time
tic = time.time()


def read_video(fn):
    global tic
    global output 

    counter.add(1)

    try :
        cmd = (
            ffmpeg
            .input(fn)
            .filter('fps', fps=30)
            .filter('scale', 224, 224)
        )
        out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
        video = np.frombuffer(out, np.uint8).reshape([-1, 224, 224, 3])
    
    except:
        print (f'Cound not read this video: {fn}')
        with open('problematic_videos.txt', 'a') as f:
            f.write(f'{fn}')
        video = np.zeros(5)

    if counter.value % 1000 == 0 and counter.value >= 100:
        toc = (time.time() - tic) /60
        print('{} / {} in {} min'.format(counter.value, len(video_to_process), toc))


    output[fn] = video.shape

    if counter.value % 50000 == 0:
        pickle_save(output, f'duration_dict_{args.start}_{args.end}_{counter.value}.pkl')
        
    return fn, video.shape

video_clips = pickle_load('video_clips.pkl')
video_to_process = video_clips[args.start : args.end]
output = defaultdict()
counter = Counter()


# Multi processsing 
p = Pool(40)
results = p.map(read_video, video_to_process)
output_final = {res[0]: res[1] for res in results}
pickle_save(output_final, f'duration_dict__{args.start}_{args.end}_final.pkl')

# python duration.py --start 0 --end 100000       Running in decade
# python duration.py --start 100000 --end 200000  
# python duration.py --start 200000 --end 300000  Running in tamatoa
# python duration.py --start 300000 --end 400000  Running in amber
# python duration.py --start 400000 --end 500000
# python duration.py --start 500000 --end 600000
