import json

from nbformat import read
import ffmpeg
import numpy as np
from multiprocessing import Pool, Value
from collections import defaultdict
from utils import pickle_load, pickle_save

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
        print('{} / {} in {} min'.format(counter.value, len(all_files), toc))


    output[fn] = video.shape

    if counter.value % 50000 == 0:
        pickle_save(output, f'duration_dict_{counter.value}.pkl')
        
    return fn, video.shape
 
output = defaultdict()

with open ('runs/videoclipfeats/meta_plan.json', 'r') as f:
    ann = json.load(f)

all_files = ann['video_path']
all_files = [x for x in all_files if x[-3:] == 'mp4']

# done_pkl = pickle_load('duration_dict_1000000.pkl')
# done_files = list(done_pkl.keys())
# all_files = [x for x in all_files if x not in done_files]
# print(done_files[0])
# print(all_files[0])
# print(f'\n\n Starting again for {len(all_files)} files')

counter = Counter()


# Multi processsing 
p = Pool(40)
results = p.map(read_video, all_files)
output_final = {res[0]: res[1] for res in results}
pickle_save(output_final, 'duration_dict_final.pkl')

# Iterative 
# import time
# tic = time.time()
# output = defaultdict()
# # iteration
# for ii, fn in enumerate(all_files):
#     fn, shape = read_video(fn)
#     output[fn] = shape
#     if ii%100==0:
#         toc = (time.time() -tic) / 60
#         print(f'Completed {ii} files in {toc} min')
#         tic = time.time()
        




