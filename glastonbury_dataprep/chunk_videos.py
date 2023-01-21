import os 
import numpy as np
import pandas as pd 
from utils import pickle_load, pickle_save, mkdir_if_missing
import time
from multiprocessing import Pool, Value
from collections import defaultdict

class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value
   

dict1 = pickle_load('ann/ada4_clips_duration.pkl')
df = pd.DataFrame({
        "fnames": list(dict1.keys()),
        "durations": list(dict1.values())
        })

df_valid = df[df['durations'].apply(lambda x:len(x) == 4)]
df_okay = df_valid [df_valid['durations'].apply(lambda x: x[0]>15 and x[0]<=750)]
df_g750 = df_valid [df_valid['durations'].apply(lambda x: x[0]>750)]

size = 25

def split_video(ii):
    path = df_g750.iloc[ii].fnames
    nframes = df_g750.iloc[ii].durations[0]
    fn = path.split('/')[-1].split('.')[0]    
    total_time = int(nframes/30)
    
    splits = list(range(0,total_time, size))
    if len(splits)==1:
        starts = [splits[0]]
    else:
        starts = splits[:-1]
    lengths = [size] * len(starts)
    
    #split_fns = [x[:-4]+f'_split_{jj}.mp4' for jj, x in enumerate([path]*len(starts))] 
    
    save_folder = f'chunked_videos/{fn}/'
    mkdir_if_missing(save_folder)
    split_fns = [f'{save_folder}split_{jj}.mp4' for jj, x in enumerate([path]*len(starts))]
    
    # if not all([os.path.exists(x) for x in split_fns]):
    #     raise('Not found')
        

    # with open('chunked_log.txt', 'a') as f:
    #     f.write('{fn}\n')

    # TEsting
    # 
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    
    # split_fns = [f'{save_folder}split_{jj}.mp4' for jj, x in enumerate([path]*len(starts))]    
  
    # csv = pd.DataFrame(
    #         { "start_time": starts,
    #           "length": lengths,
    #           "rename_to":split_fns
    #         })
    # csv_fn = f'tmp/{fn}.csv'
    # csv.to_csv(csv_fn, index=False)   
    # os.system(f'cp -rf {path} {save_folder}')
    
    for s,l,f in  zip(starts, lengths, split_fns):
        cmd = f'ffmpeg -ss {s} -t {l} -i {path}  {f}'
        os.system(cmd)

    #os.system(f'python ffmpeg_split.py -f {path} -m {csv_fn}')
    #os.system(f'rm -rf {csv_fn}')
    return ii, split_fns



# split_video(0)
tic = time.time()

counter = Counter()
p = Pool(40)
print("[INFO] Start")
ids = list (range (len(df_g750)))
results = p.map(split_video, ids) 
all_feats = {res[0]: res[1] for res in results}
pickle_save(all_feats, 'chunked_ann.pkl')
toc = (time.time() - tic) /60
print(f'\n\n Completed videos in {toc:.4f} mins \n\n')









