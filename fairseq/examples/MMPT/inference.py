import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import numpy as np
from mmpt.models import MMPTModel

import sys
import argparse
import pickle

sys.path.append('scripts/video_feature_extractor')
import time

def pickle_load(fn):
    with open(fn,'rb') as f:
        data = pickle.load(f)
    print(f'Obj loaded from {fn}')
    return data

def pickle_save(data,fn):
    with open(fn, 'wb') as f:
        data = pickle.dump(data,f)
    print(f'Obj saved to {fn}')

#%%

from torch.utils.data import DataLoader
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
from tqdm import tqdm
from pathbuilder import PathBuilder
from videoreader import VGGSound_VideoLoader

parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument('--vdir', type=str, default = '/mnt/tamatoa/scratch/alex/vggsound/data/videos/train/')
parser.add_argument('--fdir', type=str, default='runs/vggsoundfeats/')
parser.add_argument('--feat_fn', type=str, default='runs/vggsoundfeats/feats_count_99994.pkl')
parser.add_argument('--hflip', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64,)
parser.add_argument('--type', type=str, default='s3d', help='CNN type')
parser.add_argument('--half_precision', type=int, default=1,  help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=8, help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1, help='l2 normalize feature')

args = parser.parse_args()

CONFIGS = {
     "s3d": {
        "fps": 30,
        "size": 224,
        "centercrop": True,
        "shards": 0,
    }
}

config = CONFIGS[args.type]

video_dirs = args.vdir
feature_dir = args.fdir
video_dict = PathBuilder.build(video_dirs, feature_dir, ".npy", config["shards"])

if os.path.exists(args.feat_fn):
    feats_dict = pickle_load(args.feat_fn)
    vfnames = feats_dict['vfnames']
    vfeats = feats_dict['vfeats']
    counter = len(vfnames)
else:
    vfnames = []
    vfeats = []
    counter = 0


dataset = VGGSound_VideoLoader(
    done_videos=vfnames,
    video_dict=video_dict,
    framerate=config["fps"],
    size=config["size"],
    centercrop=config["centercrop"],
    hflip=args.hflip
)

n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=None,
)
preprocess = Preprocessing(args.type)

model, tokenizer, aligner = MMPTModel.from_pretrained(
    "projects/retri/videoclip/how2.yaml")
model.eval()
model=model.cuda()

print (f'Length of the dataloader is {len(loader)}')

tic = time.time()
with torch.no_grad():
    for k, data in tqdm(enumerate(loader), total=loader.__len__(), ascii=True):
        if k == 100000:
            break
        
        if k%10000==0:
            save_fn = f'runs/vggsoundfeats/feats_count_{counter}.pkl'
            save_data = {'vfeats': vfeats, 'vfnames': vfnames, 'counter':counter}
            pickle_save(save_data,save_fn)
            
        input_file = data['input'][0]
        video_fn = input_file.split('/')[-1].split('.')[0]
        output_file = data['output'][0]
        
        if data['video'].shape[1] == 0 :
            print(f'Skiping this video {input_file}')
            continue
         
        if len(data['video'].shape) > 3: # [1, 300, 3, 224, 224]
            video = data['video'].squeeze()  
            if len(video.shape) == 4:
                video = preprocess(video)
                video = video.permute (0,2,3,4,1)  #[10, 3, 30, 224, 224]
                video = video.unsqueeze(dim=0)
                #print(video.shape)
                caps, cmasks = aligner._build_text_seq(tokenizer("I am singing a song in the party", add_special_tokens=False)["input_ids"])
                caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
                video, caps, cmasks = video.cuda(), caps.cuda(), cmasks.cuda()
                
                output = model(video, caps, cmasks, return_score=True, video_featonly=False )

                vfeats.append(output['pooled_video'].detach().cpu())
                #tfeats.append(output['pooled_text'])
                vfnames.append(input_file)
        counter+=1
        



# video_frames = np.load('scripts/video_feature_extractor/vggsoundfeats/2213.npy')
# print (video_frames.shape)

# B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
# video_frames =  video # torch.randn(1, 10, 30, 224, 224, 3)
# caps, cmasks = aligner._build_text_seq(
#     tokenizer("I am singing a song in the party", add_special_tokens=False)["input_ids"]
# )

# caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

# with torch.no_grad():
#     output = model(video_frames, caps, cmasks, return_score=True)
#     print(output['pooled_video'].shape)
    
# print(output["score"])  # dot-product