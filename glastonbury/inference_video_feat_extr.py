import os 
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser(description='Easy video feature extractor')
parser.add_argument('--video_list', type=str, default = '/home/dipu/content_ps/data/gstbry/ann/final_clips_25sec.pkl')
parser.add_argument('--feat_dir', type=str, default='runs/videoclipfeats/')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100000)
parser.add_argument('--done_videos', type=str, default='')
parser.add_argument('--gpuid', type=str, default = '3')
parser.add_argument('--hflip', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64,)
parser.add_argument('--type', type=str, default='s3d', help='CNN type')
parser.add_argument('--half_precision', type=int, default=1,  help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=8, help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1, help='l2 normalize feature')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid


import torch
import sys
sys.path.append('../fairseq/examples/MMPT')
from mmpt.models import MMPTModel

sys.path.append('../fairseq/examples/MMPT/scripts/video_feature_extractor')
from torch.utils.data import DataLoader
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
from tqdm import tqdm
from pathbuilder import PathBuilder
from videoreader import Glastonbury_VideoLoader2
from utils import pickle_load, pickle_save, mkdir_if_missing

CONFIGS = {
     "s3d": {
        "fps": 30,
        "size": 224,
        "centercrop": True,
        "shards": 0,
    }
}

config = CONFIGS[args.type]
video_list = pickle_load(args.video_list)
video_list_to_process = video_list[args.start: args.end]
mkdir_if_missing(args.feat_dir)

feat_fn = f'{args.feat_dir}feats_start{args.start}_end{args.end}.pkl'

if os.path.exists(feat_fn):
    feats_dict = pickle_load(feat_fn)
    vfnames = feats_dict['vfnames']
    vfeats = feats_dict['vfeats']
    counter = len(vfnames)
    if 'skipped_videos' in feats_dict.keys():
        skipped_videos = feats_dict['skipped_videos']
    else:
        skipped_videos = []
else:
    vfnames = []
    vfeats = []    
    counter = 0
    skipped_videos = []


dataset = Glastonbury_VideoLoader2(
    video_list = video_list_to_process,
    done_videos=vfnames,
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
max_frames = 0
tic = time.time()

with torch.no_grad():
    for k, data in tqdm(enumerate(loader), total=loader.__len__(), ascii=True):  
        
        input_file = data['input'][0]
        #video_fn = input_file.split('/')[-1].split('.')[0]
        video_fn = input_file

        if data['video'].shape[1] == 0:
            print(f'Skiping this video {input_file}')
            continue
         
        if len(data['video'].shape) > 3: # [1, 300, 3, 224, 224]
            video = data['video'].squeeze()
            
            """
            # if video.shape[0] > 800:
            #     print(f'Skipping this video of {video.shape[0]} frames:  {input_file}')
            #     skipped_videos.append(input_file ) 
            #     continue
            
            # if video.shape[0] > max_frames:
            #     max_frames = video.shape[0]
            #     print(f'New max frame = {max_frames}')
            #     print(f'filename: {input_file}')
            """
            
            
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
                del output
                torch.cuda.empty_cache()
        
        counter+=1
        if k%10000==0:
            save_data = {'vfeats': vfeats, 'vfnames': vfnames, 'counter':counter, 'skipped_videos': skipped_videos}
            pickle_save(save_data, feat_fn)

save_data = {'vfeats': vfeats, 'vfnames': vfnames, 'counter':counter, 'skipped_videos': skipped_videos}
pickle_save(save_data, feat_fn)


# video_frames = np.load('scripts/video_feature_extractor/vggsoundfeats/2213.npy')
# print (video_frames.shape)

# B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
# video_frames =  video # torch.randn(1, 10, 30, 224, 224, 3)
# caps, cmasks = aligner._build_text_seq(
#     tokenizer("I am singing a song in the party", add_special_tokens=False)["input_ids"]
# )
# caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

# with torch.no_grad():
#     output = model(video_frames, caps, cmasks, returactivaten_score=True)
#     print(output['pooled_video'].shape)
    
# print(output["score"])  # dot-product

# python inference_video_feat_extr.py --start 0 --end 100000  # tamatoa gpu 3
# python inference_video_feat_extr.py --start 100000 --end 200000 --gpuid '0' 
# python inference_video_feat_extr.py --start 200000 --end 300000 --gpuid '1' 
# python inference_video_feat_extr.py --start 300000 --end 400000 --gpuid '2'
# python inference_video_feat_extr.py --start 400000 --end 600000 --gpuid '3'  
