from pytube import YouTube
import cv2
import os
from PIL import Image
import clip
import torch
import math
import numpy as np
import argparse
from utils import pickle_load, pickle_save, mkdir_if_missing
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Easy video feature extractor')
#parser.add_argument('--video_list', type=str, default = '/home/dipu/content_ps/data/gstbry/ann/final_clips_25sec.pkl')
parser.add_argument('--feat_dir', type=str, default='runs/clipfeats/')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100000)
parser.add_argument('--skip_frames', type=int, default=30)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--ps_mount', type=str, default='content_ps')

args = parser.parse_args()
args.video_list = f'/home/dipu/{args.ps_mount}/data/gstbry/ann/final_clips_25sec.pkl'

#Model
model, preprocess = clip.load("ViT-B/32", device=args.device)

def extract_frames(fn):
    # Open the video file
    # print(fn)
    capture = cv2.VideoCapture(fn)
    fps = capture.get(cv2.CAP_PROP_FPS)

    video_frames = []
    current_frame = 0
    while capture.isOpened():
        # Read the current frame
        ret, frame = capture.read()
        # Convert it to a PIL image (required for CLIP) and store it

        if ret == True:
            video_frames.append(Image.fromarray(frame[:, :, ::-1]))
        else:
            break
        # Skip N frames
        current_frame += args.skip_frames
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # Print some statistics
    # print(f"Frames extracted: {len(video_frames)}")

    return video_frames

video_list = pickle_load(args.video_list)
video_list_to_process = video_list[args.start: args.end]
mkdir_if_missing(args.feat_dir)
feat_fn = f'{args.feat_dir}feats_start{args.start}_end{args.end}.pkl'


if os.path.exists(feat_fn):
    feats_dict = pickle_load(feat_fn)
    fnames = feats_dict['fnames']
    vfeats = feats_dict['vfeats']
    done = [x[0] for x in fnames]
    done = list(set(done))
    video_list_to_process = [x for x in video_list_to_process if x not in done]

    print (f'Loading partially completed feat_dict with {len(fnames)} files')
else:
    fnames = []
    vfeats = []    



for ii, fn in tqdm(enumerate (video_list_to_process), total= len(video_list_to_process), ascii=True) :
    video_frames = extract_frames(fn)
    
    batches = math.ceil(len(video_frames) / args.batch_size)
    # The encoded features will bs stored in video_features
    video_features = torch.empty([0, 512], dtype=torch.float16).to(args.device)
    # Process each batch
    for i in range(batches):
        # print(f"Processing batch {i+1}/{batches}")
        # Get the relevant frames
        batch_frames = video_frames[i*args.batch_size : (i+1)*args.batch_size]
        # Preprocess the images for the batch
        batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(args.device)
        
        # Encode with CLIP and normalise
        with torch.no_grad():
            batch_features = model.encode_image(batch_preprocessed)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)

        # Append the batch to the list containing all features
        video_features = torch.cat((video_features, batch_features))
        video_features = video_features.cpu()
        tmp_fn_list = [fn] * video_features.shape[0]
        
        fnames.append(tmp_fn_list)
        vfeats.append(video_features)
        
    if ii%10000==0:
            save_data = {'vfeats': vfeats, 'fnames': fnames}
            pickle_save(save_data, feat_fn)
        

save_data = {'vfeats': vfeats, 'fnames': fnames}
pickle_save(save_data, feat_fn)



# python featextr_glastonbury.py --start 0 --end 80000 --device 'cpu'        # feat1
# python featextr_glastonbury.py --start 80000 --end 160000 --device 'cpu'   # feat2
# python featextr_glastonbury.py --start 160000 --end 240000  --device 'cpu' # feat3 
# python featextr_glastonbury.py --start 240000 --end 320000  --device 'cuda:0'  #feat4
# python featextr_glastonbury.py --start 320000 --end 400000  --device 'cuda:1'  # feat5
# python featextr_glastonbury.py --start 400000 --end 480000  --device 'cuda:2' # feat6 
# python featextr_glastonbury.py --start 480000 --end 600000  --device 'cuda:3'  # feat7 