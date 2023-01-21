from pytube import YouTube
import cv2
from PIL import Image
import clip
import torch
import math
import numpy as np

def download_ytv(url, fn):
    streams = YouTube(url).streams.filter(adaptive=True, subtype="mp4", resolution="360p", only_video=True)
    if len(streams) == 0: # Check if there is a valid stream
      raise "No suitable stream found for this YouTube video!"    
    print("Downloading...")
    streams[0].download(filename=fn) # Download the video
    print("Download completed.")
    return


N = 120    # How much frames to skip
url =  "https://www.youtube.com/watch?v=PGMu_Z89Ao8"
fn = 'video.mp4'
download_ytv(url, fn)

video_frames = []

# Open the video file
capture = cv2.VideoCapture('video.mp4')
fps = capture.get(cv2.CAP_PROP_FPS)

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
  current_frame += N
  capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

# Print some statistics
print(f"Frames extracted: {len(video_frames)}")

#%% Load pulbic CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Encode all frames using CLIP. The encoding is done in batches for a better efficiency.
# You can try tuning the batch size for very large videos, but it should usually be OK
batch_size = 256
batches = math.ceil(len(video_frames) / batch_size)

# The encoded features will bs stored in video_features
video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

# Process each batch
for i in range(batches):
  print(f"Processing batch {i+1}/{batches}")

  # Get the relevant frames
  batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
  
  # Preprocess the images for the batch
  batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
  
  # Encode with CLIP and normalize
  with torch.no_grad():
    batch_features = model.encode_image(batch_preprocessed)
    batch_features /= batch_features.norm(dim=-1, keepdim=True)

  # Append the batch to the list containing all features
  video_features = torch.cat((video_features, batch_features))

# Print some stats
print(f"Features: {video_features.shape}")





