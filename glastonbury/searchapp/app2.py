import os
import pickle
import streamlit as st 
import torch
import sys
import torch.nn.functional as F
import time
import os 
from collections import OrderedDict

gpu_id = 1 
#os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
import clip

device = f'cuda:{gpu_id}'
#device = 'cpu'
#device = 'cpu'
sys.path.append('/home/dipu/content_trading/code/fairseq/examples/MMPT/')
from mmpt.models import MMPTModel

st.set_page_config(layout="wide")

@st.cache
def model_load():
    model, tokenizer, aligner = MMPTModel.from_pretrained("/home/dipu/content_trading/code/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml")
    model.eval()
    model=model.to(device)
    print('Model loading done!!!')
    return model, tokenizer, aligner


@st.cache
def load_features():
    with open('../runs/videoclipfeats/allfeats.pkl', 'rb') as f:
        data = pickle.load(f)
    print ('object loaded from loadfeatures')
    feats = data['feats']
    fnames = data['fnames']
    feats = torch.cat(feats,dim=0)
    feats = F.normalize(feats)
    return feats, fnames

@st.cache
def load_imageclip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache
def load_imclip_features():
    with open('/home/dipu/content_trading/code/CLIPVideoSearch/runs/clipfeats/allclipfeats.pkl', 'rb') as f:
        data = pickle.load(f)
        print('CLipfeats loaded!!!')
    feats = data['vfeats']
    fnames = data['fnames']
    return feats, fnames 


# =============== STREAMLIT CONFIG =========================
st.image('logo2.png', use_column_width  = True)
st.markdown("<h1 style='text-align: center; color: Black;'>Search Glastonbury Clips Using Free-form Text </h1>", unsafe_allow_html=True)
# with st.beta_expander("Configuration Option"):
#     st.write("**AutoCrop** help the model by finding and cropping the biggest face it can find.")
#     st.write("**Gamma Adjustment** can be used to lighten/darken the image")

# menu = ['Image Based', 'Video Based']
st.sidebar.header('Config')
menu = ['VideoCLIP', 'ImageClip']
choice = st.sidebar.selectbox('Choose a model to run', menu)
# ============================================================

with st.form(key='my_form'):
	textquery = st.text_input('Search Query', 'Wide angle shot of crowd cheering.')
	submit_button = st.form_submit_button(label='Submit')

# ============================================================
# Precomputed video features # Video Clip 
feats, fnames = load_features()
model, tokenizer, aligner = model_load()
tic = time.time()
caps, cmasks = aligner._build_text_seq(tokenizer(textquery, add_special_tokens=False)["input_ids"])
caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
caps, cmasks = caps.to(device), cmasks.to(device)
text_feat = output = model(None, caps, cmasks, text_featonly=True)
text_feat = text_feat['pooled_text']
text_feat = F.normalize(text_feat)

sim = torch.matmul(feats, text_feat.cpu().t())
_, top_inds = torch.topk(sim,20,dim=0)
toc = time.time() - tic

topk_fnames = [fnames[x] for x in top_inds]
# print(topk_fnames)
st.text(f'Feat extraction and search completed in {toc:2f} sec')

# ============================================================

# Image Clip 
model_imclip, preprocess = load_imageclip_model()
ifeats, ifnames = load_imclip_features()


tic = time.time()
with torch.no_grad():
    text_features = model_imclip.encode_text(clip.tokenize(textquery).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_features = text_features.type(torch.float32)

# print (torch.dtype(ifeats))
# print (torch.dtype(text_features.cpu()))

similarities =  torch.matmul(ifeats, text_features.cpu().t()) #(100.0 * ifeats @ text_features.T)
values, best_photo_idx = similarities.topk(20000, dim=0)

best_clips = [ifnames[x] for x in best_photo_idx]
best_clips = list(OrderedDict.fromkeys(best_clips))
best_clips = best_clips[:6]
# print(best_clips)

toc = time.time() - tic
st.text(f'Image CLIP extraction and search completed in {toc:2f} sec')

all_video_bytes = []
for names in topk_fnames:
    video_file = open(names, 'rb')
    video_bytes = video_file.read()
    all_video_bytes.append(video_bytes)

all_video_bytes_clip = []
for names in best_clips:
    video_file = open(names, 'rb')
    video_bytes = video_file.read()
    all_video_bytes_clip.append(video_bytes)



col11, col12 = st.columns(2)
col21, col22 = st.columns(2)
col31, col32 = st.columns(2)
col41, col42 = st.columns(2)
col51, col52 = st.columns(2)
col61, col62 = st.columns(2)

with col11:
    st.header("1. VideoClip")
    st.video(all_video_bytes[0])
with col21:
    st.header("2. VideoClip")
    st.video(all_video_bytes[1])
with col31:
    st.header("3. VideoClip")
    st.video(all_video_bytes[2])
with col41:
    st.header("4. VideoClip")
    st.video(all_video_bytes[3])
with col51:
    st.header("5. VideoClip")
    st.video(all_video_bytes[4])
with col61:
    st.header("6. VideoClip")
    st.video(all_video_bytes[5])


with col12:
    st.header("1. ImageClip")
    st.video(all_video_bytes_clip[0])
with col22:
    st.header("2. ImageClip")
    st.video(all_video_bytes_clip[1])
with col32:
    st.header("3. ImageClip")
    st.video(all_video_bytes_clip[2])
with col42:
    st.header("4. ImageClip")
    st.video(all_video_bytes_clip[3])
with col52:
    st.header("5. ImageClip")
    st.video(all_video_bytes_clip[4])
with col62:
    st.header("6. ImageClip")
    st.video(all_video_bytes_clip[5])

# streamlit run app.py



#   Local URL: http://localhost:8501
#   Network URL: http://131.227.94.253:8501
