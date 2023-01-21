import os
import pickle
import streamlit as st 
import torch
import sys
import torch.nn.functional as F

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

sys.path.append('../')
from mmpt.models import MMPTModel

st.set_page_config(layout="wide")

@st.cache
def model_load():
    model, tokenizer, aligner = MMPTModel.from_pretrained("/home/dipu/deepdiscover/mnt/content_trading_temp/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml")
    model.eval()
    model=model.cuda()
    print('Model loading done!!!')
    return model, tokenizer, aligner

@st.cache
def load_features():
    with open('../runs/vggsoundfeats/feats_count_129987.pkl', 'rb') as f:
        data = pickle.load(f)
    print ('object loaded from loadfeatures')
    feats = data['vfeats']
    fnames = data['vfnames']
    feats = torch.cat(feats,dim=0)
    #fnames = [x for sublist in fnames for x in sublist]
    print(feats.shape)
    print(len(fnames))
    feats = F.normalize(feats)
    return feats, fnames

# =============== STREAMLIT CONFIG =========================
st.image('logo2.png', use_column_width  = True)
st.markdown("<h1 style='text-align: center; color: Black;'>Search VGGSound Videos Using Free-form Text Queries</h1>", unsafe_allow_html=True)
# with st.beta_expander("Configuration Option"):
#     st.write("**AutoCrop** help the model by finding and cropping the biggest face it can find.")
#     st.write("**Gamma Adjustment** can be used to lighten/darken the image")

# menu = ['Image Based', 'Video Based']
st.sidebar.header('Config')
menu = ['VideoCLIP']
choice = st.sidebar.selectbox('Choose a model to run', menu)
# ============================================================

# Precomputed video features
feats, fnames = load_features()
model, tokenizer, aligner = model_load()



with st.form(key='my_form'):
	textquery = st.text_input('Search Query', 'A boy is skating.')
	submit_button = st.form_submit_button(label='Submit')


caps, cmasks = aligner._build_text_seq(tokenizer(textquery, add_special_tokens=False)["input_ids"])
caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
caps, cmasks = caps.cuda(), cmasks.cuda()
text_feat = output = model(None, caps, cmasks, text_featonly=True)
text_feat = text_feat['pooled_text']
text_feat = F.normalize(text_feat)

sim = torch.matmul(feats, text_feat.cpu().t())
_, top_inds = torch.topk(sim,6,dim=0)

topk_fnames = [fnames[x] for x in top_inds]

all_video_bytes = []
for names in topk_fnames:
    video_file = open(names, 'rb')
    video_bytes = video_file.read()
    all_video_bytes.append(video_bytes)

col1, col2, col3 = st.columns(3)
with col1:
    st.header("1.")
    st.video(all_video_bytes[0])

with col2:
    st.header("2.")
    st.video(all_video_bytes[1])

with col3:
    st.header("3.")
    st.video(all_video_bytes[2])

col21, col22, col23 = st.columns(3)

with col21:
    st.header("4.")
    st.video(all_video_bytes[3])

with col22:
    st.header("5.")
    st.video(all_video_bytes[4])

with col23:
    st.header("6.")
    st.video(all_video_bytes[5])


