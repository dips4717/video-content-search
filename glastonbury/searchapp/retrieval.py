#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 09:55:20 2022

@author: dipu
"""


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import pickle
import streamlit as st 
import torch
import torch.nn.functional as F
import sys

sys.path.append('/home/dipu/deepdiscover/mnt/content_trading_temp/fairseq/examples/MMPT/')
from mmpt.models import MMPTModel

st.set_page_config(layout="wide")

def model_load():
    model, tokenizer, aligner = MMPTModel.from_pretrained("/home/dipu/deepdiscover/mnt/content_trading_temp/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml")
    model.eval()
    model=model.cuda()
    print('Model loading done!!!')
    return model, tokenizer, aligner


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
    return feats, fnames


model, tokenizer, aligner = model_load()
feats, fnames = load_features()
feats = F.normalize(feats)

caps, cmasks = aligner._build_text_seq(tokenizer("A boy is sitting and playing some instruments", add_special_tokens=False)["input_ids"])
caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
caps, cmasks = caps.cuda(), cmasks.cuda()

text_feat = output = model(None, caps, cmasks, text_featonly=True)
text_feat = text_feat['pooled_text']
text_feat = F.normalize(text_feat)

sim = torch.matmul( feats, text_feat.cpu().t())
_, top_inds = torch.topk(sim,3,dim=0)

aa = [fnames[x] for x in top_inds]