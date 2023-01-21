#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 20:20:35 2022

@author: dipu
"""

from utils import pickle_load, pickle_save

feat1 = pickle_load('runs/videoclipfeats/feats_start0_end100000.pkl')
feat2 = pickle_load('runs/videoclipfeats/feats_start100000_end200000.pkl')
feat3 = pickle_load('runs/videoclipfeats/feats_start200000_end300000.pkl')
feat4 = pickle_load('runs/videoclipfeats/feats_start300000_end400000.pkl')
feat5 = pickle_load('runs/videoclipfeats/feats_start400000_end600000.pkl')

all_feats = []
all_names = []

for feat in [feat1, feat2, feat3, feat4, feat5]:
    all_feats+= feat['vfeats']
    all_names+= feat['vfnames']

data_dict= {'feats': all_feats, 'fnames': all_names}
pickle_save(data_dict, 'runs/videoclipfeats/allfeats.pkl')
