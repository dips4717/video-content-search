#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:46:53 2022

@author: dipu
"""

from utils import pickle_load

feat1 = pickle_load('runs/clipfeats/feats_start0_end80000.pkl')
feat2 = pickle_load('runs/clipfeats/feats_start80000_end160000.pkl')
feat3 = pickle_load('runs/clipfeats/feats_start160000_end240000.pkl')
feat4 = pickle_load('runs/clipfeats/feats_start240000_end320000.pkl')
feat5 = pickle_load('runs/clipfeats/feats_start320000_end400000.pkl')
feat6 = pickle_load('runs/clipfeats/feats_start400000_end480000.pkl')
feat7 = pickle_load('runs/clipfeats/feats_start480000_end600000.pkl')


def remove_duplicates (feats):
    fnames = feats['fnames']
    vfeats = feats['vfeats']

    fnames_ = []
    feats_ = []
    
    for fname, feat in zip(fnames, vfeats):
        if fname not in fnames_:
            fnames_.append(fname)
            feats_.append(feat)
    
    return fnames_, feats_
     

fnames1, vfeats1 =remove_duplicates(feat1)
fnames2, vfeats2 =remove_duplicates(feat2)
fnames3, vfeats3 =remove_duplicates(feat3)
fnames4, vfeats4 =remove_duplicates(feat4)
fnames5, vfeats5 =remove_duplicates(feat5)
fnames6, vfeats6 =remove_duplicates(feat6)
fnames7, vfeats7 =remove_duplicates(feat7)

def flatten(lofl):
    return [x for y in lofl for x in y]

fnames1_flat = flatten(fnames1)
fnames2_flat = flatten(fnames2)
fnames3_flat = flatten(fnames3)
fnames4_flat = flatten(fnames4)
fnames5_flat = flatten(fnames5)
fnames6_flat = flatten(fnames6)
fnames7_flat = flatten(fnames7)

vfeats1_flat = flatten(vfeats1)
vfeats2_flat = flatten(vfeats2)
vfeats3_flat = flatten(vfeats3)
vfeats4_flat = flatten(vfeats4)
vfeats5_flat = flatten(vfeats5)
vfeats6_flat = flatten(vfeats6)
vfeats7_flat = flatten(vfeats7)

all_vfeats = vfeats1_flat + vfeats2_flat + vfeats3_flat + vfeats4_flat +\
                vfeats5_flat + vfeats6_flat + vfeats7_flat
    
all_fnames = fnames1_flat +  fnames2_flat + fnames3_flat + fnames4_flat +\
                fnames5_flat + fnames6_flat + fnames7_flat                   


#%%
from utils import pickle_load, pickle_save
import torch

feat_re = pickle_load('runs/clipfeats/feats_start480000_end600000_remaining.pkl')


feat1to7 = pickle_load('runs/clipfeats/feats1to7.pkl')
feats_17 = torch.stack(feat1to7['vfeats'],dim=0)
fnames_17 = feat1to7['fnames']

feats_re = torch.cat(feat_re['vfeats'],dim=0)
fnames_re = feat_re['fnames']
fnames_re = [x for y in fnames_re for x in y]


all_feats = torch.cat([feats_17, feats_re], dim=0)
all_fnames = fnames_17 + fnames_re

data = {'fnames': all_fnames, 'vfeats': all_feats}

pickle_save(data, 'runs/clipfeats/allclipfeats.pkl')
