# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import yaml
from omegaconf import OmegaConf
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
# from sdes.sdes import MixSDE
from datasets import NoisyDataset, WSJ0_mix, musdb_mix
from pl_model import DiffSepModel
import musdb
import museval
import soundfile as sf
import IPython.display as ipd


batch_size = 16
max_len_s = 3
sr = 16000
max_data = max_len_s*sr
device = 'cuda'
audio_save_path = 'data/musdb18_16000/test/'
sdr_save_path = 'eval_sdr'

model = DiffSepModel.load_from_checkpoint('exp/musdb18_16000/2023-11-28_13-16-04_experiment-music-separation-16000/checkpoints/epoch-5799_si_sdr-0.000.ckpt')
# transfer to GPU
model = model.to(device)
model.eval()


def separate(filename, device='cuda'):
    def separate_one_channel(mix):
        mix_full = mix.unsqueeze(0)

        for t in range(mix_full.shape[-1]//(max_data*batch_size)+1):
            mix = mix_full[:,:,t*max_data*batch_size:(t+1)*max_data*batch_size]
            mix = list(mix.split(max_data, dim=2))
            
            if mix[-1].shape[-1] != max_data:
                mix[-1] = torch.nn.functional.pad(mix[-1], (0, max_data-mix[-1].shape[-1]))
            
            mix = torch.cat(mix, dim=0).to(device)
            est = model.separate(mix)
            est = torch.cat(est.split(1, dim=0), dim=2).squeeze().detach().cpu()
            if t == 0:
                est_full = est
            else:
                est_full = torch.cat([est_full, est], dim=1)
        est_full = est_full[:,:mix_full.shape[-1]]
        est_full = est_full
        return est_full
    
    
    data = torch.cat(
            [torchaudio.load(f'{filename}/{inst}')[0] for inst in ['mixture.wav', 'drums.wav', 'bass.wav', 'other.wav', 'vocals.wav']], dim=0
        )
    data0 = data[::2]
    data1 = data[1::2]

    mix = [data0[[0]], data1[[0]]]
    est = [separate_one_channel(mix[0]), separate_one_channel(mix[1])]
    tgt = [data0[1:], data1[1:]]
    
    return est, tgt, mix


class Audio():
    def __init__(self, audio):
        self.audio = audio

class Track():
    def __init__(self, name, mixture, drums, bass, other, vocals, subset='test'):
        self.targets = {"mixture": Audio(mixture), "drums": Audio(drums), "bass": Audio(bass), "other": Audio(other), "vocals": Audio(vocals)}
        self.rate = sr
        self.name = name
        self.subset = subset
      
def calculate_sdr(name, est, tgt, mix):
    inst = ["drums","bass","other","vocals"]
    
    track_list = []
    track_list.append(Audio(torch.cat([mix[0], mix[1]], dim=0).transpose(0,1).numpy()))
    pred_dict = {}
    for i in range(len(inst)):
        tgt_ = torch.cat([tgt[0][[i]], tgt[1][[i]]], dim=0).transpose(0,1).numpy()
        est_ = torch.cat([est[0][[i]], est[1][[i]]], dim=0).transpose(0,1).numpy()
        est_ = est_ / np.std(est_) * np.std(tgt_)
        pred_dict[inst[i]] = est_
        track_list.append(tgt_)
    
    
    track = Track(name, *track_list)
  
    score = museval.eval_mus_track(
        track, pred_dict, output_dir=sdr_save_path
    )
    return score


inst = ["drums","bass","other","vocals"]

for song in tqdm(os.listdir(audio_save_path)):
    filename = f'{audio_save_path}/{song}'
    if os.path.isdir(filename):
        est, tgt, mix = separate(filename)
        new_filename=f'data/musdb18_16000_sample/test/{song}'
        os.makedirs(new_filename, exist_ok=True)
        for i in range(4):
            sf.write(f'{new_filename}/{inst[i]}.wav', torch.cat([est[0][[i]], est[1][[i]]], dim=0).transpose(0,1).numpy(), sr)
        
        scores = calculate_sdr(song, est, tgt, mix)
        
    
        