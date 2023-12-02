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

# from sdes.sdes import MixSDE
from datasets import NoisyDataset, WSJ0_mix, musdb_mix
from pl_model import DiffSepModel
import musdb

def main():
    output_dir_base = Path("results")
    batch_size = 16
    max_len_s = 1
    sr = 44100
    max_data = max_len_s*sr
    device = 'cuda:0'

    model = DiffSepModel.load_from_checkpoint('exp/musdb18/2023-12-03_00-33-15_experiment-music-separation/checkpoints/epoch-006_si_sdr-0.000.ckpt')
    # transfer to GPU
    model = model.to(device)
    model.eval()
    
    musdb_list = musdb.DB(root='data/musdb18', subsets="test")
    
    for idx in range(len(musdb_list)):
        data = musdb_list[idx].stems
        
        data = list(map(lambda x: torch.from_numpy(x).float().transpose(0,1).to(device)[[0]], data))
        
        mix_full = data[0].unsqueeze(0)
        tgt_full = torch.cat(data[1:], dim=0).unsqueeze(0)
        
        for t in range(mix_full.shape[-1]//(max_data*batch_size)+1):
            mix = mix_full[:,:,t*max_data*batch_size+44100*30:(t+1)*max_data*batch_size+44100*30]
            mix = list(mix.split(max_data, dim=2))
            
            tgt = tgt_full[:,:,t*max_data*batch_size+44100*30:(t+1)*max_data*batch_size+44100*30]
            tgt = list(tgt.split(max_data, dim=2))
            if mix[-1].shape[-1] != max_data:
                mix[-1] = torch.nn.functional.pad(mix[-1], (0, max_data-mix[-1].shape[-1]))
                tgt[-1] = torch.nn.functional.pad(tgt[-1], (0, max_data-tgt[-1].shape[-1]))

            mix = torch.cat(mix, dim=0)
            tgt = torch.cat(tgt, dim=0)
            print(f"mix1 size: {mix.size()}")
            print(f"tgt size: {tgt.size()}")
            batch, *stats = model.normalize_batch((mix, tgt))

            mix, target = batch
            print(f"mix2 size: {mix.size()}")
            print(f"target size: {target.size()}")
            
            est, *_ = model.separate(mix)
            print(f"est1 size: {est.size()}")

            est = model.denormalize_batch(est, *stats)
            print(f"est2 size: {est.size()}")
            est = torch.cat(est.split(1, dim=0), dim=2).squeeze()
            print(f"est3 size: {est.size()}")
            if t == 0:
                est_full = est
                break
            else:
                est_full = torch.cat([est_full, est], dim=1)
            
        break

if __name__ == "__main__":
    main()