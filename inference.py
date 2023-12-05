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
import fast_bss_eval

def demucs_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    references = references.transpose(1, 2).double().cpu()[None]
    estimates = estimates.transpose(1, 2).double().cpu()[None]
    
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=(2,3))
    den = torch.sum(torch.square(references - estimates), dim=(2,3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores[0]


def main():
    output_dir_base = Path("results")
    batch_size = 16
    max_len_s = 7
    sr = 44100
    max_data = max_len_s*sr
    device = 'cuda:0'

    model = DiffSepModel.load_from_checkpoint('exp/musdb18/2023-12-03_10-54-48_experiment-music-separation/checkpoints/epoch-239_si_sdr--9.370.ckpt')
    # model = DiffSepModel.load_from_checkpoint('exp/musdb18_demucs/2023-12-03_05-03-31_experiment-music-separation-demucs/checkpoints/epoch-999_si_sdr--16.382.ckpt')
    
    # transfer to GPU
    model = model.to(device)
    model.eval()
    
    musdb_list = musdb.DB(root='data/musdb18', subsets="test")
    
    each_loss = np.zeros(4)
    all_loss, demucs_all_loss = np.zeros(1), np.zeros(1)
    
    target_audios = [np.zeros(1) for _ in range(4)]
    est_audios = [np.zeros(1) for _ in range(4)]
    
    for idx in tqdm(range(len(musdb_list)//batch_size)):
        data = np.stack([musdb_list[idx].stems[:,:,np.random.randint(0,2)] 
                         for idx in range(idx * batch_size, (idx+1) * batch_size)])
        data = torch.tensor(data).float().to(model.device)

        mix = data[:,0,:].unsqueeze(dim=1)
        target = data[:,1:,:]
        
        batch, *stats = model.normalize_batch((mix, target))
        mix, target = batch
        print(f"{idx}th batch data prep done")
        
        print(f"{idx}th batch source sep in progress")
        est, *_ = model.separate(mix)
        print(f"{idx}th batch source sep done")
        est = model.denormalize_batch(est, *stats)
        
        test_loss = fast_bss_eval.si_sdr_pit_loss(est, target, clamp_db=None)
        each_loss += torch.mean(test_loss, axis=0).cpu().numpy()
        test_loss_mean = torch.mean(test_loss, axis=1)
        all_loss += torch.mean(test_loss_mean, axis=0).cpu().numpy()
        print(f"Test SI-SDR Loss: {test_loss}")
        print(f"Test SI-SDR Loss Mean: {test_loss_mean}")
        
        # demucs_test_loss = demucs_sdr(target, est)
        # # demucs_test_loss = museval.metrics.bss_eval(target.transpose(1,2).double().cpu().numpy(), est.transpose(1,2).double().cpu().numpy())
        # demucs_all_loss += torch.mean(demucs_test_loss, axis=0).cpu().numpy()
        # print(f"Demucs SDR Loss: {demucs_test_loss}")
        
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                target_audios[j] = np.concatenate((target_audios[j], target[i][j].cpu().numpy()))
            for j in range(est.shape[1]):
                est_audios[j] = np.concatenate((est_audios[j], est[i][j].cpu().numpy()))
        
        if idx == 0:
            if not os.path.exists("output"):
                os.makedirs("output")
            for i in range(target.shape[1]):
                sf.write(f"output/target{i}.wav", target[0][i].cpu().numpy(), 44100)
            for i in range(est.shape[1]):
                sf.write(f"output/est{i}.wav", est[0][i].cpu().numpy(), 44100)
    
    print(f"Test SI-SDR: {each_loss / (len(musdb_list)//batch_size)}, Mean:{all_loss / (len(musdb_list)//batch_size)}")
    # print(f"Demucs SDR: {demucs_all_loss / (len(musdb_list)//batch_size)}")
    
    if not os.path.exists("output"):
        os.makedirs("output")
    print("Saving Target Audios..")
    for i in range(target.shape[1]):
        sf.write(f"output/target{i}_all.wav", target_audios[i], 44100)
    print("Saving Est Audios..")
    for i in range(est.shape[1]):
        sf.write(f"output/est{i}_all.wav", est_audios[i], 44100)

if __name__ == "__main__":
    main()