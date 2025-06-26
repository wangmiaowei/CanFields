#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import pytorch3d.ops.sample_farthest_points as sfp
import torch
import torch.utils.data
import point_cloud_utils as pcu
from pytorch3d.ops.knn import knn_gather, knn_points
#import deep_sdf.workspace as ws

class DeformSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        with_missing=False,
        train_rate = 1, # train/total_rate 
        min_norm_normal=1e-5
    ):
        obj_name = filename.split('/')[-1]
        pc_path = filename+"/4k_pc/"


        total_lens = len(os.listdir(pc_path))
        total_idx = set(range(total_lens))
        if train_rate==1:
            self.lens = total_lens
            self.train_idx = total_idx
            self.train_idx = list(self.train_idx)
            self.train_idx.sort()
            self.eval_idx = None
        else:
            train_num = int(total_lens * train_rate)
            self.lens = train_num #self.lens: training dataset lens
            self.train_idx = (np.array(range(self.lens))*(total_lens/self.lens)).astype(np.int64)
            self.eval_idx = total_idx - set(self.train_idx)
            self.eval_idx = list(self.eval_idx)
            self.eval_idx.sort()
        
        self.train_xx = []
        self.train_nn= []
        self.normalized_times = []
        for j in self.train_idx:
            self.normalized_times.append(torch.tensor(j/total_lens))
            full_name = pc_path + obj_name+"_"+str(j)+".ply"
            #print('full_name: ',full_name)
            v, _, n = pcu.load_mesh_vfn(full_name, dtype=np.float32)
            v, idx, _ = pcu.deduplicate_point_cloud(v, 1e-15, return_index=True)  # Deduplicate point cloud when loading it
            n = n[idx]
            mask = np.linalg.norm(n, axis=-1) > min_norm_normal
            # Keep the good points and normals
            x = v[mask].astype(np.float32)
            n = n[mask].astype(np.float32)
            n /= np.linalg.norm(n, axis=-1, keepdims=True)
            self.train_xx.append(torch.tensor(x))
            self.train_nn.append(torch.tensor(n))
        
        big_pc = torch.cat(self.train_xx,dim=0)
        self.mean = big_pc.mean(0,keepdim=True)#
        self.scale = 1.0/torch.max(torch.abs(big_pc - self.mean))
        #self.scale = (1.0/torch.mean(torch.abs(big_pc - self.mean),dim=0,keepdim=True))
        big_pc = (big_pc-self.mean) * self.scale
        self.bb_min = big_pc.min(0,keepdim=True)[0]  #block_min
        self.bb_max = big_pc.max(0,keepdim=True)[0]  #block_max
        del big_pc
        for idx in range(int(self.lens)):
            self.train_xx[idx] = (self.train_xx[idx] - self.mean) * self.scale
            
        # prepare sigmas for computing
        """
        self.sm_sigmas = []
        self.big_sigmas = []
        for idx in range(int(self.lens)):
            self.train_xx[idx] = (self.train_xx[idx] - self.mean) * self.scale
            x_nn = knn_points(self.train_xx[idx].unsqueeze(0),self.train_xx[idx].unsqueeze(0),K=11)
            self.sm_sigmas.append(((x_nn.dists[0][:,-1])**0.5).unsqueeze(1).repeat(1,3))
            self.big_sigmas.append(0.3 * torch.ones_like((self.train_xx[idx])))
        """
        # if eval_idx != None
        if self.eval_idx != None:
            self.eval_names = []
            for j in self.eval_idx:
                eval_name = pc_path + obj_name+"_"+str(j)+".ply"
                self.eval_names.append(eval_name)
    def __len__(self):
        return self.lens
    def evaluate_train(self,idx):
        pts = self.train_xx[idx]
        return pts
    
    def __getitem__(self, idx):
        raw_pts = self.train_xx[idx]
        raw_nms = self.train_nn[idx]
        
        new_pts, new_idx =  sfp(raw_pts[None],K=1000)

        normalized_time = self.normalized_times[idx]
        return {"pts": new_pts[0], "nms":raw_nms[new_idx[0]],"normed_time":normalized_time}