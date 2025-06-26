#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import point_cloud_utils as pcu
from pytorch3d.ops.knn import knn_gather, knn_points
#import deep_sdf.workspace as ws

class DeformSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        with_missing=False,
        train_rate = 1, 
        min_norm_normal=1e-5
    ):

        pc_path = None
        total_lens = len(os.listdir(pc_path))
        self.lens = total_lens

        obj_name = "foxXAT_Rotate180R"
        self.train_xx = []
        self.train_nn= []
        self.normalized_times = []


        for j in range(total_lens):
            self.normalized_times.append(torch.tensor(j/total_lens))
            full_name = pc_path + obj_name+"_"+str(5*j)+".ply"
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
        self.scale = (1.0/torch.mean(torch.abs(big_pc - self.mean),dim=0,keepdim=True))
        big_pc = (big_pc-self.mean) * self.scale
        self.bb_min = big_pc.min(0,keepdim=True)[0]  #block_min
        self.bb_max = big_pc.max(0,keepdim=True)[0]  #block_max
        del big_pc

        
        # prepare sigmas for computing
        self.sm_sigmas = []
        self.big_sigmas = []
        for idx in range(self.lens):
            self.train_xx[idx] = (self.train_xx[idx] - self.mean) * self.scale
            x_nn = knn_points(self.train_xx[idx].unsqueeze(0),self.train_xx[idx].unsqueeze(0),K=11)
            self.sm_sigmas.append(((x_nn.dists[0][:,-1])**0.5).unsqueeze(1).repeat(1,3))
            self.big_sigmas.append(0.3 * torch.ones_like((self.train_xx[idx])))


    def __len__(self):
        return self.lens
    def evaluate_train(self,idx):
        pts = self.train_xx[idx]
        return pts
    
    def __getitem__(self, idx):
        pts = self.train_xx[idx]
        nms = self.train_nn[idx]
        normalized_time = self.normalized_times[idx]
        #return {"pts": pts, "nms":nms,"normed_time":normalized_time}
        sm_sigma = self.sm_sigmas[idx]
        big_sigma = self.big_sigmas[idx]

        sald_sps = torch.cat([pts + torch.normal(mean=0.0,std = sm_sigma),
                              pts + torch.normal(mean=0.0,std = sm_sigma),
                              pts + torch.normal(mean=0.0,std = big_sigma),
                              pts + torch.normal(mean=0.0,std = big_sigma)],dim=0)
        
        non_mnpts = torch.rand(20000,3) * (self.bb_max - self.bb_min) +self.bb_min
        
        return {"pts": pts, "nms":nms,"non_mnpts":non_mnpts,"sald_sps":sald_sps,"normed_time":normalized_time}
    
if __name__ == "__main__":
    from pytorch3d.io import IO
    from pytorch3d.structures.pointclouds import Pointclouds
    dataset = DeformSamples()
    dps = dataset[0]
    print('pts.shape: ',dps["pts"].shape)
    print('nms.shape: ',dps["nms"].shape)
