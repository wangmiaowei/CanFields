#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
#import pytorch3d.ops.sample_farthest_points as sfp
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
        min_norm_normal=1e-5,
        start_idx = 0,
        gap = 16,
        stride = 1
    ):
        #obj_name = filename.split('/')[-1]
        pc_path = filename+"/4k_pc/"
        sub_files = os.listdir(pc_path)
        sub_files = sorted(sub_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[start_idx:start_idx+gap:stride]

        self.lens = len(sub_files)
        self.sub_files = sub_files
        self.train_xx = []
        self.train_nn= []
        self.normalized_times = []
        frame_idx = 0
        for sub_file in sub_files:
            self.normalized_times.append(torch.tensor(frame_idx/self.lens))
            full_name = pc_path + sub_file

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
            frame_idx = frame_idx+1
        
        big_pc = torch.cat(self.train_xx,dim=0)
        self.mean = big_pc.mean(0,keepdim=True)#
        self.scale = 1.0/torch.max(torch.abs(big_pc - self.mean))

        big_pc = (big_pc-self.mean) * self.scale
        self.bb_min = big_pc.min(0,keepdim=True)[0]  #block_min
        self.bb_max = big_pc.max(0,keepdim=True)[0]  #block_max
        del big_pc
        for idx in range(int(self.lens)):
            self.train_xx[idx] = (self.train_xx[idx] - self.mean) * self.scale
        
            new_indices = torch.randperm(len(self.train_xx[idx]))[:8000]
            new_pts = (self.train_xx[idx])[new_indices]
            new_nms = (self.train_nn[idx])[new_indices]

            #new_pts, new_idx =  sfp((self.train_xx[idx])[None],K=8000)
            self.train_xx[idx] = new_pts
            self.train_nn[idx] = new_nms



    def __len__(self):
        return self.lens
    def evaluate_train(self,idx):
        pts = self.train_xx[idx]
        return pts
    
    def __getitem__(self, idx):
        raw_pts = self.train_xx[idx]
        raw_nms = self.train_nn[idx]
        new_indices = torch.randperm(len(raw_pts))[:2000]
        new_pts = raw_pts[new_indices]
        new_nms = raw_nms[new_indices]
        #new_pts, new_idx =  sfp(raw_pts[None],K=2000)
        
        normalized_time = self.normalized_times[idx]
        return {"pts": new_pts, "nms":new_nms,"normed_time":normalized_time}
    
if __name__ == "__main__":
    from pytorch3d.io import IO
    from pytorch3d.structures.pointclouds import Pointclouds
    path = None
    dataset = DeformSamples(path,start_idx = 6,gap = 27,stride=3)
    print("lens(:",len(dataset))
    print('normalized_times: ',dataset.normalized_times)
