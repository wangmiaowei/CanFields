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
import pytorch3d
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
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
        pc_path = filename+"/gt_mesh/"


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
        self.train_faces = []
        self.normalized_times = []

        for j in self.train_idx:
            self.normalized_times.append(torch.tensor(j/total_lens))
            full_name = pc_path + obj_name+"_"+str(j)+".ply"
            v, f, _ = pcu.load_mesh_vfn(full_name, dtype=np.float32)
            #n /= np.linalg.norm(n, axis=-1, keepdims=True)
            self.train_xx.append(torch.tensor(v))
            #self.train_nn.append(torch.tensor(n))
            self.train_faces.append(torch.tensor(f))

        big_pc = torch.cat(self.train_xx,dim=0)
        self.mean = big_pc.mean(0,keepdim=True)
        self.scale = 1.0/torch.mean(torch.abs(big_pc - self.mean))

        big_pc = (big_pc-self.mean) * self.scale
        self.bb_min = big_pc.min(0,keepdim=True)[0]  #block_min
        self.bb_max = big_pc.max(0,keepdim=True)[0]  #block_max
        del big_pc

        for idx in range(int(self.lens)):
            self.train_xx[idx] = (self.train_xx[idx] - self.mean) * self.scale


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
    """
    def intermediate_train(self):
        #median len()//2
        pts = self.train_xx[int(self.lens//2)]
        pts_mean = pts.mean(0,keepdim=True)#
        pts_scale = (1.0/torch.mean(torch.abs(pts - pts_mean),dim=0,keepdim=True))
        return pts_mean,pts_scale
    """
    def __getitem__(self, idx):
        mesh = Meshes(verts=[self.train_xx[idx]],faces=[self.train_faces[idx]])
        #pytorch3d.io.save_ply("see_results.ply", mesh.verts_list()[0],mesh.faces_list()[0])
        normalized_time = self.normalized_times[idx]
        samples,normals = sample_points_from_meshes(mesh,return_normals=True)
        return {"pts":samples,"nms":normals,"normed_time":normalized_time}
    
if __name__ == "__main__":
    from pytorch3d.io import IO
    from pytorch3d.structures.pointclouds import Pointclouds
    path = r"/exports/csce/eddie/inf/groups/IPAB_CGCV/dataset/Deforming4D/notexture/canieLTT_Idles1"
    dataset = DeformSamples(path,train_rate=0.3)
    pts = dataset[0]
    print('pts[dps]: ',pts["pts"].shape)
    print('pts[dps]: ',pts["nms"].shape)