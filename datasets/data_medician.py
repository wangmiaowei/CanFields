import numpy as np
import os
import random
import torch
import point_cloud_utils as pcu

class DeformSamples(torch.utils.data.Dataset):
    def __init__(self,filename,train_step = 1):
        """
        train_step: 2,3,4
        """
        pc_path = filename+"/4k_pc/"
        sub_files = os.listdir(pc_path)
        sub_files = sorted(sub_files, key=lambda x: int((x.split('.')[0]).split('_')[1]))
        print('sub_files: ',sub_files)
        total_lens = len(sub_files) #total lens of dataset

        self.train_xx = []
        self.train_nn= []
        self.normalized_times = []

        min_norm_normal=1e-5
        train_lens = 0

        for i in range(0,total_lens,int(train_step)):
            self.normalized_times.append(torch.tensor(i/total_lens))

            sub_file = sub_files[i]
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
            train_lens = train_lens +1
        self.lens = train_lens

        big_pc = torch.cat(self.train_xx,dim=0)
        self.mean = big_pc.mean(0,keepdim=True)#
        self.scale = 1.0/torch.max(torch.abs(big_pc - self.mean))
        big_pc = (big_pc-self.mean) * self.scale
        self.bb_min = big_pc.min(0,keepdim=True)[0]  #block_min
        self.bb_max = big_pc.max(0,keepdim=True)[0]  #block_max
        del big_pc
        for idx in range(int(self.lens)):
            self.train_xx[idx] = (self.train_xx[idx] - self.mean) * self.scale
    def __len__(self):
        return self.lens
    def evaluate_train(self,idx):
        pts = self.train_xx[idx]
        return pts
    def __getitem__(self,idx):
        pts = self.train_xx[idx]
        nms = self.train_nn[idx]
        normalized_time = self.normalized_times[idx]
        return {"pts": pts, "nms":nms,"normed_time":normalized_time}
if __name__ == "__main__":
    path = None
    dataset = DeformSamples(path,train_step = 1)
    print('lens: ',len(dataset))
    print('lens: ',dataset.normalized_times)
    dps = dataset[0]
    pts = dps["pts"]#
    normalized_time = dps["normed_time"]
    print('pts.shape: ',pts.shape)
    print('normalzied_time: ',normalized_time)
    