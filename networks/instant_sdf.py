import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == 'frequency':
        #encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sphere_harmonics':
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from .gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
    
    elif encoding == 'tiledgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)
    
    elif encoding == 'ash':
        from ashencoder import AshEncoder
        encoder = AshEncoder(input_dim=input_dim, output_dim=16, log2_hashmap_size=log2_hashmap_size, resolution=desired_resolution)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim


class SDFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf

        self.encoder, self.in_dim = get_encoder(encoding)

        backbone = []

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            elif l in self.skips:
                in_dim = self.hidden_dim + self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))

        self.backbone = nn.ModuleList(backbone)

    def forward(self,inputs):
        outputs = {}
        outputs["gt_nms"] = inputs['nms'].cuda()
        outputs['pts'] = inputs["pts"].cuda()
        outputs['pts'].requires_grad = True
        #outputs["latent_code"] = temp_shape_code
        # on surface points compute 
        outputs['mns_preds']  = self.forward_unit(outputs['pts'])
        # near surface points compute
        outputs['near_pts'] = inputs["sald_sps"].cuda()
        outputs['near_pts'].requires_grad = True
        outputs['near_preds']  = self.forward_unit(outputs['near_pts'])

        # ELk uniform sampling predict
        outputs['uniform_pts'] = inputs["non_mnpts"].cuda()
        outputs['uniform_pts'].requires_grad = True
        outputs['uniform_preds']  = self.forward_unit(outputs['uniform_pts'])
        return outputs
    def evaluate(self,x):
        return self.forward_unit(x)
    
    def forward_unit(self, x):
        # x: [B, 3]

        x = self.encoder(x)

        h = x
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h
    
if __name__ == "__main__":
    model = SDFNetwork(encoding="hashgrid").cuda()
    g = torch.rand((1000,3)).cuda()
    outputs = model(g).max()

    print('model: ',model)
    print('output: ',outputs)