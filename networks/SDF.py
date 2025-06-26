import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class SdfDecoder(nn.Module):
    def __init__(
        self,
        dims,
        latent_size=256,
        dropout=None,
        dropout_prob=0.2,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        geometric_init = True,
        input_dims = 3,
    ):
        super(SdfDecoder, self).__init__()
        dims = [input_dims] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in #[4]
        self.xyz_in_all = xyz_in_all #false
        self.weight_norm = weight_norm
        self.max_frequency = 10
        bias = 1.0
        #self.th = nn.Tanh()
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0] #253
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3
            lin = nn.Linear(dims[layer], out_dim)

            if layer == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(lin),
                )
            else:
                setattr(self, "lin" + str(layer), lin)
            self.act = nn.Softplus(beta=100)
            self.dropout_prob = dropout_prob #0.2
            self.dropout = dropout

    def forward_unit(self,input,time=None):
        if time!=None:
            input = torch.cat([torch.broadcast_to(time, (input.shape[0],1)),input],dim=-1)
            #input = torch.cat([torch.broadcast_to(2*(time-0.5), (input.shape[0],1)),input],dim=-1)
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act(x)
            if self.dropout is not None and layer in self.dropout:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x
    def evaluate(self,input,time=None):
        if time!=None:
            input = torch.cat([torch.broadcast_to(time, (input.shape[0],1)),input],dim=-1)
            #input = torch.cat([torch.broadcast_to(2*(time-0.5), (input.shape[0],1)),input],dim=-1)
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act(x)
            if self.dropout is not None and layer in self.dropout:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x
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

class SdfDecoder_debug(nn.Module):
    def __init__(
        self,
        dims,
        latent_size=256,
        dropout=None,
        dropout_prob=0.2,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        geometric_init = True
    ):
        super(SdfDecoder_debug, self).__init__()
        dims = [80] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in #[4]
        self.xyz_in_all = xyz_in_all #false
        self.weight_norm = weight_norm
        self.max_frequency = 10

        mvn = torch.distributions.MultivariateNormal(torch.zeros(3), 
                                                     covariance_matrix=((1.5*torch.pi*2)**2) * torch.eye(3))
        # Sample from the distribution
        self.samples = mvn.sample(torch.Size([40])).cuda()
        bias = 1.0
        #self.th = nn.Tanh()
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0] #253
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3
            lin = nn.Linear(dims[layer], out_dim)

            if layer == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(lin),
                )
            else:
                setattr(self, "lin" + str(layer), lin)
            self.act = nn.Softplus(beta=100)
            self.dropout_prob = dropout_prob #0.2
            self.dropout = dropout

    def pos_encode(self,cxyz):
        pos_cos = torch.cos(cxyz@self.samples.T).unsqueeze(1) #100,10
        pos_sin = torch.sin(cxyz@self.samples.T).unsqueeze(1)
        pos_enc = torch.cat([pos_cos,pos_sin],dim=1).transpose(1,2).reshape(-1,80)
        return pos_enc
    def forward_unit(self,input):
        input = self.pos_encode(input)
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act(x)
            if self.dropout is not None and layer in self.dropout:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x
    def evaluate(self,input):

        input = self.pos_encode(input)

        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act(x)
            if self.dropout is not None and layer in self.dropout:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x
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
    

if __name__ == "__main__":
    import json
    specs_filename = "configs/specs.json"
    specs = json.load(open(specs_filename))["NetworkSpecs"]

    decoder_args = specs['decoder_kargs']
    sdf_net = SdfDecoder(**decoder_args)
    print('sdf_net: ',sdf_net)