import torch
from torch import nn
from torchmeta.modules import MetaModule
import numpy as np

def det3(M):
    det=M[...,0,0]*M[...,1,1]*M[...,2,2]\
        +M[...,0,1]*M[...,1,2]*M[...,2,0]\
        +M[...,0,2]*M[...,1,0]*M[...,2,1]\
        -M[...,0,2]*M[...,1,1]*M[...,2,0]\
        -M[...,0,1]*M[...,1,0]*M[...,2,2]\
        -M[...,0,0]*M[...,1,2]*M[...,2,1]
    return det


class Corresponding_Points_Alignment_Loss(torch.autograd.Function):
    # high oreder gradients are NOT considered!
    @staticmethod
    def forward(ctx, X:torch.Tensor, Y:torch.Tensor, weights:torch.Tensor, eps_forward:float, eps_backward:float):
        """
        X:(*,N,3)
        Y:(*,N,3)
        weight:(*,P,N)
        """
        X_ = X.unsqueeze(-3)
        Y_ = Y.unsqueeze(-3)
        sum_weight = weights[..., None].sum(dim=-2,keepdim=True)
        Xmu = (X_ * weights[..., None]).sum(dim=-2,keepdim=True) / sum_weight.clamp(eps_forward)
        Ymu = (Y_ * weights[..., None]).sum(dim=-2,keepdim=True) / sum_weight.clamp(eps_forward)
        Xc = X_ - Xmu
        Yc = Y_ - Ymu
        XYcov = (Xc*weights.unsqueeze(-1)).transpose(-1, -2)@ Yc

        U, S, VT = torch.linalg.svd(XYcov)
        f1 = torch.ones_like(S)
        f2 = torch.ones_like(S)
        f2[...,-1]=-1
        f = torch.where((det3(XYcov)>0).unsqueeze(-1),f1,f2)
        F = torch.diag_embed(f)
        S, R = S*f, U @ F @VT
        ctx.S, ctx.R, ctx.sum_weight = S, R, sum_weight
        ctx.eps_forward,ctx.eps_backward = eps_forward, eps_backward
        ctx.Xc, ctx.Yc = Xc, Yc
        ctx.save_for_backward(weights)
        term1=torch.sum(torch.sum(Xc**2+Yc**2,dim=-1)*weights,dim=-1)
        term2=torch.sum(S,dim=-1)
        return term1 - 2 * term2

    @staticmethod
    def backward(ctx, grad_outputs):
        with torch.no_grad():
            Xc, Yc = ctx.Xc, ctx.Yc
            weights = ctx.saved_tensors[0]
            grad_XYcov = -2 * ctx.R
            wXc = Xc * weights.unsqueeze(-1)
            wYc = Yc * weights.unsqueeze(-1)
            grad_Xc = grad_outputs[...,None,None] * (2 * wXc + wYc @ grad_XYcov.transpose(-1,-2)) # (*,P,N,3)
            grad_Yc = grad_outputs[...,None,None] * (2 * wYc + wXc @ grad_XYcov) # (*,P,N,3)
            sum_weight = ctx.sum_weight# (*,P,1,1)
            eps_forward, eps_backward = ctx.eps_forward, ctx.eps_backward
            grad_X = grad_Xc - grad_Xc.sum(dim=-2,keepdim=True) * weights.unsqueeze(-1)/sum_weight.clamp(eps_forward)
            grad_Y = grad_Yc - grad_Yc.sum(dim=-2,keepdim=True) * weights.unsqueeze(-1)/sum_weight.clamp(eps_forward)
            grad_input_X=grad_X.sum(dim=-3)
            grad_input_Y=grad_Y.sum(dim=-3)
            grad_input_w = grad_outputs[...,None] * ((Xc @ grad_XYcov) * Yc + Xc**2+Yc**2).sum(dim=-1)\
                        -((grad_Xc.sum(dim=-2,keepdim=True) * Xc + grad_Yc.sum(dim=-2,keepdim=True) * Yc)/sum_weight.clamp(eps_backward)).sum(dim=-1)\
                        
        return grad_input_X, grad_input_Y, grad_input_w, None, None
class Sine(nn.Module):
    def __init__(self, omega = 30):
        super().__init__()
        self.omega = omega

    def forward(self, input):
        return torch.sin(self.omega * input)
class BatchLinear(nn.Linear,MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.
    '''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):

        if params is None:
            return nn.Linear.forward(self,input)

        else:

            bias = params.get('bias', None)
            weight = params['weight']

            output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
            output += bias.unsqueeze(-2)
            return output
        
def sine_init(m,freq=30):
    with torch.no_grad():
        if type(m) == BatchLinear or type(m) == nn.Linear:
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
        if type(m) == nn.Conv1d:
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-2)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)              
class PartNet(nn.Module):
    def __init__(self, hidden_features,out_features,num_hidden_layers, in_features=3,initial_first = False):
        super().__init__()
        freq=15
        nl = Sine(freq)
        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features, hidden_features), nl
        ))

        for _ in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))

        self.net.append(nn.Sequential(
            nn.Linear(hidden_features, out_features)
        ))

        self.net = nn.Sequential(*self.net)
        self.apply(lambda m:sine_init(m,freq))
        if initial_first: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_sine_init)
    def get_part_id(self,coords):
        part_prob=self(coords)
        part_prob=torch.softmax(part_prob,dim=-1)
        val,part_id=part_prob.max(dim=-1)
        return part_id
    def forward(self, coords, **kwargs):
        output = self.net(coords)
        return output
def corresponding_points_alignment_loss(X:torch.Tensor, Y:torch.Tensor, weights:torch.Tensor, eps_forward:float=1e-7, eps_backward:float=1e-7):
    # The backward process suffers from small weights very much
    return Corresponding_Points_Alignment_Loss.apply(X,Y,weights,eps_forward,eps_backward)
if __name__ == "__main__":
    part_field = PartNet(hidden_features=128,out_features=20,in_features=3,num_hidden_layers=1,initial_first=True).cuda()
    old_locs = torch.rand((50,3)).cuda()
    new_locs = torch.rand((50,3)).cuda()
    target_template_part_prob=torch.softmax(part_field(new_locs),dim=-1)
    print('target_temp: ',(target_template_part_prob.transpose(-1,-2)).shape)
    alignment_loss =corresponding_points_alignment_loss(new_locs.unsqueeze(0),old_locs.unsqueeze(0),(target_template_part_prob.transpose(-1,-2)).unsqueeze(0),eps_backward = 1e-7).sum(dim=-1)#/points_num
    print('aligns: ',alignment_loss)