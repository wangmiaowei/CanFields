#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import logging
import torch.nn as nn
import torch
import numpy as np
import torch.autograd as ag
from .of4d_velo import *
from .other_volocities import  VelocityAABBSur
#from .SDF import SdfDecoder_debug
from .SDF import SdfDecoder
from .DIGS import DiGSNetwork
#from torchdiffeq import odeint_adjoint as odeint
from .part_seg import PartNet
from torchdiffeq import odeint
from .modules import fit_latent

from .rigid_body import exp_so3
class MLP(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.pts_linears = nn.ModuleList( [nn.Linear(width, width) for i in range(depth - 1)])
        self.relu = nn.Softplus(beta=100)
    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = self.relu(x)
        return x
    
class adjust_net(nn.Module):
    def __init__(self):
        super().__init__()
        #self.embeddings = None
        self.input= nn.Sequential( nn.Linear(4,512), nn.Softplus(beta=100))

        self.cond = nn.Linear(128, 512)

        self.mlp = MLP(depth=3,width=512)
        self.trn_branch = nn.Linear(512, 3)
        
        self.rot_branch = nn.Linear(512, 3)

        self.prob_branch = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.mlp_scale = 1
        self.tanh = nn.Tanh()
        self._reset_parameters()
    def get_Rotation (self, fea):
        R = self.mlp_scale * self.rot_branch( fea )
        theta = torch.norm(R, dim=-1, keepdim=True)
        w = R / theta
        R = exp_so3(w, theta)
        return R

    def pos_encode(self,cxyz):
        pos_cos = torch.cos(cxyz@self.samples.T).unsqueeze(1) #100,10
        pos_sin = torch.sin(cxyz@self.samples.T).unsqueeze(1)
        pos_enc = torch.cat([pos_cos,pos_sin],dim=1).transpose(1,2).reshape(-1,20)
        return pos_enc
    
    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prob_branch.weight)
        torch.nn.init.xavier_uniform_(self.rot_branch.weight)
        torch.nn.init.constant_(self.cond.weight, 0.00)  
        torch.nn.init.constant_(self.cond.bias, 0.00)    
        
        for p in self.input.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self,src_normed_time,cxyz,embeddings):
        encoded_pos = torch.cat([torch.broadcast_to(src_normed_time, (cxyz.shape[0],1)),cxyz],dim=-1)
        fea = self.input(encoded_pos)

        shape_features =  self.tanh(self.cond(embeddings)) 

        feas = fea*(1+shape_features)


        fea = self.mlp(feas) + feas
         
        delta_t = self.mlp_scale * self.trn_branch(fea)

        R = self.get_Rotation(fea)
        x_ = (R @ cxyz[..., None]).squeeze() + delta_t

        proba =self.sigmoid( self.mlp_scale * self.prob_branch(fea) )
        outputs =  x_.squeeze()
        return outputs,1-proba
class ODEFunc(nn.Module):
    '''
    This refers to the dynamics function f(x,t) in a IVP defined as dh(x,t)/dt = f(x,t). 
    For a given location (t) on point (x) trajectory, it returns the direction of 'flow'.
    Refer to Section 3 (Dynamics Equation) in the paper for details. 
    '''
    def __init__(self, hidden_size, latent_size):
        '''
        Initialization. 
        num_hidden: number of nodes in a hidden layer
        latent_len: size of the latent code being used
        '''
        super(ODEFunc, self).__init__()
        self.l1 = nn.Linear(4,hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.cond = nn.Linear(latent_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 3)

        self.compute_gradient = False
        self.tanh = nn.Tanh()
        self.relu = nn.Softplus(beta=100) #nn.ReLU()
        #self.embeddings = None
        self.nfe = 0
        self.grads = []
    def create(self,embeddings):
        self.embeddings = embeddings
    def df(self,x,wrt):
        M = x.shape[0]
        return ag.grad(x.flatten(), wrt,
                        grad_outputs=torch.ones(M).to(x.dtype).to(wrt.device),create_graph=True)[0]
    
    def forward(self, t, cxyz):
        '''
        t: Torch tensor of shape (1,) 
        cxyz: Torch tensor of shape (N, zdim+3). Along dimension 1, the point and shape embeddings are concatenated. 
        
        **NOTE**
        For the uniqueness property to hold, a single dynamics function (operating in 3D) must be used to compute 
        trajectories pertaining to points of a single shape. 
        
        Here, the shape encoding (same for all points of a shape) is used to choose a function which is applied over all the shape points.
        Hence, even though the input xz appears to be a 3+zdim dimensional state, the ODE is still restricted to a 3D state-space. 
        The concatenation is purely to make programming simpler without affecting the underlying theory. 
        
        '''
        input = torch.cat([torch.broadcast_to(t-0.5, (cxyz.shape[0],1)),cxyz],dim=-1)
        point_features = self.relu(self.l1(input))
        shape_features = self.tanh(self.cond(fit_latent(self.embeddings.weight,t)))  
        
        point_shape_features = point_features*shape_features  # Compute point-shape features by elementwise multiplication

        point_shape_features = self.relu(self.l2(point_shape_features)) + point_shape_features
        dyns_x_t = self.l4(point_shape_features)
        if self.compute_gradient:
            total_dxyz_dxyz=[]
            for o in range(3):
                derivs = self.df(dyns_x_t[:,o],input)
                total_dxyz_dxyz.append(derivs.unsqueeze(1))
            total_dxyz_dxyz = torch.cat(total_dxyz_dxyz,dim=1)
            self.grads.append(total_dxyz_dxyz)
        self.nfe+=1  #To check #ode evaluations
        return dyns_x_t
    
class MLPWithSkipConnection(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=3, num_layers=4):
        super(MLPWithSkipConnection, self).__init__()
        self.nfe = 0
        self.compute_gradient = False
        self.grads = []

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.skip_connections = nn.ModuleList([
            nn.Linear(input_dim + i * hidden_dim, hidden_dim) for i in range(1, num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def create(self,embeddings):
        self.embeddings = embeddings
    def count_learned(self):
        return sum(p.numel() for p in self.pyramid.parameters() if p.requires_grad)
    def df(self,x,wrt):
        M = x.shape[0]
        return ag.grad(x.flatten(), wrt,
                        grad_outputs=torch.ones(M).to(x.dtype).to(wrt.device),create_graph=True)[0]
    def forward(self, t,xyz):
        x = torch.cat([torch.broadcast_to(t-0.5, (xyz.shape[0],1)),xyz],dim=-1)

        out = self.activation(self.input_layer(x))
        skip_connection_inputs = [x]

        for i in range(len(self.hidden_layers)):
            out = self.activation(self.hidden_layers[i](out))
            skip_connection_inputs.append(out)
            skip_connection_input = torch.cat(skip_connection_inputs, dim=1)
            out = out + self.activation(self.skip_connections[i](skip_connection_input))

        out = self.output_layer(out)
        if self.compute_gradient:
            total_dxyz_dxyz=[]
            for o in range(3):
                derivs = self.df(out[:,o],x)
                total_dxyz_dxyz.append(derivs.unsqueeze(1))
            total_dxyz_dxyz = torch.cat(total_dxyz_dxyz,dim=1)
            self.grads.append(total_dxyz_dxyz)
        self.nfe+=1  #To check #ode evaluations
        return out


class VelocityLayer(nn.Module):
    '''
    This refers to the dynamics function f(x,t) in a IVP defined as dh(x,t)/dt = f(x,t). 
    For a given location (t) on point (x) trajectory, it returns the direction of 'flow'.
    Refer to Section 3 (Dynamics Equation) in the paper for details. 
    '''
    def __init__(self):
        '''
        Initialization. 
        num_hidden: number of nodes in a hidden layer
        latent_len: size of the latent code being used
        '''
        super(VelocityLayer, self).__init__()
        self.max_frequency = 5
        self.hidden_features = 512
        self.l1 = nn.Linear(40,self.hidden_features)

        self.l2 = nn.Linear(self.hidden_features,self.hidden_features)

        self.output_layer = nn.Linear(self.hidden_features, 3)
        self.relu = nn.Softplus(beta=100) #nn.ReLU()
        self.nfe = 0
        self.compute_gradient = False
        self.grads = []
        mvn = torch.distributions.MultivariateNormal(torch.zeros(4), covariance_matrix=((0.4*torch.pi*2)**2) * torch.eye(4))
        # Sample from the distribution
        self.samples = mvn.sample(torch.Size([20])).cuda()
    def create(self,embeddings):
        self.embeddings = embeddings
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def df(self,x,wrt):
        M = x.shape[0]
        return ag.grad(x.flatten(), wrt,
                        grad_outputs=torch.ones(M).to(x.dtype).to(wrt.device),create_graph=True)[0]
    
    def count_learned(self):
        return sum(p.numel() for p in self.pyramid.parameters() if p.requires_grad)
    def pos_encode(self,cxyz):
        cxyz = cxyz.unsqueeze(1)
        gcs = torch.matmul(cxyz, self.samples.T)
        pos_sin = torch.sin(gcs)
        pos_cos = torch.cos(gcs)
        pos_enc = torch.cat([pos_cos, pos_sin], dim=2)
        return pos_enc.view(pos_enc.size(0), -1)
    
    def forward(self,t,xyz):
        input = torch.cat([torch.broadcast_to((t-0.5), (xyz.shape[0],1)),xyz],dim=-1)
        #################################Pose Encoding #########################
        post = self.pos_encode(input)
        #########################Pose Encoding Ending############################
        point_features = self.relu(self.l1(post))# N*8

        new_point_features = 0.5*self.relu(self.l2(point_features))+ 0.5*point_features
        dyns_x_t = self.output_layer(new_point_features)

        if self.compute_gradient:
            total_dxyz_dxyz=[]
            for o in range(3):
                derivs = self.df(dyns_x_t[:,o],input)
                total_dxyz_dxyz.append(derivs.unsqueeze(1))
            total_dxyz_dxyz = torch.cat(total_dxyz_dxyz,dim=1)
            self.grads.append(total_dxyz_dxyz)
        self.nfe+=1  #To check #ode evaluations
        return dyns_x_t


class NODEBlock(nn.Module):
    '''
    Function to solve an IVP defined as dh(x,t)/dt = f(x,t). 
    We use the differentiable ODE Solver by Chen et.al used in their NeuralODE paper.
    '''
    def __init__(self, odefunc, tol):
        '''
        Initialization. 
        odefunc: The dynamics function to be used for solving IVP
        tol: tolerance of the ODESolver
        '''
        super(NODEBlock, self).__init__()
        self.odefunc = odefunc
        self.cost = 0
        self.rtol = tol
        self.atol = tol

        
    def forward(self, cxyz, src_time,tgt_time):
        '''
        Solves the ODE in the forward / reverse time. 
        '''
        if src_time.to(torch.float32) == tgt_time.to(torch.float32):
            return cxyz
        else:
            self.odefunc.nfe = 0  #To check #ode evaluations
            
            self.times = torch.tensor([src_time,tgt_time]).to(cxyz)  # Time of integration (must be monotinically increasing!)
            # Solve the ODE with initial condition x and interval time.
            out = odeint(self.odefunc, cxyz, self.times, rtol = self.rtol, atol = self.rtol)
            
            self.cost = self.odefunc.nfe  # Number of evaluations it took to solve it
            
            return out[1]

class Warper_multi(nn.Module):
    '''
    A single DeformBlock is made up of two NODE Blocks. Refer secion 3 (Overall Architecture)
    '''
    def __init__(self, 
                 latent_size,
                 hidden_size, 
                 steps,
                 time=1.0,
                 tol = 1e-5):
        super(Warper_multi, self).__init__()
        '''
        Initialization.
        time: some number 0-1
        num_hidden: Number of hidden nodes in the MLP of dynamics
        latent_len: Length of shape embeddings
        tol: tolerance of the ODE Solver
        '''
        
        # Several NODE Blocks
        self.integral1 = NODEBlock(ODEFunc(hidden_size, latent_size), tol = tol)
        self.integral2 = NODEBlock(ODEFunc(hidden_size, latent_size), tol = tol)
        
        self.integral3 = NODEBlock(ODEFunc(hidden_size, latent_size), tol = tol)
        self.integral4 = NODEBlock(ODEFunc(hidden_size, latent_size), tol = tol)

        self.start_time = None
        self.end_time = None
        self.median1 = None
        self.median2 = None
        self.median3 = None
    
    def equip_embedding(self,embeddings):
        self.integral1.odefunc.create(embeddings)
        self.integral2.odefunc.create(embeddings)
        self.integral3.odefunc.create(embeddings)
        self.integral4.odefunc.create(embeddings)
    def unload_embedding(self):
        self.integral1.odefunc.embeddings = None 
        self.integral2.odefunc.embeddings = None 
        self.integral3.odefunc.embeddings = None 
        self.integral4.odefunc.embeddings = None 
    def turon_grad(self):
        self.integral1.odefunc.compute_gradient = True
        self.integral2.odefunc.compute_gradient = True
        self.integral3.odefunc.compute_gradient = True
        self.integral4.odefunc.compute_gradient = True
    def turoff_grad(self):
        self.integral1.odefunc.compute_gradient = False
        self.integral2.odefunc.compute_gradient = False
        self.integral3.odefunc.compute_gradient = False
        self.integral4.odefunc.compute_gradient = False
    def return_grads(self):
        grads1 = self.integral1.odefunc.grads
        grads2 = self.integral2.odefunc.grads
        grads3 = self.integral3.odefunc.grads
        grads4 = self.integral4.odefunc.grads
        return [grads1,grads2,grads3,grads4] 
    def clean_grads(self):
        self.integral1.odefunc.grads = []
        self.integral2.odefunc.grads = []
        self.integral3.odefunc.grads = []
        self.integral4.odefunc.grads = []

    def split_time(self,start_time,end_time):
        self.start_time = start_time.to(torch.float32)
        self.end_time = end_time.to(torch.float32)
        small_seg = ((self.end_time - self.start_time)/4).to(torch.float32)
        self.median1 = self.start_time+small_seg
        self.median2 = self.median1 + small_seg
        self.median3 = self.median2 + small_seg

    def forward(self, inputs, src_time, tgt_time):
        if src_time==tgt_time:
            return inputs
        else:
            smal_t = min(src_time,tgt_time)
            larg_t = max(src_time,tgt_time)
            if smal_t<self.median1:
                if larg_t<self.median1:
                    xyz = self.integral1(inputs,src_time,tgt_time)
                elif larg_t<self.median2:
                    if src_time<tgt_time:
                        xyz = self.integral1(inputs,src_time,self.median1)
                        xyz = self.integral2(xyz,self.median1,tgt_time)
                    elif src_time>tgt_time:
                        xyz = self.integral2(inputs,src_time,self.median1)
                        xyz = self.integral1(xyz,self.median1,tgt_time)
                elif larg_t<self.median3:
                    if src_time<tgt_time:
                        xyz = self.integral1(inputs,src_time,self.median1)
                        xyz = self.integral2(xyz,self.median1,self.median2)
                        xyz = self.integral3(xyz,self.median2,tgt_time)
                    elif src_time > tgt_time:
                        xyz = self.integral3(inputs,src_time,self.median2)
                        xyz = self.integral2(xyz,self.median2,self.median1)
                        xyz = self.integral1(xyz,self.median1,tgt_time)
                else:
                    if src_time<tgt_time:
                        xyz = self.integral1(inputs,src_time,self.median1)
                        xyz = self.integral2(xyz,self.median1,self.median2)
                        xyz = self.integral3(xyz,self.median2,self.median3)
                        xyz = self.integral4(xyz,self.median3,tgt_time)
                    elif src_time>tgt_time:
                        xyz = self.integral4(inputs,src_time,self.median3)
                        xyz = self.integral3(xyz,self.median3,self.median2)
                        xyz = self.integral2(xyz,self.median2,self.median1)
                        xyz = self.integral1(xyz,self.median1,tgt_time)
            elif smal_t<self.median2:
                if larg_t<self.median2:
                    xyz = self.integral2(inputs,src_time,tgt_time)
                elif larg_t<self.median3:
                    if src_time<tgt_time:
                        xyz = self.integral2(inputs,src_time,self.median2)
                        xyz = self.integral3(xyz,self.median2,tgt_time)
                    else:
                        xyz = self.integral3(inputs,src_time,self.median2)
                        xyz = self.integral2(xyz,self.median2,tgt_time)
                else:
                    if src_time<tgt_time:
                        xyz = self.integral2(inputs,src_time,self.median2)
                        xyz = self.integral3(xyz,self.median2,self.median3)
                        xyz = self.integral4(xyz,self.median3,tgt_time)
                    else:
                        xyz = self.integral4(inputs,src_time,self.median3)
                        xyz = self.integral3(xyz,self.median3,self.median2)
                        xyz = self.integral2(xyz,self.median2,tgt_time)
            elif smal_t < self.median3:
                if larg_t<self.median3:
                    xyz = self.integral3(inputs,src_time,tgt_time)
                else:
                    if src_time<tgt_time:
                        xyz = self.integral3(inputs,src_time,self.median3)
                        xyz = self.integral4(xyz,self.median3,tgt_time)
                    else:
                        xyz = self.integral4(inputs,src_time,self.median3)
                        xyz = self.integral3(inputs,self.median3,tgt_time)
            else:
                xyz = self.integral4(inputs,src_time,tgt_time)
            return xyz

class Warper(nn.Module):
    '''
    A single DeformBlock is made up of two NODE Blocks. Refer secion 3 (Overall Architecture)
    '''
    def __init__(self, 
                 latent_size,
                 hidden_size, 
                 steps,
                 time=1.0,
                 tol = 1e-5,
                 use_multi_ode="False"):
        super(Warper, self).__init__()
        '''
        Initialization.
        time: some number 0-1
        num_hidden: Number of hidden nodes in the MLP of dynamics
        latent_len: Length of shape embeddings
        tol: tolerance of the ODE Solver
        '''
        
        if use_multi_ode=="True":
            self.integral1 = NODEBlock(ODEFunc(hidden_size, latent_size), tol = tol)
        else:
            self.integral1 = NODEBlock(VelocityLayer(), tol = tol)
    
    def equip_embedding(self,embeddings):
        self.integral1.odefunc.create(embeddings)
    def unload_embedding(self):
        self.integral1.odefunc.embeddings = None 
    def return_grads(self):
        new_grads = self.integral1.odefunc.grads
        return [new_grads] 
    def clean_grads(self):
        self.integral1.odefunc.grads = []
    def split_time(self,start_time,end_time):
        pass

    def forward(self, inputs, src_time, tgt_time):
        if src_time==tgt_time:
            return inputs
        else:
            xyz = self.integral1(inputs,src_time,tgt_time)
            return xyz
    def turon_grad(self):
        self.integral1.odefunc.compute_gradient = True
    def turoff_grad(self):
        self.integral1.odefunc.compute_gradient = False

class Decoder(nn.Module):
    def __init__(self,use_multi_ode,latent_size, warper_kargs, decoder_kargs):
        super(Decoder, self).__init__()
        self.warper = Warper(latent_size, **warper_kargs,use_multi_ode=use_multi_ode)
        self.sdf_decoder = DiGSNetwork(in_dim=3,out_dim=1, 
                                       decoder_hidden_dim=256,nl="sine",
                                       decoder_n_hidden_layers=8, init_type='siren',
                                       sphere_init_params=[1.6,0.1])#.cuda()
    
        self.partseg = PartNet(hidden_features=128,out_features=20,in_features=3,num_hidden_layers=1,initial_first=True)
    
    def setup_level(self,level):
        self.level = level
        self.warper.integral1.odefunc.setup_pyramid(level)
    def forward(self, inputs,temp_time=0.5,requires_grad=True):
        outputs = {}
        
        outputs["gt_nms"] = inputs['nms'].cuda()
        outputs['pts'] = inputs["pts"].cuda()
        if requires_grad ==True:
            self.warper.turon_grad()
        elif requires_grad==False:
            self.warper.turoff_grad()

        p_final  = self.warper(outputs['pts'], inputs["normed_time"],temp_time)
        outputs['mns_preds']  = self.sdf_decoder.forward_unit(p_final)
        part_prob = torch.softmax(self.partseg(p_final),dim=-1)
        outputs['mns_prob'] = part_prob
        outputs['mns_final'] = p_final

        # near surface points compute
        outputs['near_pts'] = inputs["sald_sps"].cuda()
        
        near_final  = self.warper(outputs['near_pts'], inputs["normed_time"],temp_time)
        
        outputs['near_preds']  = self.sdf_decoder.forward_unit(near_final)
        self.warper.turoff_grad()

        # ELk uniform sampling predict
        outputs['uniform_pts'] = inputs["non_mnpts"].cuda()
        uniform_final  = self.warper(outputs['uniform_pts'], inputs["normed_time"],temp_time)
        outputs['uniform_preds']  = self.sdf_decoder.forward_unit(uniform_final)
        return outputs

if __name__ == "__main__":
    inputs = torch.rand((1000,3)).cuda()

    import math
    import json
    specs_filename = r"configs/specs.json"
    specs = json.load(open(specs_filename))
    latent_size = specs["CodeLength"]
    # initialize the weights 
    start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.0).cuda()
    torch.nn.init.normal_(
        start_end_vecs.weight.data,
        0.0,
        1.0 / math.sqrt(256),
    )
    warper = Warper(latent_size,**specs["NetworkSpecs"]["warper_kargs"]).cuda()
    inputs.requires_grad = True
    start_time = torch.tensor(0.1).cuda()
    end_time = torch.tensor(0.9).cuda()
    temp_time = (start_time+end_time)/2
    warper.split_time(start_time,end_time)

    outs = warper(inputs,torch.tensor(0.1).cuda(),temp_time)
    print('outs.shape: ',outs.shape)
