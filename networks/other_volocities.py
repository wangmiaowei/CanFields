import torch.nn as nn
import torch
from torchdiffeq import odeint
import torch.autograd as ag
from .SDF import SdfDecoder
from .part_seg import PartNet
from .of4d_velo import *
from .DIGS import DiGSNetwork
class TPSR(nn.Module):
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
        super(TPSR, self).__init__()
        self.l1 = nn.Linear(3,hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.cond = nn.Linear(latent_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 3)

        self.compute_gradient = False
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
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
        point_features = self.relu(self.l1(cxyz))
        shape_features = self.tanh(self.cond(self.embeddings.weight))  
        
        point_shape_features = point_features*shape_features  # Compute point-shape features by elementwise multiplication

        point_shape_features = self.relu(self.l2(point_shape_features)) + point_shape_features
        dyns_x_t = self.l4(point_shape_features)

        if self.compute_gradient:
            total_dxyz_dxyz=[]
            for o in range(3):
                derivs = self.df(dyns_x_t[:,o],cxyz)
                total_dxyz_dxyz.append(derivs.unsqueeze(1))
            total_dxyz_dxyz = torch.cat(total_dxyz_dxyz,dim=1)
            self.grads.append(total_dxyz_dxyz)
        self.nfe+=1
        return dyns_x_t
    

class NODEBlock(nn.Module):
    '''
    Function to solve an IVP defined as dh(x,t)/dt = f(x,t). 
    We use the differentiable ODE Solver by Chen et.al used in their NeuralODE paper.
    '''
    def __init__(self, odefunc, tol,id_flag=False):
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
        self.id_flag=id_flag
        
    def forward(self, cxyz, src_time,tgt_time):
        '''
        Solves the ODE in the forward / reverse time. 
        '''
        if self.id_flag == True:
            return cxyz
        if src_time.to(torch.float32) == tgt_time.to(torch.float32):
            return cxyz
        else:
            self.odefunc.nfe = 0  #To check #ode evaluations
            
            self.times = torch.tensor([src_time,tgt_time]).to(cxyz)  # Time of integration (must be monotinically increasing!)
            # Solve the ODE with initial condition x and interval time.
            out = odeint(self.odefunc, cxyz, self.times, rtol = self.rtol, atol = self.rtol)
            
            self.cost = self.odefunc.nfe  # Number of evaluations it took to solve it
            
            return out[1]
class VelocityAABBSur(nn.Module):

    def __init__(self):
        super(VelocityAABBSur, self).__init__()
        self.encode_dim = 3
        in_dim = 4 + 4 * 2 * self.encode_dim
        hidden_dim = 128
        self.weight_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        for _ in range(4):
            self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, 6)))
        self.grads = []
        self.compute_gradient = False
        self.nfe = 0
        
        self.frequency_bands = 2.0 ** torch.linspace(
                0.0,
                self.encode_dim - 1,
                self.encode_dim,
                dtype=torch.float32)
    def create(self,embeddings):
        self.embeddings = embeddings
    
    def df(self,x,wrt):
        M = x.shape[0]
        return ag.grad(x.flatten(), wrt,
                        grad_outputs=torch.ones(M).to(x.dtype).to(wrt.device),create_graph=True)[0]
    def pos_encoder(self,x):  
        encoding = [x]
        for freq in self.frequency_bands:
            encoding.append(torch.sin(x * freq))
            encoding.append(torch.cos(x * freq))
        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            pos_codes = torch.cat(encoding, dim=-1)
            return pos_codes
    def get_basis(self, xt):
        x, y, z = xt[..., 0], xt[..., 1], xt[..., 2]
        zeros = xt[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        a4 = torch.stack([zeros, -y, -z], dim=-1)
        a5 = torch.stack([-x, zeros, -z], dim=-1)
        a6 = torch.stack([-x, -y, zeros], dim=-1)
        return torch.stack([b1, b2, b3, b4, b5, b6], dim=-2), torch.stack([b1, b2, b3, a4, a5, a6], dim=-2)
    def get_vel(self, xt):
        v_basis, _ = self.get_basis(xt)
        weights = self.weight_net(self.pos_encoder(xt))
        v = torch.einsum('...ij,...i->...j', v_basis, weights)
        return v

    def forward(self, t,x):
        norm_t = t -0.5
        # xt
        xt = torch.cat([x,norm_t.unsqueeze(0).repeat(x.shape[0],1)],dim=-1)
        vel = torch.zeros_like(xt[..., :3])
        vel = self.get_vel(xt)[..., :3]

        if self.compute_gradient:
            total_dxyz_dxyz=[]
            for o in range(3):
                derivs = self.df(vel[:,o],xt)
                derivs= torch.cat((derivs[:, 1:], derivs[:, :1]), dim=1)#.clone()
                total_dxyz_dxyz.append(derivs.unsqueeze(1))
            total_dxyz_dxyz = torch.cat(total_dxyz_dxyz,dim=1)
            self.grads.append(total_dxyz_dxyz)
        self.nfe+=1  #To check #ode evaluations
        return vel
class Warper(nn.Module):
    '''
    A single DeformBlock is made up of two NODE Blocks. Refer secion 3 (Overall Architecture)
    '''
    def __init__(self, 
                 methods,
                 time=1.0,
                 tol = 1e-5):
        super(Warper, self).__init__()
        '''
        Initialization.
        time: some number 0-1
        num_hidden: Number of hidden nodes in the MLP of dynamics
        latent_len: Length of shape embeddings
        tol: tolerance of the ODE Solver
        '''
        
        # Several NODE Blocks
        self.methods = methods
        
        if methods == "NVFI":
            self.integral1 = NODEBlock(VelocityAABBSur(), tol = tol)
        elif methods == "OF4D":
            self.integral1 = NODEBlock(OF4D(), tol = tol)
        elif methods == "4DSDF":
            self.integral1 = NODEBlock(VelocityAABBSur(), tol = tol,id_flag=True)

    def equip_embedding(self,embeddings):
        if self.methods == "NVFI":
            pass
        elif self.methods=="OF4D":
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
    def turon_grad(self):
        self.integral1.odefunc.compute_gradient = True
    def turoff_grad(self):
        self.integral1.odefunc.compute_gradient = False
    def forward(self, inputs, src_time, tgt_time):
        if src_time==tgt_time:
            return inputs
        xyz =  self.integral1(inputs,src_time,tgt_time)
        return xyz
  

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
        self.integral1 = NODEBlock(TPSR(hidden_size, latent_size), tol = tol)
        self.integral2 = NODEBlock(TPSR(hidden_size, latent_size), tol = tol)
        
        self.integral3 = NODEBlock(TPSR(hidden_size, latent_size), tol = tol)
        self.integral4 = NODEBlock(TPSR(hidden_size, latent_size), tol = tol)

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
        

class Decoder(nn.Module):
    def __init__(self,methods,latent_size,warper_kargs,decoder_kargs):
        super(Decoder, self).__init__()
        self.methods = methods
        if methods == "TPSR":
            self.warper = Warper_multi(latent_size, **warper_kargs)
        
        elif methods == "NVFI" or methods == "OF4D" or methods=="4DSDF":
            self.warper = Warper(methods)  
        if methods=="4DSDF":
            self.sdf_decoder = SdfDecoder(**decoder_kargs,input_dims=4)
        else:

            self.sdf_decoder = DiGSNetwork(in_dim=3,out_dim=1, 
                                       decoder_hidden_dim=256,nl="sine",
                                       decoder_n_hidden_layers=8, init_type='siren',
                                       sphere_init_params=[1.6,0.1])
            
        self.partseg = PartNet(hidden_features=128,out_features=20,in_features=3,num_hidden_layers=1,initial_first=True)
    def forward(self, inputs,temp_time=0.5,requires_grad=False):
        outputs = {}
        outputs["gt_nms"] = inputs['nms'].cuda()
        outputs['pts'] = inputs["pts"].cuda()
        if requires_grad ==True:
            self.warper.turon_grad()
        elif requires_grad==False:
            self.warper.turoff_grad()
        
        p_final  = self.warper(outputs['pts'], inputs["normed_time"],temp_time)
        if self.methods=="4DSDF":
            outputs['mns_preds']  = self.sdf_decoder.forward_unit(p_final,inputs["normed_time"])
        else:
            outputs['mns_preds']  = self.sdf_decoder.forward_unit(p_final)
        part_prob = torch.softmax(self.partseg(p_final),dim=-1)
        outputs['mns_prob'] = part_prob
        outputs['mns_final'] = p_final

        # near surface points compute
        outputs['near_pts'] = inputs["sald_sps"].cuda()
        
        near_final  = self.warper(outputs['near_pts'], inputs["normed_time"],temp_time)
        if self.methods=="4DSDF":
            outputs['near_preds']  = self.sdf_decoder.forward_unit(near_final,inputs["normed_time"])
        else:
            outputs['near_preds']  = self.sdf_decoder.forward_unit(near_final)
        self.warper.turoff_grad()

        # ELk uniform sampling predict
        outputs['uniform_pts'] = inputs["non_mnpts"].cuda()
        uniform_final  = self.warper(outputs['uniform_pts'], inputs["normed_time"],temp_time)
        if self.methods == "4DSDF":
            outputs['uniform_preds']  = self.sdf_decoder.forward_unit(uniform_final,inputs["normed_time"])
        else:
            outputs['uniform_preds']  = self.sdf_decoder.forward_unit(uniform_final)
        return outputs

if __name__ == "__main__":
    warper = OF4D()
    nums = sum(p.numel() for p in warper.parameters() if p.requires_grad)
    print('nums: ',nums)