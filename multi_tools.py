from utils import *
import pynvml
import pytest
import argparse
import torch
import torch.autograd as ag
import gc
import math
from networks.multi_velocity_debug import adjust_net
from pytorch3d.ops.knn import knn_gather, knn_points
from losses.recon import ReconLoss_debug

def build_args():
    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--specifications",
        default="configs/specs.json",
        dest="specifications",
        required=False,  
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--adjust_weights",
        "-a",
        dest="adjust_weights",
        required=True,  
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    
    arg_parser.add_argument(
        "--epoch_velo",
        "-ev",
        dest="epoch_velo",
        required=True,  
        help="The requires number to use epoches if epoch=-1 means do not add velocity constrant",
    )
    arg_parser.add_argument(
        "--adjust_use",
        "-u",
        dest="adjust_use",
        required=True,  
        help="do not use adjust",
    )
    
    arg_parser.add_argument(
        "--methods",
        "-me",
        default= None,
        dest="methods",
        required=False,  
        help="use other methods: 4DDeepSDF,TPSR,NVFI,OF4D",
    )
    arg_parser.add_argument(
        "--multi_ode",
        "-m",
        dest="multi_ode",
        required=True, 
        help="whether to use multi_ode",
    )
    arg_parser.add_argument(
        "--clip",
        "-c",
        dest="clip",
        required=True,  
        help="clip raw dataset with start_idx,gap,stride",
    )
    arg_parser.add_argument(
        "--experiment",
        default="exp_res/formulation/ode_solve",
        dest="experiment_directory",
        required=False,  
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--dataset_name",
        "-d",
        dest="dataset_name",
        required=True,  
        help="please choose 'animals' or 'defaust' ",
    )
    arg_parser.add_argument(
        "--pts_num",
        "-pts",
        dest="pts_num",
        required=False,  
        help="select the number to load",
    )
    arg_parser.add_argument(
        "--object_name",
        "-o",
        dest="object_name",
        required=True,  
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    add_common_args(arg_parser)
    args = arg_parser.parse_args()
    if args.methods!= None:
        args.specifications = (args.specifications).replace("specs.json","others_specs.json")


    configure_logging(args)
    return args
def df(x,wrt):
    M = x.shape[0]
    return ag.grad(x.flatten(), wrt,
                       grad_outputs=torch.ones(M, dtype=torch.float32).to(wrt.device),create_graph=True)[0]
def det3(M):
    det=M[...,0,0]*M[...,1,1]*M[...,2,2]\
        +M[...,0,1]*M[...,1,2]*M[...,2,0]\
        +M[...,0,2]*M[...,1,0]*M[...,2,1]\
        -M[...,0,2]*M[...,1,1]*M[...,2,0]\
        -M[...,0,1]*M[...,1,0]*M[...,2,2]\
        -M[...,0,0]*M[...,1,2]*M[...,2,1]
    return det
def compute_elastic_loss(jacobian=None):
    svals = torch.linalg.svdvals(jacobian)
    differential_det = det3(jacobian)
    f2 = torch.ones_like(svals)
    f2[...,-1] = differential_det
    sq_residual = torch.sum((svals-f2)**2, dim=-1)
    return sq_residual
def gpu_memory_usage(device=0):
    return torch.cuda.memory_allocated(device) / 1024.0**3


def gpu_memory_usage_all(device=0):
    usage = torch.cuda.memory_allocated(device) / 1024.0**3
    reserved = torch.cuda.memory_reserved(device) / 1024.0**3
    smi = gpu_memory_usage_smi(device)
    return usage, reserved - usage, max(0, smi - reserved)


def gpu_memory_usage_smi(device=0):
    if isinstance(device, torch.device):
        device = device.index
    if isinstance(device, str) and device.startswith("cuda:"):
        device = int(device[5:])
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024.0**3


#@pytest.fixture(autouse=True)
def memory_cleanup():

    gc.collect()
    torch.cuda.empty_cache()

    cnt = 0
    for obj in get_tensors():
        obj.detach()
        obj.grad = None
        obj.storage().resize_(0)
        cnt += 1
    gc.collect()
    torch.cuda.empty_cache()


def get_tensors(gpu_only=True):
    for obj in gc.get_objects():
        try:
            if obj.shape != torch.Size([1000,3]) or obj.shape != torch.Size([2000,3]) or obj.shape != torch.Size([10000,3]):
                continue
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue
            
            if tensor.is_cuda or not gpu_only:
                yield tensor
        except Exception:  # nosec B112 pylint: disable=broad-exception-caught
            continue

def load_network_model_others(args,specs,start_idx,end_idx):
    adjust_use = args.adjust_use
    use_multi_ode = args.multi_ode
    code_bound = get_spec_with_default(specs, "CodeBound", None)
    specs["NetworkArch"] = "multi_velocity_debug"

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    from networks.other_volocities import Decoder
    decoder = Decoder(args.methods,latent_size,specs["NetworkSpecs"]["warper_kargs"],specs["NetworkSpecs"]["decoder_debug_kargs"])
    decoder = decoder.cuda()

    # shape_codes inititalize
    if args.methods == "OF4D":
        start_end_vecs = torch.nn.Embedding(2, latent_size, max_norm=code_bound).cuda()
    else:
        start_end_vecs = torch.nn.Embedding(1, latent_size, max_norm=code_bound).cuda()

    torch.nn.init.normal_(
        start_end_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    
    return decoder,start_end_vecs
    

def load_network_model(args,specs,start_idx,end_idx):
    adjust_use = args.adjust_use
    use_multi_ode = args.multi_ode
    code_bound = get_spec_with_default(specs, "CodeBound", None)
    specs["NetworkArch"] = "multi_velocity_debug"
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(use_multi_ode,latent_size, 
                           specs["NetworkSpecs"]["warper_kargs"],specs["NetworkSpecs"]["decoder_debug_kargs"])
    decoder = decoder.cuda()

    # shape_codes inititalize
    start_end_vecs = torch.nn.Embedding(2, latent_size, max_norm=code_bound).cuda()
    torch.nn.init.normal_(
        start_end_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    if adjust_use == "True":
        adjust_start_end_vecs = torch.nn.Embedding(end_idx-start_idx+1, 128, max_norm=code_bound).cuda()
        torch.nn.init.normal_(
                adjust_start_end_vecs.weight.data,
                0.0,
                get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(128),)
        ########Adjust Net###########
        ajust =  adjust_net().cuda()
        return decoder, ajust,(start_end_vecs,adjust_start_end_vecs)
    else:
        return decoder,start_end_vecs
def specs_weights(specs):
    latent_reg_weight = specs["loss"][0]['properties']['latent_reg_weight']
    grad_loss_weight = specs["loss"][0]['properties']['grad_loss_weight']
    grad_on_surface_weight = specs["loss"][0]['properties']['grad_on_surface_weight']
    return (latent_reg_weight,grad_loss_weight,grad_on_surface_weight)

def load_dataset(args,specs,ajust_coes=None):
    # load dataset

    clip_weights = args.clip.split(',')
    clip_coes = [int(clip_weights[0]),int(clip_weights[1]),int(clip_weights[2])]

    shapepath = specs["ShapePath"]
    if ajust_coes!=None:
        coe_l2 = ajust_coes[0]
        coe_elas = ajust_coes[1]
        coe_prob = ajust_coes[2]
        coe_path = str(coe_l2)+"_elas_"+str(coe_elas)+"_prob_"+str(coe_prob)
    else:
        coe_path ="woAdjust"

    if args.dataset_name == "animals":
        from datasets import data_wmw
        start_idx = clip_coes[0] #3
        gap = clip_coes[1]
        stride =clip_coes[2] #2
        sdf_dataset = data_wmw.DeformSamples(shapepath,start_idx = start_idx,gap = gap,stride=stride)
        #sdf_dataset = data.DeformSamples(shapepath,train_rate = 0.5)
        obj_name = shapepath.split('/')[-1]+"_full_l2_"+coe_path+"_start_id_"+str(start_idx)+"_gap_"+str(gap)+"_stride_"+str(stride)
        
    elif args.dataset_name == "defaust":
        from datasets import data_dfaust
        sdf_dataset = data_dfaust.DeformSamples(shapepath,train_step = 2) #train_rate 0.5
        obj_name = shapepath.split('/')[-1]+"_S3_full_l2_"+coe_path
    elif args.dataset_name == "medical":
        from datasets import data_medician
        sdf_dataset = data_medician.DeformSamples(shapepath,train_step = 1) #train_rate 0.5
        obj_name = shapepath.split('/')[-1]+"_M1_full_l2_"+coe_path
    
    # modify experiment_directory from args to others
    if args.methods!= None:
        obj_name = obj_name+"_"+args.methods
        args.experiment_directory=(args.experiment_directory).replace("ode_solve",args.methods)
    if args.adjust_use == "False" and args.methods==None:
        obj_name = obj_name+"_"+"woAdaptor"
        args.experiment_directory=(args.experiment_directory).replace("ode_solve","woAdaptor")
        print('args.experiment_directory: ', args.experiment_directory)
    experiment_directory = os.path.join(args.experiment_directory, 
                                             obj_name)
    return sdf_dataset,experiment_directory,obj_name


def get_lr(adjust_use,specs,decoders,three_latents):

    shape_start_end_vecs = three_latents[0]
    if adjust_use=="True":
        adjust_start_end_vecs = three_latents[1]

    lr_schedules = get_learning_rate_schedules(specs)
    if adjust_use=="True":
        optimizer_all = torch.optim.Adam(
            [   {
                    "params": decoders[0].partseg.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": decoders[0].warper.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
                {
                    "params": decoders[0].sdf_decoder.parameters(),
                    "lr": lr_schedules[2].get_learning_rate(0),

                },
                {
                    "params":shape_start_end_vecs.parameters(),
                    "lr":lr_schedules[4].get_learning_rate(0),
                },
                {
                    "params": decoders[1].parameters(),
                    "lr": lr_schedules[5].get_learning_rate(0),
                },
                {
                    "params": adjust_start_end_vecs.parameters(),
                    "lr": lr_schedules[6].get_learning_rate(0),
                }
            ]
        )
    else:
        optimizer_all = torch.optim.Adam(
            [   {
                    "params": decoders[0].partseg.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": decoders[0].warper.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
                {
                    "params": decoders[0].sdf_decoder.parameters(),
                    "lr": lr_schedules[2].get_learning_rate(0),

                },
                {
                    "params":shape_start_end_vecs.parameters(),
                    "lr":lr_schedules[4].get_learning_rate(0),
                }
            ]
        )


    return optimizer_all,lr_schedules

def load_model_latents_others(decoder,start_end_vecs,experiment_directory,pts_name):
    if pts_name==None or int(pts_name)<1000:
        return decoder,start_end_vecs
    
    decoder.warper.equip_embedding(start_end_vecs)
    load_model_parameters(experiment_directory, pts_name, decoder)

    load_latent_vectors(experiment_directory, pts_name, start_end_vecs)
    decoder.warper.unload_embedding()
    return decoder,start_end_vecs

def load_model_latents(ajust,decoder,two_codes,experiment_directory,pts_name):
    if pts_name==None or int(pts_name)<1000:
        return ajust,decoder,two_codes
    decoder.warper.equip_embedding(two_codes[0])
    composed_model = torch.nn.ModuleList().cuda()
    composed_model.append(ajust)
    composed_model.append(decoder)
    load_model_parameters(experiment_directory, pts_name, composed_model)
    ajust =  composed_model[0]
    decoder = composed_model[1]
    # if two_codes[1] ==None:
    #     return ajust,decoder

    load_latent_vectors(experiment_directory, pts_name, two_codes[0])
    load_adjust_latent_vectors(experiment_directory, pts_name, two_codes[1])
    decoder.warper.unload_embedding()
    return ajust,decoder,two_codes

class Trainer:
    def __init__(self,args,
                 specs,ajust,
                 decoder,
                 sdf_dataset,
                 two_codes,
                 start_idx,end_idx,
                 ajust_coes,obj_name,
                 experiment_directory):
        
        self.l2_loss = torch.nn.MSELoss()
        self.sig_loss = torch.nn.BCELoss()
        self.identity = torch.eye(3).reshape((1, 3, 3)).cuda()
        self.ajust = ajust
        self.decoder = decoder
        self.experiment_directory = experiment_directory

        self.adjust_start_end_vecs = two_codes[1]
        self.start_end_vecs = two_codes[0]

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.sdf_dataset =sdf_dataset

        start_time = (sdf_dataset[start_idx])['normed_time'].cuda()
        end_time = (sdf_dataset[end_idx])['normed_time'].cuda()
        
        self.template_time = 0.5*(start_time + end_time)
        self.template_id = int(self.start_idx+self.end_idx//2)
        self.recon_loss = ReconLoss_debug()
        self.fit_weights = specs_weights(specs)
        self.scheduler = get_learning_rate_schedules(specs)
        self.max_fit_loss = 0
        self.min_max_fit_loss =10
        self.min_fit_loss = 1000
        self.optimizer_all = self.load_optimizer()
        self.ajust_coes = ajust_coes
        self.num_epochs = specs["NumEpochs"]
        self.obj_name = obj_name
        self.decoder.warper.split_time(start_time,end_time)
        self.epoch_velo = int(args.epoch_velo)
        self.args = args
    def load_optimizer(self):
        optimizer_all = torch.optim.Adam(
            [   {
                    "params": self.decoder.partseg.parameters(),
                    "lr": self.scheduler[0].get_learning_rate(0),
                },
                { 
                    "params": self.decoder.warper.parameters(),
                    "lr": self.scheduler[1].get_learning_rate(0),
                },
                {
                    "params": self.decoder.sdf_decoder.parameters(),
                    "lr": self.scheduler[2].get_learning_rate(0),
                },
                {
                    "params":self.start_end_vecs.parameters(),
                    "lr":self.scheduler[3].get_learning_rate(0),
                },
                {
                    "params": self.ajust.parameters(),
                    "lr": self.scheduler[4].get_learning_rate(0),
                },
                {
                    "params": self.adjust_start_end_vecs.parameters(),
                    "lr": self.scheduler[5].get_learning_rate(0),
                }
            ]
        )
        self.decoder.warper.equip_embedding(self.start_end_vecs)
        return optimizer_all
    def comput_ajust_net(self,src_pts,src_nms,rand_scene_id,src_normed_time):
        src_pts_ajust,src_prob = self.ajust(src_normed_time.to(torch.float32),
                                            src_pts,
                                            self.adjust_start_end_vecs.weight[rand_scene_id-self.start_idx]) #modify the loss
        dxyz_dxyz=[]
        

        iden_matrix = self.identity.repeat(len(src_pts),1,1)
        for o in range(3):
            derivs = df(src_pts_ajust[:,o],src_pts)
            dxyz_dxyz.append(derivs.unsqueeze(1))
        new_der = torch.cat(dxyz_dxyz,dim=1)
        src_elastic_loss= (1/len(src_pts)) * 1/3 *((torch.norm(new_der - iden_matrix,p='fro'))**2)
        src_l2_loss = self.l2_loss(src_pts_ajust,src_pts) #l2_loss
        src_prob_gts = torch.ones_like(src_prob)
        src_prob_loss = self.sig_loss(src_prob.T,src_prob_gts.T)
        src_nms_adjust = ((src_nms.unsqueeze(1)@ torch.inverse(new_der)).squeeze(1))
        return (src_elastic_loss,src_l2_loss,src_prob_loss),(src_pts_ajust,src_nms_adjust),src_prob

    def ajust_new_dataset(self,src_ajust,src_normed_time):
        src_pts_ajust = src_ajust[0]
        src_nms_adjust = src_ajust[1]

        x_nn = knn_points(src_pts_ajust.unsqueeze(0),src_pts_ajust.unsqueeze(0),K=11)
        sm_sigma = ((x_nn.dists[0][:,-1])**0.5).unsqueeze(1).repeat(1,3)
        big_sigma = 0.3 * torch.ones_like(src_pts_ajust)
        sald_sps = torch.cat([src_pts_ajust + torch.normal(mean=0.0,std = sm_sigma),
                src_pts_ajust + torch.normal(mean=0.0,std = big_sigma)],dim=0)
        bb_min = src_pts_ajust.min(0,keepdim=True)[0]
        bb_max = src_pts_ajust.max(0,keepdim=True)[0]
        bb_mean = (bb_max+bb_min)/2

        bb_max = (bb_mean+1.6*(bb_max-bb_mean))#.detach()
        bb_min = (bb_mean-1.6*(bb_mean-bb_min))#.detach()
        non_mnpts = torch.rand(10000,3).cuda() * (bb_max - bb_min) + bb_min
            
        adjust_input_dps = {"pts": src_pts_ajust, "nms":src_nms_adjust,
                             "non_mnpts":non_mnpts,"sald_sps":sald_sps,"normed_time":src_normed_time}
        return adjust_input_dps
    def comput_kL_L2(self):
        total_grads = self.decoder.warper.return_grads()
        accs = 0
        ks_losses = 0
        avg_energs = 0
        len_idx = 0
        for total_grad in total_grads:
            for this_grad in total_grad:
                acc = (this_grad[:,:,0].norm(dim=1)).mean()
                kl_loss = torch.norm(this_grad[:,:,1:] + this_grad[:,:,1:].transpose(1,2),
                                        dim=[1,2]).mean(dim=0)
                energy = torch.sum(this_grad[:,:,1]**2 + this_grad[:,:,2]**2 + this_grad[:,:, 3]**2)
                average_energe = 0.5 * (energy/ this_grad.size(0))
                accs = accs+acc
                ks_losses= ks_losses + kl_loss
                avg_energs = avg_energs + average_energe
                len_idx = len_idx+1
        if len_idx ==0:
            acce_mean=0
            ks_loss_mean =0
            energy_mean = 0
        else:
            acce_mean = accs/len_idx
            ks_loss_mean = ks_losses/len_idx
            energy_mean = avg_energs/len_idx
        return acce_mean,ks_loss_mean,energy_mean
    def compute_latents(self):
        temp_code_loss = ((self.start_end_vecs.weight)**2).mean()
        adjust_code_loss = ((self.adjust_start_end_vecs.weight)**2).mean()
        #return adjust_code_loss
        code_loss = temp_code_loss+adjust_code_loss
        return code_loss
    def get_pts(self,epoch):
        rand_scene_id = random.sample(set(range(self.start_idx,self.end_idx+1)), 1)[0]
        src_dps = self.sdf_dataset[rand_scene_id]
        src_pts = src_dps["pts"].cuda()
        src_nms = src_dps["nms"].cuda()
        src_normed_time = src_dps["normed_time"].cuda()
        src_pts.requires_grad = True
        ajust_losses,src_ajust,src_prob = self.comput_ajust_net(src_pts,src_nms,rand_scene_id,src_normed_time)
        adjust_input_dps = self.ajust_new_dataset(src_ajust,src_normed_time)
        return adjust_input_dps,ajust_losses,src_prob
    def get_template_pts(self,epoch):
        rand_scene_id = self.start_idx
        src_dps = self.sdf_dataset[rand_scene_id]
        src_pts = src_dps["pts"].cuda()
        src_nms = src_dps["nms"].cuda()
        src_normed_time = src_dps["normed_time"].cuda()
        src_pts.requires_grad = True
        ajust_losses,src_ajust,src_prob = self.comput_ajust_net(src_pts,src_nms,rand_scene_id,src_normed_time)
        adjust_input_dps = self.ajust_new_dataset(src_ajust,src_normed_time)
        return adjust_input_dps,ajust_losses,src_prob
    def log_losses(self,epoch,recon_losses,
                   code_loss,fit_loss,align_loss,
                   ajust_losses,total_loss,
                   acce_mean=None,ks_loss_mean=None,energy_mean=None):
        src_elastic_loss,src_l2_loss,src_prob_loss = ajust_losses
        log_dict =  {'epoch: ': epoch}
        log_dict.update(recon_losses)
        log_dict.update({"total loss":total_loss})
        log_dict.update({"code loss":code_loss})
        log_dict.update({"fit loss":fit_loss})
        log_dict.update({"align loss":align_loss})
        log_dict.update({"ajust elastic_loss":src_elastic_loss})
        log_dict.update({"ajust L2_loss":src_l2_loss})
        log_dict.update({"ajust Prob_loss":src_prob_loss})
        if (self.epoch_velo != -1) and (epoch>self.epoch_velo):
            log_dict.update({"accelator mean":acce_mean})
            log_dict.update({"killing vector":ks_loss_mean})
            log_dict.update({"dirchlet energy: ":energy_mean})

        if epoch%100 ==0:
            logging.info("epoch {}...".format(epoch))
            logging.info("Max_GRADParameters {}...".format(self.ajust.trn_branch.weight.max()))
            logging.info("memory_allcoated(MB) {}".format(torch.cuda.memory_allocated()/1048576))
            logging.info("memory_reserved(MB): {}".format(torch.cuda.memory_reserved()/1048576))
            logging.info("Max_allocated_GPU {}".format(torch.cuda.max_memory_allocated()/1024**3))
            logging.info("max_fit_loss = {}".format(self.max_fit_loss))
            logging.info("min_max_fit_loss = {}".format(self.min_max_fit_loss))
            logging.info("min_fit_loss = {}".format(self.min_fit_loss))
            if epoch>12005 and self.min_max_fit_loss>(self.max_fit_loss+0.0015):
                self.min_max_fit_loss =min(self.min_max_fit_loss,self.max_fit_loss)
            if epoch>12020 and (self.min_max_fit_loss == self.max_fit_loss):
                self.save_model(epoch,self.experiment_directory)
                self.save_mesh(epoch)
            self.max_fit_loss = 0
            logging.info("ajust_elastic_loss = {}".format(src_elastic_loss))
            logging.info("part_align_loss = {}".format(align_loss))
            logging.info("adjust_l2_loss = {}".format(src_l2_loss))
            logging.info("total_loss = {}".format(total_loss))
            logging.info("ajust_prob_loss {}...".format(src_prob_loss))
            logging.info("temp_code_loss = {}".format(code_loss))
            logging.info("temporal_code_weights = {}".format(self.start_end_vecs.weight.mean()))
            #logging.info("Currentmaximum_velo_weights = {}".format(self.decoder.warper.integral1.odefunc.l4.weight.mean()))
            
            if (self.epoch_velo != -1) and (epoch>self.epoch_velo):
                logging.info("accelator mean = {}".format(acce_mean))
                logging.info("killing vector = {}".format(ks_loss_mean))
                logging.info("dirchletenergy = {}".format(energy_mean))
        return log_dict

    def forward_epoch(self,epoch):
        self.decoder.warper.clean_grads()
        adjust_input_dps,ajust_losses, src_prob= self.get_pts(epoch)
        #adjust_template_input_dps,ajust_template_losses, src_temp_prob = self.get_template_pts(epoch)
        if (self.epoch_velo != -1) and (epoch>self.epoch_velo):
            pred_pts = self.decoder(adjust_input_dps,self.template_time,requires_grad=True)
            acce_mean,ks_loss_mean,energy_mean = self.comput_kL_L2()
        else:
            pred_pts = self.decoder(adjust_input_dps,self.template_time,requires_grad=False)

        recon_losses = self.recon_loss(pred_pts,src_prob)

        fit_loss,align_loss = self.compute_fit_loss(recon_losses,self.fit_weights)

        if epoch>4500 and fit_loss <(self.min_fit_loss-0.003):
            self.min_fit_loss = fit_loss

        self.max_fit_loss = max(self.max_fit_loss,fit_loss)
        code_loss = self.compute_latents()
        
        total_loss = self.compute_total_loss(epoch,fit_loss,
                                            code_loss,align_loss,ajust_losses)
        if (self.epoch_velo != -1) and (epoch>self.epoch_velo):
            #6e-4 1e-2
            if epoch==(self.epoch_velo+1):
                print('the first acce_mean: ',acce_mean)
                print('the first ks_loss_mean: ',ks_loss_mean)
                print('the first energy_mean: ',energy_mean)
            #if epoch<15000:

            acce_coe = 1e-3 +(1e-2-1e-3)*(epoch-self.epoch_velo)/(20000-self.epoch_velo)
            ks_coe = 5e-5 +(1.5e-3-5e-5)*(epoch-self.epoch_velo)/(20000-self.epoch_velo)
            en_coe = 8e-5 +(1.9e-3-8e-5)*(epoch-self.epoch_velo)/(20000-self.epoch_velo)

            total_loss = total_loss + 0.2*(acce_coe*acce_mean+ks_coe*ks_loss_mean + en_coe*energy_mean)
            log_dicts = self.log_losses(epoch,recon_losses,
                                    code_loss,fit_loss,align_loss,
                                    ajust_losses,total_loss,
                                    acce_mean,ks_loss_mean,energy_mean)
        else:
            log_dicts = self.log_losses(epoch,recon_losses,
                        code_loss,fit_loss,align_loss,
                        ajust_losses,total_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.warper.parameters(), 0.8)
        self.optimizer_all.step()
        self.optimizer_all.zero_grad()

        if epoch%25 ==0:
            adjust_learning_rate(self.scheduler, 
                                 self.optimizer_all, 
                                 epoch//100)
            gc.collect()
            torch.cuda.empty_cache()
        
        del pred_pts["uniform_pts"].grad,pred_pts["near_pts"].grad,pred_pts["pts"].grad
        del adjust_input_dps["pts"].grad,adjust_input_dps["non_mnpts"].grad,adjust_input_dps["sald_sps"].grad
        if  epoch%3000==0:
            try:
                self.save_mesh(epoch)
            except Exception as e:
                logging.warning("SurfaceLevelReconstructionError:   {}".format(e))
        return log_dicts
    def save_model(self,epoch,experiment_directory):
        save_adjust_checkpoints(epoch,experiment_directory,self.ajust,
                        self.decoder,
                        self.start_end_vecs,
                        self.adjust_start_end_vecs)
        
    def save_mesh(self,epoch):
        self.decoder.eval()
        #l2_c,elas_c,prob_c = self.ajust_coes
        sw_pts = self.sdf_dataset.evaluate_train(self.template_id)
        sw_bbox_min = sw_pts.min(0,keepdim=True)[0]
        sw_bbox_max= sw_pts.max(0,keepdim=True)[0]
        # compute the swbbox middle point
        swbbox_middle = 0.5 * (sw_bbox_min + sw_bbox_max)
        # compute the half length of diagonal bbox
        swbbox_len = torch.abs(sw_bbox_max - swbbox_middle)
        min_swbbox = swbbox_middle - 1.7*swbbox_len
        save_mesh_path = "results/"+self.obj_name+str(epoch)+"BestMiddle"+"use_adaptor_Finally_start_OLdODE"
        create_mesh_wmw(self.decoder.sdf_decoder,None,save_mesh_path,
                            N=256,sm_offset=min_swbbox[0],
                            sm_len=2*1.7*swbbox_len,
                            lag_mean=self.sdf_dataset.mean,lag_scale=self.sdf_dataset.scale)
        self.decoder.train()
    def compute_total_loss(self,epoch,fit_loss,code_loss,
                           align_loss,ajust_losses):
        src_elastic_loss,src_l2_loss,src_prob_loss = ajust_losses


        if epoch<4000:
            prob_coes = self.ajust_coes[2]
        else:
            prob_coes = self.ajust_coes[2]+0.3*(epoch-4000)/45000
        prob_coes = min(prob_coes,0.30)

        if epoch<3000:
            l2_coes = self.ajust_coes[0]
            elas_coes = self.ajust_coes[1]
        else:
            l2_coes = self.ajust_coes[0]+(epoch-3000)*2/15000
            elas_coes = self.ajust_coes[1]+(epoch-3000)*1.5/2500
        l2_coes = min(l2_coes,2)#2
        elas_coes = min(elas_coes,30)

        total_loss = fit_loss+1e-4*code_loss+l2_coes*src_l2_loss+ elas_coes*src_elastic_loss+prob_coes*src_prob_loss + align_loss*1e3
        return total_loss
    def compute_fit_loss(self,recon_losses,weights):
        latent_reg_weight = weights[0]
        grad_loss_weight = weights[1]
        grad_on_surface_weight = weights[2]

        eikonal_uniform_loss  = recon_losses["eikonal_uniform"]
        eikonal_near_loss = recon_losses["eikonal_near"]
        surface_norm_loss = recon_losses["surface_norm"]
        surface_pts_loss = recon_losses["surface_pts"]
        near_pts_loss =  recon_losses["near_pts"]
        near_norm_loss = recon_losses["near_norm"]
        #latent_reg_loss = recon_losses["latent_loss"]
        align_loss = recon_losses["align_loss"]

        fit_loss = 0.5*grad_loss_weight * (eikonal_uniform_loss+eikonal_near_loss) +\
        0.5*(surface_pts_loss+near_pts_loss) +\
              0.5*grad_on_surface_weight*(surface_norm_loss+near_norm_loss)
        #+latent_reg_weight*latent_reg_loss
        return fit_loss,align_loss


class Trainer_woadjust:
    def __init__(self,args,
                 specs,decoder,
                 sdf_dataset,
                 start_end_vecs,
                 start_idx,end_idx,
                 obj_name,experiment_directory):
        self.decoder = decoder
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.sdf_dataset =sdf_dataset
        self.start_end_vecs = start_end_vecs
        start_time = (sdf_dataset[start_idx])['normed_time'].cuda()
        end_time = (sdf_dataset[end_idx])['normed_time'].cuda()
        self.decoder.warper.split_time(start_time,end_time)

        self.template_time = 0.5*(start_time + end_time)
        self.recon_loss = ReconLoss_debug()
        self.fit_weights = specs_weights(specs)

        self.scheduler = get_learning_rate_schedules(specs)
        self.max_fit_loss = 0
        self.optimizer_all = self.load_optimizer()
        self.num_epochs = specs["NumEpochs"]
        self.obj_name = obj_name
        self.args = args
        self.epoch_velo = int(args.epoch_velo)
    def load_optimizer(self):
        optimizer_all = torch.optim.Adam(
            [
                {
                    "params": self.decoder.partseg.parameters(),
                    "lr": self.scheduler[0].get_learning_rate(0),
                },
                { 
                    "params": self.decoder.warper.parameters(),
                    "lr": self.scheduler[1].get_learning_rate(0),
                },
                {
                    "params": self.decoder.sdf_decoder.parameters(),
                    "lr": self.scheduler[2].get_learning_rate(0),
                },
                {
                    "params":self.start_end_vecs.parameters(),
                    "lr":self.scheduler[3].get_learning_rate(0),
                }
            ]
        )
        self.decoder.warper.equip_embedding(self.start_end_vecs)
        return optimizer_all

    def comput_kL_L2(self):
        total_grads = self.decoder.warper.return_grads()
        accs = 0
        ks_losses = 0
        avg_energs = 0
        len_idx = 0
        for total_grad in total_grads:
            for this_grad in total_grad:
                acc = (this_grad[:,:,0].norm(dim=1)).mean()
                kl_loss = torch.norm(this_grad[:,:,1:] + this_grad[:,:,1:].transpose(1,2),
                                        dim=[1,2]).mean(dim=0)
                energy = torch.sum(this_grad[:,:,1]**2 + this_grad[:,:,2]**2 + this_grad[:,:, 3]**2)
                average_energe = 0.5 * (energy/ this_grad.size(0))
                accs = accs+acc
                ks_losses= ks_losses + kl_loss
                avg_energs = avg_energs + average_energe
                len_idx = len_idx+1
        if len_idx ==0:
            acce_mean=0
            ks_loss_mean =0
            energy_mean = 0
        else:
            acce_mean = accs/len_idx
            ks_loss_mean = ks_losses/len_idx
            energy_mean = avg_energs/len_idx
        return acce_mean,ks_loss_mean,energy_mean
    def compute_latents(self):
        temp_code_loss = ((self.start_end_vecs.weight)**2).mean()
        return temp_code_loss

    def get_pts(self):
        rand_scene_id = random.sample(set(range(self.start_idx,self.end_idx+1)), 1)[0]
        src_dps = self.sdf_dataset[rand_scene_id]
        src_pts = src_dps["pts"].cuda()
        src_nms = src_dps["nms"].cuda()
        src_normed_time = src_dps["normed_time"].cuda()
        src_pts.requires_grad = True
        
        x_nn = knn_points(src_pts.unsqueeze(0),src_pts.unsqueeze(0),K=11)
        sm_sigma = ((x_nn.dists[0][:,-1])**0.5).unsqueeze(1).repeat(1,3)
        big_sigma = 0.3 * torch.ones_like(src_pts)
        sald_sps = torch.cat([src_pts + torch.normal(mean=0.0,std = sm_sigma),
                src_pts + torch.normal(mean=0.0,std = big_sigma)],dim=0)
        bb_min = src_pts.min(0,keepdim=True)[0]
        bb_max = src_pts.max(0,keepdim=True)[0]
        bb_mean = (bb_max+bb_min)/2

        bb_max = (bb_mean+1.6*(bb_max-bb_mean))#.detach()
        bb_min = (bb_mean-1.6*(bb_mean-bb_min))#.detach()
        non_mnpts = torch.rand(10000,3).cuda() * (bb_max - bb_min) + bb_min
            
        input_dps = {"pts": src_pts, "nms":src_nms,
                             "non_mnpts":non_mnpts,"sald_sps":sald_sps,"normed_time":src_normed_time}
        return input_dps
    def log_losses(self,epoch,recon_losses,
                   code_loss,fit_loss,align_loss,total_loss,
                   acce_mean=None,ks_loss_mean=None,energy_mean=None):
        log_dict =  {'epoch: ': epoch}
        log_dict.update(recon_losses)
        log_dict.update({"total loss":total_loss})
        log_dict.update({"code loss":code_loss})
        log_dict.update({"fit loss":fit_loss})
        log_dict.update({"align loss":align_loss})
        if (self.epoch_velo != -1) and (epoch>self.epoch_velo):
            log_dict.update({"accelator mean":acce_mean})
            log_dict.update({"killing vector":ks_loss_mean})
            log_dict.update({"dirchlet energy: ":energy_mean})

        if epoch%100 ==0:
            logging.info("epoch {}...".format(epoch))
            logging.info("memory_allcoated(MB) {}".format(torch.cuda.memory_allocated()/1048576))
            logging.info("memory_reserved(MB): {}".format(torch.cuda.memory_reserved()/1048576))
            logging.info("Max_allocated_GPU {}".format(torch.cuda.max_memory_allocated()/1024**3))
            logging.info("max_fit_loss = {}".format(self.max_fit_loss))
            self.max_fit_loss = 0
            logging.info("part_align_loss = {}".format(align_loss))
            logging.info("total_loss = {}".format(total_loss))
            logging.info("temp_code_loss = {}".format(code_loss))
            
            if (self.epoch_velo != -1) and (epoch>self.epoch_velo):
                logging.info("accelator mean = {}".format(acce_mean))
                logging.info("killing vector = {}".format(ks_loss_mean))
                logging.info("dirchletenergy = {}".format(energy_mean))
        return log_dict

    def forward_epoch(self,epoch):
        self.decoder.warper.clean_grads()
        input_dps = self.get_pts()

        if (self.epoch_velo != -1) and (epoch>self.epoch_velo):
            pred_pts = self.decoder(input_dps,self.template_time,requires_grad=True)
            acce_mean,ks_loss_mean,energy_mean = self.comput_kL_L2()
        else:
            pred_pts = self.decoder(input_dps,self.template_time,requires_grad=False)

        recon_losses = self.recon_loss(pred_pts)

        fit_loss,align_loss = self.compute_fit_loss(recon_losses,self.fit_weights)
        
        self.max_fit_loss = max(fit_loss,self.max_fit_loss)
        
        code_loss = self.compute_latents()
        
        total_loss = self.compute_total_loss(fit_loss,
                                            code_loss,align_loss)
        if (self.epoch_velo != -1) and (epoch>self.epoch_velo):
            #1e-3~8e-3 for acce_mean
            #5e-5 5e-4
            #8e-5 8e-4
            acce_coe = 1e-3 +(5e-3-1e-3)*(epoch-self.epoch_velo)/(self.num_epochs-self.epoch_velo)
            ks_coe = 5e-5 +(3e-4-5e-5)*(epoch-self.epoch_velo)/(self.num_epochs-self.epoch_velo)
            en_coe = 8e-5 +(4e-4-5e-5)*(epoch-self.epoch_velo)/(self.num_epochs-self.epoch_velo)

            total_loss = total_loss + acce_coe*acce_mean+ks_coe*ks_loss_mean + en_coe*energy_mean
            log_dicts = self.log_losses(epoch,recon_losses,
                                    code_loss,fit_loss,align_loss,total_loss,
                                    acce_mean,ks_loss_mean,energy_mean)
        else:
            log_dicts = self.log_losses(epoch,recon_losses,
                        code_loss,fit_loss,align_loss,total_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.warper.parameters(), 0.5)
        self.optimizer_all.step()
        self.optimizer_all.zero_grad()

        if epoch%25 ==0:
            adjust_learning_rate(self.scheduler, 
                        self.optimizer_all, 
                        epoch//100)
            gc.collect()
            torch.cuda.empty_cache()
        del pred_pts["uniform_pts"].grad,pred_pts["near_pts"].grad,pred_pts["pts"].grad
        if epoch % 5000==0:
            try:
                self.save_mesh(epoch)
            except Exception as e:
                logging.warning("SurfaceLevelReconstructionError:   {}".format(e))
        return log_dicts

    def save_model(self,epoch,experiment_directory):
        save_checkpoints(epoch,experiment_directory,
                         self.decoder,self.optimizer_all,
                         self.start_end_vecs)
        
    def save_mesh(self,epoch):
        self.decoder.eval()
        #l2_c,elas_c,prob_c = self.ajust_coes
        sw_pts = self.sdf_dataset.evaluate_train(int(self.start_idx+self.end_idx//2))
        sw_bbox_min = sw_pts.min(0,keepdim=True)[0]
        sw_bbox_max= sw_pts.max(0,keepdim=True)[0]
        # compute the swbbox middle point
        swbbox_middle = 0.5 * (sw_bbox_min + sw_bbox_max)
        # compute the half length of diagonal bbox
        swbbox_len = torch.abs(sw_bbox_max - swbbox_middle)
        min_swbbox = swbbox_middle - 1.7*swbbox_len
        #sub_path= str(l2_c)+"_"+str(elas_c)+"_"+str(prob_c)+"_"
        save_mesh_path = "results/"+self.obj_name+str(epoch)
        if self.args.methods=="4DSDF":
            create_mesh_wmw(self.decoder.sdf_decoder,None,save_mesh_path,
                            N=256,sm_offset=min_swbbox[0],
                            sm_len=2*1.7*swbbox_len,
                            lag_mean=self.sdf_dataset.mean,lag_scale=self.sdf_dataset.scale,this_time =self.template_time)
        else:
            create_mesh_wmw(self.decoder.sdf_decoder,None,save_mesh_path,
                            N=256,sm_offset=min_swbbox[0],
                            sm_len=2*1.7*swbbox_len,
                            lag_mean=self.sdf_dataset.mean,lag_scale=self.sdf_dataset.scale)
        self.decoder.train()

    def compute_total_loss(self,fit_loss,code_loss,
                           align_loss):
        if self.args.methods!=None:
            total_loss = fit_loss+1e-4*code_loss
        else:
            total_loss = fit_loss+1e-4*code_loss+align_loss*1e3
        return total_loss

    def compute_fit_loss(self,recon_losses,weights):
        latent_reg_weight = weights[0]
        grad_loss_weight = weights[1]
        grad_on_surface_weight = weights[2]

        eikonal_uniform_loss  = recon_losses["eikonal_uniform"]
        eikonal_near_loss = recon_losses["eikonal_near"]
        surface_norm_loss = recon_losses["surface_norm"]
        surface_pts_loss = recon_losses["surface_pts"]
        near_pts_loss =  recon_losses["near_pts"]
        near_norm_loss = recon_losses["near_norm"]
        #latent_reg_loss = recon_losses["latent_loss"]
        align_loss = recon_losses["align_loss"]

        fit_loss = 0.5*grad_loss_weight * (eikonal_uniform_loss+eikonal_near_loss) +\
        0.5*(surface_pts_loss+near_pts_loss) +\
              0.5*grad_on_surface_weight*(surface_norm_loss+near_norm_loss)
        return fit_loss,align_loss
