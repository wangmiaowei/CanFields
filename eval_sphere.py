import os 
from utils import *
import time
import argparse
import json
import torch.autograd as ag
import point_cloud_utils as pcu
import logging
import math
import os
import sys
from networks.multi_velocity_debug import adjust_net
import random
import time
import torch
import trimesh
import pytorch3d
from pytorch3d.io import IO
from losses.recon import compute_truncated_chamfer_distance
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
        "--experiment",
        default="exp_res/formulation/ode_solve",
        dest="experiment_directory",
        required=False,  
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
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
        "--object_name",
        "-o",
        dest="object_name",
        required=False,  
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--dataset_name",
        "-d",
        dest="dataset_name",
        required=True,  
        help="please choose 'animals','medical' or 'defaust' ",
    )
    arg_parser.add_argument(
        "--epoch",
        "-e",
        dest="epoch",
        required=True,  
        help="save_path epoch number:15000 or 20000",
    )
    arg_parser.add_argument(
        "--step",
        "-s",
        dest="step",
        required=True,  
        help="step_1 canonic shape step:2 move canonic shape",
    )    
    arg_parser.add_argument(
        "--multi_step",
        "-m",
        dest="multi_step",
        required=True,  
        help="step_1 canonic shape step:2 move canonic shape",
    )

    add_common_args(arg_parser)
    args = arg_parser.parse_args()
    if args.methods!= None:
        args.specifications = (args.specifications).replace("specs.json","others_specs.json")
    configure_logging(args)
    return args
def clear_folder(folder_path):
    contents = os.listdir(folder_path)
    
    for item in contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            clear_folder(item_path)
            os.rmdir(item_path)
            print(f"Deleted folder: {item_path}")

def load_model_latents():
    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)
    args = build_args()
    specs = json.load(open(args.specifications))
    sub_obj_name = args.object_name.split("_full_l2")[0]
    obj_name = args.object_name 
    
    if args.methods!= None:
        args.experiment_directory=(args.experiment_directory).replace("ode_solve",args.methods)
        export_path = "exp_res/formulation"+"/"+args.methods+"_manifold/"+ obj_name+"/"
    elif "woAdaptor" in obj_name:
        args.experiment_directory=(args.experiment_directory).replace("ode_solve","woAdaptor")
        export_path = "exp_res/formulation"+"/"+"woAdaptor_manifold/"+ obj_name+"/" 
    else:
        export_path = "results/manifold/"

    if args.dataset_name == "animals":
        
        from datasets import data_wmw
        anima_path ="datasets/Deforming4D/notexture/"
        specs["ShapePath"] = anima_path + sub_obj_name
        if args.methods!= None or "_woAdaptor" in obj_name:
            gap_lenth = int(obj_name.split("_")[-4])
            start_idx = int(obj_name.split("_")[-6])
            stride = int(obj_name.split("_")[-2])
        else:
            gap_lenth = int(obj_name.split("_")[-3])
            start_idx = int(obj_name.split("_")[-5])
            stride = int(obj_name.split("_")[-1])

        sdf_dataset = data_wmw.DeformSamples(specs["ShapePath"],start_idx=start_idx,gap=gap_lenth,stride=stride)

    elif args.dataset_name =="defaust":
        from datasets import data_dfaust
        defaust_path= "datasets/D_Faust/"
        specs["ShapePath"] = defaust_path + args.object_name
        sdf_dataset = data_dfaust.DeformSamples(specs["ShapePath"],train_step = 3)
        obj_name = specs["ShapePath"].split('/')[-1] + "_S3_full_l2_0.25_elas_20.0_prob_0.12" 
    elif args.dataset_name =="medical":
        from datasets import data_medician
        medical_path= "datasets/Medical/"
        specs["ShapePath"] = medical_path + args.object_name
        sdf_dataset = data_medician.DeformSamples(specs["ShapePath"],train_step = 1)
        obj_name = specs["ShapePath"].split('/')[-1] + "_M1_full_l2_0.25_elas_10.0_prob_0.18" 
    start_idx = 0
    end_idx = len(sdf_dataset) - 1
    experiment_directory = os.path.join(args.experiment_directory, obj_name)
    latent_size = specs["CodeLength"]
    pts_name = args.epoch

    if args.methods!= None:
        from networks.other_volocities import Decoder
        decoder = Decoder(args.methods,latent_size,specs["NetworkSpecs"]["warper_kargs"],specs["NetworkSpecs"]["decoder_debug_kargs"])
        decoder = decoder.cuda()
        if args.methods=="OF4D":
            start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.).cuda()
        else:
            start_end_vecs = torch.nn.Embedding(1, 128, max_norm=1.).cuda()
        decoder.warper.equip_embedding(start_end_vecs)
        print('experiment_directory: ',experiment_directory)
        load_model_parameters(experiment_directory, pts_name, decoder)
        load_latent_vectors(experiment_directory, pts_name, start_end_vecs)
        return args,sdf_dataset,decoder,start_end_vecs,start_idx,end_idx,export_path
    elif "woAdaptor" in args.object_name:
        specs["NetworkArch"] = "multi_velocity_debug"
        arch = __import__("networks." + specs["NetworkArch"], fromlist=["Warper"])

        decoder = arch.Decoder(args.multi_step,latent_size, 
                            specs["NetworkSpecs"]["warper_kargs"],specs["NetworkSpecs"]["decoder_debug_kargs"])
        decoder = decoder.cuda()
        start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.).cuda()
        decoder.warper.equip_embedding(start_end_vecs)
        load_model_parameters(experiment_directory, pts_name, decoder)
        load_latent_vectors(experiment_directory, pts_name, start_end_vecs)
        return args,sdf_dataset,decoder,start_end_vecs,start_idx,end_idx,export_path
    else:
        specs["NetworkArch"] = "multi_velocity_debug"
        arch = __import__("networks." + specs["NetworkArch"], fromlist=["Warper"])

        decoder = arch.Decoder(args.multi_step,latent_size, 
                            specs["NetworkSpecs"]["warper_kargs"],specs["NetworkSpecs"]["decoder_debug_kargs"])
        decoder = decoder.cuda()
        # shape_codes inititalize
        ajust =  adjust_net().cuda()
        start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.).cuda()
        decoder.warper.equip_embedding(start_end_vecs)
        adjust_start_end_vecs = torch.nn.Embedding(end_idx-start_idx+1, 128, max_norm=1.).cuda()
        composed_model = torch.nn.ModuleList().cuda()
        composed_model.append(ajust)
        composed_model.append(decoder)
        #print('comnposed_model: ',composed_model)
        print('experiment_directory: ',experiment_directory)
        load_model_parameters(experiment_directory, pts_name, composed_model)
        load_adjust_latent_vectors(experiment_directory, pts_name, adjust_start_end_vecs)
        ajust = composed_model[0].eval()
        decoder = composed_model[1].eval()
        return args,sdf_dataset,decoder,ajust,adjust_start_end_vecs,start_idx,end_idx,export_path
    
def creat_colored_canonic_mesh(decoder,sdf_dataset,start_idx,end_idx):
    sw_pts = sdf_dataset.evaluate_train(int(start_idx+end_idx//2))
    sw_bbox_min = sw_pts.min(0,keepdim=True)[0]
    sw_bbox_max= sw_pts.max(0,keepdim=True)[0]
    # compute the swbbox middle point
    swbbox_middle = 0.5 * (sw_bbox_min + sw_bbox_max)
    # compute the half length of diagonal bbox
    swbbox_len = torch.abs(sw_bbox_max - swbbox_middle)
    min_swbbox = swbbox_middle - 1.7*swbbox_len
    #inter_latent = None
    create_mesh_colored_wmw(decoder,None,"results/"+"normalized_tempt",
                                N=200,sm_offset=min_swbbox[0],
                                sm_len=2*1.7*swbbox_len,
                                lag_mean=None,lag_scale=None)

def find_closest_number(numbers,target):
        closest_number = min(numbers,key=lambda x: abs(x-target))
        index = numbers.index(closest_number)
        return closest_number,index
def move_canonic_shape(methods,sdf_dataset,decoder,start_time,end_time,gt_time,export_path):
    #backward_time = torch.linspace(start_time,end_time,50).cuda()
    backward_time = sdf_dataset.normalized_times

    if methods == "4DSDF":

        os.makedirs(export_path, mode = 0o777, exist_ok = True)
        clear_folder(export_path)
        d_s = sdf_dataset.scale
        d_m = sdf_dataset.mean
        this_idx = 0
        for back_time in backward_time:
            sub_file = sdf_dataset.sub_files[this_idx]
            print('sub_file: ',sub_file)
            _,index = find_closest_number(sdf_dataset.normalized_times,back_time)
            sw_pts = sdf_dataset.evaluate_train(int(index))
            sw_bbox_min = sw_pts.min(0,keepdim=True)[0]
            sw_bbox_max= sw_pts.max(0,keepdim=True)[0]
            # compute the swbbox middle point
            swbbox_middle = 0.5 * (sw_bbox_min + sw_bbox_max)
            # compute the half length of diagonal bbox
            swbbox_len = torch.abs(sw_bbox_max - swbbox_middle)
            min_swbbox = swbbox_middle - 1.7*swbbox_len
            #inter_latent = None
            mesh_path =export_path + str(back_time.cpu().item())
            print('mesh_path: ',mesh_path)
            create_mesh_colored_wmw(decoder,None,mesh_path,
                                        N=128,sm_offset=min_swbbox[0],
                                        sm_len=2*1.7*swbbox_len,
                                        lag_mean=d_m,lag_scale=d_s,this_time =back_time.cuda())
            this_idx = this_idx+1
    else:
        templat_full_path = "results/normalized_tempt.ply"

        temp_mesh = trimesh.load(templat_full_path)
        temp_v = np.asarray(temp_mesh.vertices,dtype=np.float32)
        temp_f = np.asarray(temp_mesh.faces)
        temp_v = torch.tensor(temp_v).cuda()
        num_chunks = 5
        d_s = sdf_dataset.scale
        d_m = sdf_dataset.mean
        os.makedirs(export_path, mode = 0o777, exist_ok = True)
        clear_folder(export_path)

        
        time.sleep(3)
        this_idx  =0 
        for back_time in backward_time:
            sub_file = sdf_dataset.sub_files[this_idx]
            chunks = torch.chunk(temp_v, num_chunks, dim=0)
            v_final = []
            for i, chunk in enumerate(chunks):
                chunk_final = decoder.warper(chunk, gt_time,back_time)
                v_final.append(chunk_final)
            v_final = torch.cat(v_final,dim=0)/d_s.cuda() + d_m.cuda()
            if methods == "4DSDF":
                mesh_path =export_path + str(back_time.cpu().item())
            else:
                mesh_path =export_path + "/"+sub_file

            print('mesh_path: ',mesh_path)
            new_mesh = trimesh.Trimesh(vertices=v_final.detach().clone().cpu().numpy(), faces=temp_f)
            new_mesh.visual.face_colors = temp_mesh.visual.face_colors
            new_mesh.export(mesh_path)
            this_idx = this_idx+1
if __name__ == "__main__":
    # python eval_sphere.py -o grizzRSS_Attack0  -d animals
    setup_seed(3407)
    args = build_args()
    if args.methods!=None or "woAdaptor" in args.object_name:
        args,sdf_dataset, decoder,start_end_vecs,start_idx,end_idx,export_path = load_model_latents()
    else:
        args,sdf_dataset, decoder, ajust, adjust_start_end_vecs,start_idx,end_idx,export_path = load_model_latents()

    if args.step == "1":
        creat_colored_canonic_mesh(decoder,sdf_dataset,start_idx,end_idx)


    start_time = (sdf_dataset[start_idx])["normed_time"].cuda()
    end_time = (sdf_dataset[end_idx])["normed_time"].cuda()

    decoder.warper.split_time(start_time,end_time)
    gt_time = 0.5*(start_time + end_time)
    if args.step == "2":
        move_canonic_shape(args.methods,sdf_dataset,decoder,start_time,end_time,gt_time,export_path)