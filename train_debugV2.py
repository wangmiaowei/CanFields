#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import GPUtil
import os
def set_most_idle_gpu_as_visible():
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPUs found!")
        return
    most_idle_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(most_idle_gpu.id)

set_most_idle_gpu_as_visible()
from utils import wandb_name_init
from multi_tools import *
import json
import logging
import os

import sys
import signal
import random
import time


def velocity_test():
    args = build_args()
    wandb = wandb_name_init(debug=True,
                            obj_name = args.object_name,
                            dataset_name = args.dataset_name)
    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)
    logging.debug("running " + args.experiment_directory)
    signal.signal(signal.SIGINT, signal_handler)
    if not os.path.isfile(args.specifications):
        raise Exception(
            'The experiment directory does not include such files'
        )
    
    #######################################################
    specs = json.load(open(args.specifications))
    if args.dataset_name == "animals":
        anima_path = "datasets/Deforming4D/notexture/"
        specs["ShapePath"] = anima_path + args.object_name
    elif args.dataset_name =="defaust":
        defaust_path= "datasets/D_Faust/"
        specs["ShapePath"] = defaust_path + args.object_name
    
    elif args.dataset_name =="medical":
        medical_path= "datasets/Medical/"
        specs["ShapePath"] = medical_path + args.object_name
    else:
        raise ValueError(f"The variable must be in either animals,medical or defaust")
    
    # load_dataset
    if args.adjust_use == "True":
        ajust_weights = args.adjust_weights.split(',')
        ajust_coes = [float(ajust_weights[0]),float(ajust_weights[1]),float(ajust_weights[2])]
        sdf_dataset, experiment_directory,obj_name = load_dataset(args,specs,ajust_coes)
    else:
        sdf_dataset, experiment_directory,obj_name = load_dataset(args,specs)
    ##############################################################################
        
    start_idx = 0#
    end_idx = len(sdf_dataset)-1 
    
    # load_network_latents
    if args.pts_num!=None:
        save_pts_name = args.pts_num
    else:
        save_pts_name = "1"

    if args.adjust_use=="True":
        decoder, ajust,two_codes = load_network_model(args,specs,start_idx,end_idx)
        ajust,decoder,two_codes = load_model_latents(ajust,decoder,two_codes,experiment_directory,save_pts_name)

    else:
        if args.methods!=None:
            decoder,start_end_vecs  = load_network_model_others(args,specs,start_idx,end_idx)
        else:
            decoder,start_end_vecs = load_network_model(args,specs,start_idx,end_idx)
        decoder,start_end_vecs = load_model_latents_others(decoder,start_end_vecs,experiment_directory,save_pts_name)

    start_epoch = int(save_pts_name)
    num_epochs = specs["NumEpochs"]# + 10_000
    #print('total_num_epochs: ',num_epochs)
    train_lens = len(sdf_dataset)
    logging.info("There are {} time instances".format(train_lens))
    logging.info("starting from epoch {}".format(start_epoch))
    ####################Load Model Params#####################
    decoder.train()
    times = []
    print('Use Other Methods? ',args.methods)
    if args.adjust_use =="True":
        trainer = Trainer(args,specs,ajust,decoder,
                        sdf_dataset,
                        two_codes,
                        start_idx,end_idx,
                        ajust_coes,obj_name,experiment_directory)
    else:
        trainer = Trainer_woadjust(
                        args,
                        specs,
                        decoder,
                        sdf_dataset,
                        start_end_vecs,
                        start_idx,end_idx,
                        obj_name,experiment_directory)
    
    for epoch in range(start_epoch,num_epochs+1):
        t1 = time.time()
        log_dict = trainer.forward_epoch(epoch)
        t2 = time.time()

        times.append(t2-t1)
        wandb.log(log_dict)
        if epoch%100 == 0:
            mean_time = np.array(times).mean()
            logging.info("computed_time_per100 = {}".format(100*mean_time))
            times= []
        if epoch%5000==0:#200
            trainer.save_model(epoch,experiment_directory)
if __name__ == "__main__":
    setup_seed(3407)
    velocity_test()