#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:02:03 2023

@author: lcl
"""


from abc import ABC
import torch
import math
import util.load_object as lob
import encoder.encoder as enc

class FOCUSmodel(ABC):

    
    
    def __init__(self,args):
        ''' Initialize the FOCUS model 
        '''
        self.args=args
    
    def init_hyper_parameters(self,model_path):
        device = self.args.device
        width = 224
        height = 224


        # 3dmm data
        self.obj = lob.Object3DMM(model_path,device,is_crop = True)
        self.A =  torch.Tensor([[9.06*224/2, 0,  (width-1)/2.0, 0, 9.06*224/2, (height-1)/2.0, 0, 0, 1]]).view(-1, 3, 3).to(device) #intrinsic camera mat
        self.T_ini = torch.Tensor([0, 0, 1000]).to(device)   #camera translation(direction of conversion will be set by flg later)
        sh_ini = torch.zeros(3, 9,device=device)    #offset of spherical harmonics coefficient
        sh_ini[:, 0] = 0.7 * 2 * math.pi
        self.sh_ini = sh_ini.reshape(-1)
        
        return self.width,self.height,self.obj,self.A,self.T_ini,self.sh_ini
    
    def init_full_facemodel(self):
        current_path = os.getcwd()  
        model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
        self.obj_intact = lob.Object3DMM(model_path,device)
        self.triangles_intact = self.obj_intact.face.detach().to('cpu').numpy().T
    
    
    def init_models(self):
        
        if os.path.exists(self.args.pretrained_encnet_path):
            self.enc_net = torch.load(self.args.pretrained_encnet_path, map_location=self.args.device)
        else:
            self.enc_net = enc.FaceEncoder(self.obj).to(self.args.device)
        if self.ars.where_occmask.lower()=='unet':
            if os.path.exists(self.args.pretrained_unet_path):
                self.unet_for_mask = torch.load(self.args.pretrained_unet_path, map_location=self.args.device)
            else: print('No Segmentation Model loaded!')

    
    
    def data_to_device(self,data):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.args.device)
        return data

if __name__ == '__main__':
    import os
    import argparse
    
    current_path = os.getcwd()  
    model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
    par = argparse.ArgumentParser(description='MoFA')

    par.add_argument('--learning_rate',default=0.1,type=float,help='The learning rate')
    par.add_argument('--epochs',default=130,type=int,help='Total epochs')
    par.add_argument('--batch_size',default=12,type=int,help='Batch sizes')
    par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
    par.add_argument('--pretrained_model',default=00,type=int,help='Pretrained model')
    par.add_argument('--img_path',type=str,help='Root of the training samples')
    
    args = par.parse_args()
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device= device


    FOCUSModel = FOCUSmodel()
    width,height,obj,A,T_ini,sh_ini = FOCUSModel.init_hyper_parameters(args,model_path)