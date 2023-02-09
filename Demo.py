 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 10:42:21 2021

@author: li0005
"""

import torch
import os
import math
import util.util as util
import util.load_dataset_v2 as load_dataset
import util.load_object as lob
import renderer.rendering as ren
import encoder.encoder as enc
import cv2
import numpy as np
import argparse
import FOCUS_model.FOCUS_basic as FOCUSModel
from util.get_landmarks import get_landmarks_main
import time
par = argparse.ArgumentParser(description='Test: MoFA+UNet ')
par.add_argument('--pretrained_model_test',default='./MoFA_UNet_Save/MoFA_UNet_CelebAHQ/unet_200000.model',type=str,help='Path of the pre-trained model')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--batch_size',default=8,type=int,help='Batch size')
par.add_argument('--img_path',type=str,help='Root of the images')
#par.add_argument('--output_path',default = './reconstructed_imgs',type=str,help='Root to save samples')
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

args = par.parse_args()
args.where_occmask = 'UNet'
#output_path = (args.output_path +'/').replace('//','/')
args.img_path = (args.img_path +'/').replace('//','/')

GPU_no = args.gpu
trained_mofa_path = args.pretrained_model_test
device = torch.device("cuda:{}".format(GPU_no) if torch.cuda.is_available() else "cpu")
args.device=device
#parameters
batch = args.batch_size
width = height = 224

current_path = os.getcwd()  
model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
obj_intact = lob.Object3DMM(model_path,device)
FOCUSModel = FOCUSModel.FOCUSmodel(args)
width,height,obj_cropped,A,T_ini,sh_ini = FOCUSModel.init_hyper_parameters(model_path)

# 3dmm data
triangles_intact = obj_intact.face.detach().to('cpu').numpy().T
#renderer and encoder
render_net = ren.Renderer(32)   #block_size^2 pixels are simultaneously processed in renderer, lager block_size consumes lager memory
render_net_cropped = ren.Renderer(32) 

def occlusionPhotometricLossWithoutBackground(gt,rendered,fgmask,standardDeviation=0.043,backgroundStDevsFromMean=3.0):
	normalizer = (-3 / 2 * math.log(2 * math.pi) - 3 * math.log(standardDeviation))
	fullForegroundLogLikelihood = (torch.sum(torch.pow(gt - rendered,2), axis=1)) * -0.5 / standardDeviation / standardDeviation + normalizer
	uniformBackgroundLogLikelihood = math.pow(backgroundStDevsFromMean * standardDeviation, 2) * -0.5 / standardDeviation / standardDeviation + normalizer
	occlusionForegroundMask = fgmask * (fullForegroundLogLikelihood > uniformBackgroundLogLikelihood).type(torch.FloatTensor).cuda(util.device_ids[GPU_no ])
	foregroundLogLikelihood = occlusionForegroundMask*fullForegroundLogLikelihood
	lh = torch.mean(foregroundLogLikelihood)
	return -lh, occlusionForegroundMask


#main processing
#################################################################
def proc(images, occlusion_mode=False,landmark_visible_mask=None, valid_mask=None,image_org=None,where_occmask=None):
    #valid_mask: 1 indicating unoccluded part of faces, vice versa
	'''
	images: network_input
    landmarks: landmark ground truth
    render_mode: renderer mode
    occlusion mode: use occlusion robust loss
	'''
	shape_param, exp_param, color_param, camera_param, sh_param = enc_net(images)
	color_param *= 3    #adjust learning rate
	camera_param[:,:3] *= 0.3
	camera_param[:,5] *= 0.005
	shape_param[:,80:] *= 0 #ignore high dimensional component of BFM
	exp_param[:,64:] *= 0
	color_param[:,80:] *= 0
    
	#vertex, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param*0, color_param, camera_param, sh_param,obj,T_ini,sh_ini,False)
	nonexp_vertex_intact, color, R, T, sh_coef = enc.convert_params_noexp(shape_param, exp_param*0, color_param, camera_param, sh_param,obj_intact,T_ini,sh_ini,False)
	
	projected_vertex_intact, _,_, _, raster_image_intact, _ = render_net(obj_intact.face, nonexp_vertex_intact,color, sh_coef, A, R, T, images,ren.RASTERIZE_DIFFERENTIABLE_IMAGE,False, 5, True)
	lmring = nonexp_vertex_intact[:,:,obj_intact.ringnet_lm]
    
	vertex_cropped, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param, color_param, camera_param, sh_param,obj_cropped,T_ini,sh_ini,False)
	_, _, _, _, raster_image_fitted, raster_mask = render_net_cropped(obj_cropped.face, vertex_cropped,color, sh_coef, A, R, T, images,ren.RASTERIZE_DIFFERENTIABLE_IMAGE,False, 5, True)
    
	if where_occmask== 'Occrobust':
		rec_loss, occlusion_fg_mask = occlusionPhotometricLossWithoutBackground(images, raster_image_fitted, raster_mask)
	elif where_occmask== 'UNet':
		image_concatenated = torch.cat(( raster_image_fitted,images),axis = 1)
		unet_est_mask = unet_for_mask(image_concatenated)
		occlusion_fg_mask = raster_mask.unsqueeze(1)*unet_est_mask
	else:
		occlusion_fg_mask=False
	
	#util.show_tensor_images(raster_image_intact,lmring)
	raster_image_fitted = images*(1-raster_mask.unsqueeze(1))+raster_image_fitted*raster_mask.unsqueeze(1)
	return raster_image_fitted,raster_mask ,lmring,nonexp_vertex_intact,occlusion_fg_mask

#################################################################

def write_lm_txt(filename_save,lms_ringnet_temp):
    anno_file = open( filename_save,"w")
    
    lms  = lms_ringnet_temp.T.astype(np.int)
            
    str_temp = ''
    for i_temp in range(7):
        str_temp +='{} {} {}\n'.format(lms[i_temp,0],lms[i_temp,1],lms[i_temp,2])
    anno_file.write(str_temp)    
    anno_file.close()
    

    
    
enc_net = torch.load(trained_mofa_path.replace('/unet_','/enc_net_') , map_location='cuda:{}'.format(util.device_ids[GPU_no ]))
try:
    unet_model_path =trained_mofa_path.replace('enc_net_','unet_')
    unet_for_mask = torch.load(unet_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))
    where_mask='UNet'
except:
    where_mask='Occrobust'
print('loading encoder: ' + trained_mofa_path.replace('/unet_','/enc_net_')  )
print('Masks estimated by '+where_mask)

#if not os.path.exists(output_path):
#    os.mkdir(output_path)


img_path = args.img_path
save_name = time.time()
save_dir = './Results/{}'.format(save_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_recon_result = os.path.join(save_dir,'reconstruction_results')
if not os.path.exists(save_dir_recon_result):
    os.makedirs(save_dir_recon_result)
landmark_filepath = get_landmarks_main(img_path,save_dir) 
testset = load_dataset.CelebDataset(device,img_path,False,landmark_file=landmark_filepath)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,shuffle=False, num_workers=0)

im_iter=0
    

'''---------------------------------------------------
Reconstructed results, masks
---------------------------------------------------'''
with torch.no_grad():
    enc_net.eval()
    for i, data in enumerate(testloader, 0):
        data=FOCUSModel.data_to_device(data)

        lm=data['landmark']
        images=data['img']
        image_paths = data['filename'] 
        
        image_results,raster_mask,lmring,vertex_3d,occlusion_fg_mask= proc(images,where_occmask=where_mask)

        
        img_num,_,_,_ = image_results.shape
        for iter_save in range(img_num):
            
            path_save = image_paths[iter_save].replace(img_path,'')
            path_save = os.path.join(save_dir_recon_result,path_save)
            dir_save = os.path.split(path_save)[0]
            filepath_woext_path = os.path.splitext(path_save)[0]
            
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)
            
            
            img_result = image_results[iter_save].transpose(0,1).transpose(1,2).detach().to('cpu').numpy()* 255
            img_result = np.flip(img_result , 2)
            
            #Save target images
            img_org = images[iter_save].transpose(0,1).transpose(1,2).detach().to('cpu').numpy()*255
            img_org = np.flip(img_org, 2)
                                                 
            # Save segmentation masks
            img_mask =np.round(occlusion_fg_mask[iter_save].transpose(0,1).transpose(1,2).detach().to('cpu').numpy())*255
            img_mask = np.concatenate([img_mask,img_mask,img_mask],2)
            img_result = np.concatenate([img_org,img_result,img_mask],1)
            cv2.imwrite(filepath_woext_path+'_results.jpg',img_result)
                
            # Save landmarks for NoW evaluation
            lms_ringnet_temp = lmring.detach().to('cpu').numpy()[iter_save]
            write_lm_txt(filepath_woext_path+'.txt',lms_ringnet_temp)

            # Save vertices for NoW evaluation
            vertexes_temp = vertex_3d[iter_save].detach().to('cpu').numpy()
            util.write_obj_with_colors(filepath_woext_path+'.obj', vertexes_temp.T, triangles_intact)
            im_iter+=1
    print('{} images reconstructed'.format(im_iter))


