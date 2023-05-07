

import torch
import torch.nn as nn
from os.path import isfile, isdir, join, splitext
from sklearn.metrics import classification_report
from torch.optim.swa_utils import *
import numpy as np
import importlib
from tqdm import tqdm
import copy
from pytorch_grad_cam_3d import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam_3d.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam_3d.utils.image import show_cam_on_image
from utils import ReverseModel, mask_to_bounding_box
from bagged_model import BaggedModels
import os

import matplotlib.pyplot as plt
from dataset import TorchDataset, TorchDatasetWithDWI
from skimage.util import montage
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_dilation, generate_binary_structure, binary_erosion
import scipy.io as sio

doc_pred_file = 'testing_doc_predictions.mat'

def visualize(args, model, device, 
    class_threshold=0.5,N_thresholds=500,cam_threhold=0.5):

    mat = sio.loadmat(doc_pred_file)

    test_augment = True if args.num_test_aug > 1 else False
    if args.CAM_on_DWI:
        DatasetClass = TorchDatasetWithDWI
    else:
        DatasetClass = TorchDataset
    test_data = DatasetClass(image_dir = args.valid_data,
        repeat=1,augment=test_augment,n_augs=args.num_test_aug,z_crop=args.z_crop,pre_align=args.pre_align,skull_strip=args.skull_strip,
        aug_Rrange=args.aug_Rrange, aug_Trange=args.aug_Trange,
        hflip=args.aug_hflip,vflip=args.aug_vflip)        
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,num_workers=0,pin_memory=False)

    ##
    if args.CAM_method == 'GradCAM':
        vis_class = GradCAM
    elif args.CAM_method == 'ScoreCAM':
        vis_class = ScoreCAM
    elif args.CAM_method == 'GradCAMPlusPlus':
        vis_class = GradCAMPlusPlus
    elif args.CAM_method == 'AblationCAM':
        vis_class = AblationCAM
    elif args.CAM_method == 'XGradCAM':
        vis_class = XGradCAM
    elif args.CAM_method == 'EigenCAM':
        vis_class = EigenCAM
    elif args.CAM_method == 'FullGrad':
        vis_class = FullGrad
    elif args.CAM_method == 'LayerCAM':
        vis_class = LayerCAM
    # eval_str = 'vis_class = ' + args.CAM_method
    # exec(eval_str)   
    if not isinstance(args.CAM_target_layer,list):
        vis_layers = [args.CAM_target_layer]
    else:
        vis_layers = args.CAM_target_layer
    outpath = os.path.join(args.CAM_dir,'%s_%s' % (args.model_name,args.timestamp))
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    if isinstance(model,torch.optim.swa_utils.AveragedModel):
        r_model = ReverseModel(model.module)
    else:
        r_model = ReverseModel(model)
    r_model.eval()
    if isinstance(model,BaggedModels):
        target_layers = []
        if isinstance(r_model.model.mdl_bag[0],AveragedModel):
            [target_layers.extend([mdl.module.features[idx] for idx in vis_layers])
                for mdl in r_model.model.mdl_bag]
        else:
            [target_layers.extend([mdl.features[idx] for idx in vis_layers])
                for mdl in r_model.model.mdl_bag]
    else:
        target_layers = [r_model.model.features[idx] for idx in vis_layers]
    cam = vis_class(model=r_model, target_layers=target_layers, use_cuda= torch.cuda.is_available())
    

    batch = tqdm(test_loader,ncols=100)

    prob_all = []
    pred_all = []
    label_all = []
    thresholds = np.linspace(0,1,N_thresholds)

    torch.cuda.empty_cache()

    with torch.set_grad_enabled(cam.uses_gradients):
        
        if args.num_test_aug >1 and args.MC_dropout:
            for m in cam.modules():
                if isinstance(m,nn.Dropout):
                    m.train()
        for idx,(data,label) in enumerate(batch):
            
            if args.CAM_on_DWI:
                DWI = data[:,-1,...].detach().cpu().numpy()
                data = data[:,:-1,...]

            torch.cuda.empty_cache()

            img_name = os.path.splitext(os.path.split(test_loader.dataset.image_list[idx])[1])[0]
            img_id = int(img_name)
            if args.num_test_aug >1:
                data = torch.cat(data,dim=0)
                label = label[0].unsqueeze(0)
            data = data.to(device)
            img = data[:,0,...].detach().cpu().numpy()
            label = label.to(device,dtype=torch.long)

            label = label.view(label.shape[0],-1)
            label_stroke = label.detach().cpu().numpy()
            label_stroke
            
            ## predict 
            predict= model(data)
            if not isinstance(predict,list):
                logits = predict
            else:
                logits = predict[-1]

            if args.num_test_aug >1:
                if args.test_aug_agg == 'mean':
                    logits = torch.mean(logits,dim=0,keepdim=True)
                elif args.test_aug_agg == 'median':
                    logits = torch.quantile(logits,0.5,dim=0,keepdim=True)

            #prob = torch.sigmoid(logits)
            #pred = torch.where(prob>class_threshold,1,0)
            if logits.ndim == 2 or logits.shape[-1] == 1:
                prob = torch.sigmoid(logits)
                pred = torch.where(prob>class_threshold,1,0)
            else:
                prob = torch.softmax(logits,dim=-1)
                prob, pred = torch.max(prob,dim=-1)
            ## draw heatmap on predicted stroke
            idx_stroke = np.argwhere(pred.flatten().cpu().numpy()==0).flatten()

            idx_TP = torch.where(torch.logical_and(pred.flatten()== 0, label.flatten() == 0))[0].cpu().numpy()
            idx_FP = torch.where(torch.logical_and(pred.flatten()== 0, label.flatten() == 1))[0].cpu().numpy()
            idx_FN = torch.where(torch.logical_and(pred.flatten()== 1, label.flatten() == 0))[0].cpu().numpy()
            AI_idx = {
                'TP': idx_TP,
                'FP': idx_FP,
                'FN': idx_FN,
            }
            ## for doctors
            pred_doc1 = mat['pred1'][img_id-1,:].flatten()
            pred_doc2 = mat['pred2'][img_id-1,:].flatten()
            label_docs = mat['GT'][img_id-1,:].flatten()
            a = np.concatenate((np.expand_dims(label_docs,axis=0),np.expand_dims(label.flatten().cpu().numpy(),axis=0)))
            
            idx_TP_doc1 = np.argwhere(np.logical_and(pred_doc1== 0, label_docs == 0)).flatten()
            idx_FP_doc1 = np.argwhere(np.logical_and(pred_doc1== 0, label_docs == 1)).flatten()
            idx_FN_doc1 = np.argwhere(np.logical_and(pred_doc1== 1, label_docs == 0)).flatten()
            doc1_idx = {
                'TP': idx_TP_doc1,
                'FP': idx_FP_doc1,
                'FN': idx_FN_doc1,
            }

            idx_TP_doc2 = np.argwhere(np.logical_and(pred_doc2== 0, label_docs == 0)).flatten()
            idx_FP_doc2 = np.argwhere(np.logical_and(pred_doc2== 0, label_docs == 1)).flatten()
            idx_FN_doc2 = np.argwhere(np.logical_and(pred_doc2== 1, label_docs == 0)).flatten()
            doc2_idx = {
                'TP': idx_TP_doc2,
                'FP': idx_FP_doc2,
                'FN': idx_FN_doc2,
            }
            ## for doctors
            #
            targets = [ClassifierOutputTarget(x) for x in idx_stroke]
            if not targets:      
                continue
            heatmap = cam(data,targets=targets)
            mask = data[:,2:,...].detach().cpu().numpy()
            brain = data[:,1,...].unsqueeze(1).detach().cpu().numpy()
            heatmaps = [cam(data,targets=[target]) for target in targets]
            heatmaps = np.stack(heatmaps,axis=1)
            bboxes =  mask_to_bounding_box( mask)
            if args.CAM_ROI_mask:
                masked_heatmaps = mask[:,idx_stroke,...] * heatmaps
            else:
                masked_heatmaps = brain * heatmaps

            #masked_heatmaps = bboxes[:,idx_stroke,...] * heatmaps
            masked_heatmaps = masked_heatmaps / np.max(masked_heatmaps,axis=(2,3,4),keepdims=True)            
            agg_heatmap = np.nansum(masked_heatmaps,axis=1)
            agg_heatmap = agg_heatmap / np.max(agg_heatmap,axis=(1,2,3),keepdims=True)     
            mask_stroke = mask[0,idx_stroke,...]

            # if args.CAM_on_DWI:
            #     base_img = DWI
            # else:
            #     base_img = img  
            thresholds = [0.05,0.25]
            min_alpha_list = [0.01,0.01]
            max_alpha_list = [0.99,0.99]
            max_alpha_at_list = [0.75,0.75]
            min_alpha = 0.01
            max_alpha = 0.9
            max_alpha_at = 0.75
            colormap='plasma'
            colormap='gnuplot'
            colormap='inferno'
            colormap='autumn'
            for cam_threhold,min_alpha,max_alpha,max_alpha_at in zip(thresholds,min_alpha_list,max_alpha_list,max_alpha_at_list):
                save_slices_w_doc(img,DWI,agg_heatmap,mask,AI_idx, doc1_idx,doc2_idx,outpath,img_name,args.CAM_method,
                     cam_threhold=cam_threhold,min_alpha=min_alpha,max_alpha = max_alpha,max_alpha_at=max_alpha_at,
                     colormap=colormap,
                     CAM_ROI_mask=args.CAM_ROI_mask,CAM_on_DWI=True,target_layer=vis_layers,
                     CAM_bag_sub=args.CAM_bag_sub
                     )
            pass
    return  


def visualize_bagged(args, model, device, 
    class_threshold=0.5,N_thresholds=500,cam_threhold=0.5):

    mat = sio.loadmat(doc_pred_file)

    test_augment = True if args.num_test_aug > 1 else False
    if args.CAM_on_DWI:
        DatasetClass = TorchDatasetWithDWI
    else:
        DatasetClass = TorchDataset
    test_data = DatasetClass(image_dir = args.valid_data,
        repeat=1,augment=test_augment,n_augs=args.num_test_aug,z_crop=args.z_crop,pre_align=args.pre_align,skull_strip=args.skull_strip,
        aug_Rrange=args.aug_Rrange, aug_Trange=args.aug_Trange,
        hflip=args.aug_hflip,vflip=args.aug_vflip)        
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,num_workers=0,pin_memory=False)

    ##
    if args.CAM_method == 'GradCAM':
        vis_class = GradCAM
    elif args.CAM_method == 'ScoreCAM':
        vis_class = ScoreCAM
    elif args.CAM_method == 'GradCAMPlusPlus':
        vis_class = GradCAMPlusPlus
    elif args.CAM_method == 'AblationCAM':
        vis_class = AblationCAM
    elif args.CAM_method == 'XGradCAM':
        vis_class = XGradCAM
    elif args.CAM_method == 'EigenCAM':
        vis_class = EigenCAM
    elif args.CAM_method == 'FullGrad':
        vis_class = FullGrad
    elif args.CAM_method == 'LayerCAM':
        vis_class = LayerCAM
    # eval_str = 'vis_class = ' + args.CAM_method
    # exec(eval_str)   
    if not isinstance(args.CAM_target_layer,list):
        vis_layers = [args.CAM_target_layer]
    else:
        vis_layers = args.CAM_target_layer
    outpath = os.path.join(args.CAM_dir,'%s_%s' % (args.model_name,args.timestamp))
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    if isinstance(model,torch.optim.swa_utils.AveragedModel):
        r_model = ReverseModel(model.module)
    else:
        r_model = ReverseModel(model)
    r_model.eval()

    ##
    target_layers = []
    if isinstance(r_model.model.mdl_bag[0],AveragedModel):
        [target_layers.extend([mdl.module.features[idx] for idx in vis_layers])
            for mdl in r_model.model.mdl_bag]
    else:
        [target_layers.extend([mdl.features[idx] for idx in vis_layers])
            for mdl in r_model.model.mdl_bag]
    
    cam = vis_class(model=r_model, target_layers=target_layers, use_cuda= torch.cuda.is_available())
    ##

    batch = tqdm(test_loader,ncols=100)

    prob_all = []
    pred_all = []
    label_all = []
    thresholds = np.linspace(0,1,N_thresholds)

    torch.cuda.empty_cache()

    with torch.set_grad_enabled(cam.uses_gradients):
        
        if args.num_test_aug >1 and args.MC_dropout:
            for m in cam.modules():
                if isinstance(m,nn.Dropout):
                    m.train()
        for idx,(data,label) in enumerate(batch):
            
            if args.CAM_on_DWI:
                DWI = data[:,-1,...].detach().cpu().numpy()
                data = data[:,:-1,...]

            torch.cuda.empty_cache()

            img_name = os.path.splitext(os.path.split(test_loader.dataset.image_list[idx])[1])[0]
            img_id = int(img_name)
            if args.num_test_aug >1:
                data = torch.cat(data,dim=0)
                label = label[0].unsqueeze(0)
            data = data.to(device)
            img = data[:,0,...].detach().cpu().numpy()
            label = label.to(device,dtype=torch.long)

            label = label.view(label.shape[0],-1)
            label_stroke = label.detach().cpu().numpy()
            label_stroke
            
            ## predict 
            predict, predict_single = model(data,return_list=True)
            if not isinstance(predict,list):
                logits = predict
            else:
                logits = predict[-1]

            if args.num_test_aug >1:
                if args.test_aug_agg == 'mean':
                    logits = torch.mean(logits,dim=0,keepdim=True)
                elif args.test_aug_agg == 'median':
                    logits = torch.quantile(logits,0.5,dim=0,keepdim=True)

            #prob = torch.sigmoid(logits)
            #pred = torch.where(prob>class_threshold,1,0)
            if logits.ndim == 2 or logits.shape[-1] == 1:
                prob = torch.sigmoid(logits)
                pred = torch.where(prob>class_threshold,1,0)
            else:
                prob = torch.softmax(logits,dim=-1)
                prob, pred = torch.max(prob,dim=-1)
            ## draw heatmap on predicted stroke
            idx_stroke = np.argwhere(pred.flatten().cpu().numpy()==0).flatten()
            if len(idx_stroke) == 0:
                continue
            ## get individual predictions
            predict_single = predict_single.view(-1,20)
            if not isinstance(predict,list):
                logits_single = predict_single
            else:
                logits_single = predict[-1]
            prob_single = torch.sigmoid(logits_single)
            pred_single = torch.where(prob_single>class_threshold,1,0)
            ####

            idx_TP = torch.where(torch.logical_and(pred.flatten()== 0, label.flatten() == 0))[0].cpu().numpy()
            idx_FP = torch.where(torch.logical_and(pred.flatten()== 0, label.flatten() == 1))[0].cpu().numpy()
            idx_FN = torch.where(torch.logical_and(pred.flatten()== 1, label.flatten() == 0))[0].cpu().numpy()

            AI_idx = {
                'TP': idx_TP,
                'FP': idx_FP,
                'FN': idx_FN,
            }
            ## for doctors
            pred_doc1 = mat['pred1'][img_id-1,:].flatten()
            pred_doc2 = mat['pred2'][img_id-1,:].flatten()
            label_docs = mat['GT'][img_id-1,:].flatten()
            a = np.concatenate((np.expand_dims(label_docs,axis=0),np.expand_dims(label.flatten().cpu().numpy(),axis=0)))
            
            idx_TP_doc1 = np.argwhere(np.logical_and(pred_doc1== 0, label_docs == 0)).flatten()
            idx_FP_doc1 = np.argwhere(np.logical_and(pred_doc1== 0, label_docs == 1)).flatten()
            idx_FN_doc1 = np.argwhere(np.logical_and(pred_doc1== 1, label_docs == 0)).flatten()
            doc1_idx = {
                'TP': idx_TP_doc1,
                'FP': idx_FP_doc1,
                'FN': idx_FN_doc1,
            }

            idx_TP_doc2 = np.argwhere(np.logical_and(pred_doc2== 0, label_docs == 0)).flatten()
            idx_FP_doc2 = np.argwhere(np.logical_and(pred_doc2== 0, label_docs == 1)).flatten()
            idx_FN_doc2 = np.argwhere(np.logical_and(pred_doc2== 1, label_docs == 0)).flatten()
            doc2_idx = {
                'TP': idx_TP_doc2,
                'FP': idx_FP_doc2,
                'FN': idx_FN_doc2,
            }
            ## for doctors
            #targets = [ClassifierOutputTarget(x) for x in idx_stroke]
            
            #heatmap = cam(data,targets=targets)
            mask = data[:,2:,...].detach().cpu().numpy()
            brain = data[:,1,...].unsqueeze(1).detach().cpu().numpy()
            heatmaps = []
            # for each stroke ROI
            for x in idx_stroke:
                target = ClassifierOutputTarget(x)
                pred_roi_single = pred_single[:,x]
                idx_single_stroke = np.argwhere(pred_roi_single.flatten().cpu().numpy()==0).flatten()


                ##
                target_layers = []
                if isinstance(r_model.model.mdl_bag[0],AveragedModel):
                    [target_layers.extend([r_model.model.mdl_bag[i].module.features[idx] for idx in vis_layers])
                        for i in idx_single_stroke]
                else:
                    [target_layers.extend([r_model.model.mdl_bag[i].features[idx] for idx in vis_layers])
                        for i in idx_single_stroke]
                cam = vis_class(model=r_model, target_layers=target_layers, use_cuda= torch.cuda.is_available())
                heatmaps.extend(cam(data,targets=[target]))
            ##

            #heatmaps = [cam(data,targets=[target]) for target in targets]
            heatmaps = np.expand_dims(np.stack(heatmaps,axis=0),axis=0)
            bboxes =  mask_to_bounding_box( mask)
            if args.CAM_ROI_mask:
                masked_heatmaps = mask[:,idx_stroke,...] * heatmaps
            else:
                masked_heatmaps = brain * heatmaps

            #masked_heatmaps = bboxes[:,idx_stroke,...] * heatmaps
            masked_heatmaps = masked_heatmaps / np.max(masked_heatmaps,axis=(2,3,4),keepdims=True)            
            agg_heatmap = np.nansum(masked_heatmaps,axis=1)
            agg_heatmap = agg_heatmap / np.max(agg_heatmap,axis=(1,2,2),keepdims=True)     
            mask_stroke = mask[0,idx_stroke,...]

            # if args.CAM_on_DWI:
            #     base_img = DWI
            # else:
            #     base_img = img  
            thresholds = [0.05]
            #thresholds = [0.05,0.25,0.5]

            min_alpha = 0.0
            max_alpha = 0.5
            max_alpha_at = 0.75
            colormap='plasma'
            colormap='gnuplot'
            colormap='inferno'
            for cam_threhold in thresholds:

                save_slices_w_doc(img,DWI,agg_heatmap,mask,AI_idx, doc1_idx,doc2_idx,outpath,img_name,args.CAM_method,
                     cam_threhold=cam_threhold,min_alpha=min_alpha,max_alpha = max_alpha,max_alpha_at=max_alpha_at,
                     colormap=colormap,
                     CAM_ROI_mask=args.CAM_ROI_mask,CAM_on_DWI=True,target_layer=vis_layers,
                     CAM_bag_sub=args.CAM_bag_sub
                     )
            pass
    return  

def save_slices_w_doc(img,DWI,heatmap,mask,AI_idx, doc1_idx,doc2_idx,outpath,img_name,CAM_method,
    cam_threhold=0.5,min_alpha=0.5,max_alpha=0.5,max_alpha_at=1.0,CAM_ROI_mask=False,CAM_on_DWI=False,CAM_bag_sub=False,
    transpose_img=True,
    target_layer=[2], colormap='plasma'):
    target_layer_str = ' '.join(map(str, target_layer))
    outfolder  = os.path.join(outpath,CAM_method,CAM_method)

    #idx_TP,idx_FP,idx_FN,
    ##
    if CAM_ROI_mask:
        outfolder = outfolder + '_ROIMasked'
    if CAM_on_DWI:
        outfolder = outfolder + '_OnDWI'
    if CAM_bag_sub:
        outfolder = outfolder + '_BagSubPre'
    outfolder = outfolder + ' slices(Layer %s, th %.2f)' % (target_layer_str,cam_threhold)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    low = cam_threhold
    mask_alpha_val=1.0
    mask_vmax=2.0
    
    img_vol = img[0,...]
    #img = montage(np.transpose(img,(2,0,1)))

    ##
    for ch in range(mask.shape[1]):
        for z in  range(mask.shape[4]):
            mask[0,ch,:,:,z] =  np.round( mask[0,ch,:,:,z]  - binary_erosion( mask[0,ch,:,:,z] ,iterations=1))
    ##
    heatmap_vol = heatmap[0,...]
    DWI_vol = DWI[0,...]
    mask_TP = mask[0,AI_idx['TP'],...]
    mask_FP = mask[0,AI_idx['FP'],...]
    mask_FN = mask[0,AI_idx['FN'],...]
    mask_TP_vol = np.sum(mask_TP,0)
    mask_FP_vol = np.sum(mask_FP,0)
    mask_FN_vol = np.sum(mask_FN,0)
    mask_TP_doc1 = mask[0,doc1_idx['TP'],...]
    mask_FP_doc1 = mask[0,doc1_idx['FP'],...]
    mask_FN_doc1 = mask[0,doc1_idx['FN'],...]
    mask_TP_doc1_vol = np.sum(mask_TP_doc1,0)
    mask_FP_doc1_vol = np.sum(mask_FP_doc1,0)
    mask_FN_doc1_vol = np.sum(mask_FN_doc1,0)

    mask_TP_doc2 = mask[0,doc2_idx['TP'],...]
    mask_FP_doc2 = mask[0,doc2_idx['FP'],...]
    mask_FN_doc2 = mask[0,doc2_idx['FN'],...]
    mask_TP_doc2_vol = np.sum(mask_TP_doc2,0)
    mask_FP_doc2_vol = np.sum(mask_FP_doc2,0)
    mask_FN_doc2_vol = np.sum(mask_FN_doc2,0)

    ##
    if transpose_img:
        img_vol = np.flip(np.transpose(img_vol,(1,0,2)),(0,1))
        heatmap_vol =  np.flip(np.transpose(heatmap_vol,(1,0,2)),(0,1))
        DWI_vol =  np.flip(np.transpose(DWI_vol,(1,0,2)),(0,1))
        mask_TP_vol =  np.flip(np.transpose(mask_TP_vol,(1,0,2)),(0,1))
        mask_FP_vol =  np.flip(np.transpose(mask_FP_vol,(1,0,2)),(0,1))
        mask_FN_vol =  np.flip(np.transpose(mask_FN_vol,(1,0,2)),(0,1))

        mask_TP_doc1_vol =  np.flip(np.transpose(mask_TP_doc1_vol,(1,0,2)),(0,1))
        mask_FP_doc1_vol =  np.flip(np.transpose(mask_FP_doc1_vol,(1,0,2)),(0,1))
        mask_FN_doc1_vol =  np.flip(np.transpose(mask_FN_doc1_vol,(1,0,2)),(0,1))

        mask_TP_doc2_vol =  np.flip(np.transpose(mask_TP_doc2_vol,(1,0,2)),(0,1))
        mask_FP_doc2_vol =  np.flip(np.transpose(mask_FP_doc2_vol,(1,0,2)),(0,1))
        mask_FN_doc2_vol =  np.flip(np.transpose(mask_FN_doc2_vol,(1,0,2)),(0,1))

    # mask_TP_vol = np.round(mask_TP - binary_erosion(mask_TP,iterations=1))
    # mask_FP_vol = np.round(mask_FP - binary_erosion(mask_FP,iterations=1))
    # mask_FN_vol = np.round(mask_FN - binary_erosion(mask_FN,iterations=2))
    for z in range(img.shape[-1]):    
        filename = os.path.join(outfolder,'%s_%s_slice%02d.png' % (CAM_method,img_name,z))
        
        img = img_vol[...,z]
        a = heatmap_vol[...,z]
        DWI = DWI_vol[...,z]
        mask_TP = mask_TP_vol[...,z]
        mask_FP = mask_FP_vol[...,z]
        mask_FN = mask_FN_vol[...,z]
        
        mask_TP_doc1 = mask_TP_doc1_vol[...,z]
        mask_FP_doc1 = mask_FP_doc1_vol[...,z]
        mask_FN_doc1 = mask_FN_doc1_vol[...,z]

        mask_TP_doc2 = mask_TP_doc2_vol[...,z]
        mask_FP_doc2 = mask_FP_doc2_vol[...,z]
        mask_FN_doc2 = mask_FN_doc2_vol[...,z]
        
        mask_TP = mask_TP_vol[...,z]
        mask_FP = mask_FP_vol[...,z]
        mask_FN = mask_FN_vol[...,z]
        # mask_TP = np.round(mask_TP - binary_erosion(mask_TP,iterations=1))
        # mask_FP = np.round(mask_FP - binary_erosion(mask_FP,iterations=1))
        # mask_FN = np.round(mask_FN - binary_erosion(mask_FN,iterations=1))
        alpha = np.where(a>low, np.maximum(a-low,0)/(max_alpha_at-low) * (max_alpha-min_alpha)+ min_alpha,0)
        alpha = np.where(alpha>max_alpha,max_alpha,alpha)

        alpha_TP = np.where(mask_TP>0.0,mask_alpha_val,0)
        alpha_FP = np.where(mask_FP>0.0,mask_alpha_val,0)
        alpha_FN = np.where(mask_FN>0.0,mask_alpha_val,0)
        
        
        alpha_TP_doc1 = np.where(mask_TP_doc1>0.0,mask_alpha_val,0)
        alpha_FP_doc1 = np.where(mask_FP_doc1>0.0,mask_alpha_val,0)
        alpha_FN_doc1 = np.where(mask_FN_doc1>0.0,mask_alpha_val,0)

        alpha_TP_doc2 = np.where(mask_TP_doc2>0.0,mask_alpha_val,0)
        alpha_FP_doc2 = np.where(mask_FP_doc2>0.0,mask_alpha_val,0)
        alpha_FN_doc2 = np.where(mask_FN_doc2>0.0,mask_alpha_val,0)
        mask_all = mask_TP + mask_FP + mask_FN

        if np.max(img) == 0 or np.max(mask_all) == 0:
            continue
        plt.figure(figsize=(15,10))
        plt.subplot(2,3,1)

        plt.imshow(img,cmap='gray')
        if np.max(a) > low:
            plt.imshow(a,alpha=alpha,vmin=low,vmax=1,cmap=colormap)
        plt.imshow(mask_TP,alpha=alpha_TP,cmap='Greens',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FP,alpha=alpha_FP,cmap='RdPu',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FN,alpha=alpha_FN,cmap='gray',vmin=0,vmax=1)
        plt.axis('off')
        plt.gca().set_title('Model Heatmap')

        plt.subplot(2,3,2)
        plt.imshow(img,cmap='gray')
        #plt.imshow(mask_TP,alpha=alpha_TP,cmap='Greens',vmin=0,vmax=mask_vmax)
        #plt.imshow(mask_FP,alpha=alpha_FP,cmap='RdPu',vmin=0,vmax=mask_vmax)
        #plt.imshow(mask_FN,alpha=alpha_FN,cmap='gray',vmin=0,vmax=1)
        plt.gca().set_title('Model pred on CT')
        plt.axis('off')
        plt.subplot(2,3,3)
        plt.imshow(DWI,cmap='gray')
        #plt.imshow(mask_TP,alpha=alpha_TP,cmap='Greens',vmin=0,vmax=mask_vmax)
        #plt.imshow(mask_FP,alpha=alpha_FP,cmap='RdPu',vmin=0,vmax=mask_vmax)
        #plt.imshow(mask_FN,alpha=alpha_FN,cmap='gray',vmin=0,vmax=1)
        plt.gca().set_title('Model pred on DWI')
        plt.axis('off')
        plt.subplot(2,3,4)
        plt.imshow(img,cmap='gray')
        plt.imshow(mask_TP_doc1,alpha=alpha_TP_doc1,cmap='Greens',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FP_doc1,alpha=alpha_FP_doc1,cmap='RdPu',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FN_doc1,alpha=alpha_FN_doc1,cmap='gray',vmin=0,vmax=1)
        plt.axis('off')
        plt.gca().set_title('Radiologist 1')
        plt.subplot(2,3,5)
        plt.imshow(img,cmap='gray')
        plt.imshow(mask_TP_doc2,alpha=alpha_TP_doc2,cmap='Greens',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FP_doc2,alpha=alpha_FP_doc2,cmap='RdPu',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FN_doc2,alpha=alpha_FN_doc2,cmap='gray',vmin=0,vmax=1)
        plt.axis('off')
        plt.gca().set_title('Radiologist 2')

        plt.savefig(filename)
        plt.close()
    pass

def save_slices(img,DWI,heatmap,mask,idx_TP,idx_FP,idx_FN,outpath,img_name,CAM_method,
    cam_threhold=0.5,min_alpha=0.5,max_alpha=0.5,max_alpha_at=1.0,CAM_ROI_mask=False,CAM_on_DWI=False,CAM_bag_sub=False,
    transpose_img=True,
    target_layer=[2], colormap='plasma'):
    target_layer_str = ' '.join(map(str, target_layer))
    outfolder  = os.path.join(outpath,CAM_method,CAM_method)


    ##
    if CAM_ROI_mask:
        outfolder = outfolder + '_ROIMasked'
    if CAM_on_DWI:
        outfolder = outfolder + '_OnDWI'
    if CAM_bag_sub:
        outfolder = outfolder + '_BagSubPre'
    outfolder = outfolder + ' slices(Layer %s, th %.2f)' % (target_layer_str,cam_threhold)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    low = cam_threhold
    mask_alpha_val=1.0
    mask_vmax=2.0
    
    img_vol = img[0,...]
    #img = montage(np.transpose(img,(2,0,1)))

    ##
    for ch in range(mask.shape[1]):
        for z in  range(mask.shape[4]):
            mask[0,ch,:,:,z] =  np.round( mask[0,ch,:,:,z]  - binary_erosion( mask[0,ch,:,:,z] ,iterations=1))
    ##
    heatmap_vol = heatmap[0,...]
    DWI_vol = DWI[0,...]
    mask_TP = mask[0,idx_TP,...]
    mask_FP = mask[0,idx_FP,...]
    mask_FN = mask[0,idx_FN,...]
    mask_TP_vol = np.sum(mask_TP,0)
    mask_FP_vol = np.sum(mask_FP,0)
    mask_FN_vol = np.sum(mask_FN,0)

    ##
    if transpose_img:
        img_vol = np.flip(np.transpose(img_vol,(1,0,2)),(0,1))
        heatmap_vol =  np.flip(np.transpose(heatmap_vol,(1,0,2)),(0,1))
        DWI_vol =  np.flip(np.transpose(DWI_vol,(1,0,2)),(0,1))
        mask_TP_vol =  np.flip(np.transpose(mask_TP_vol,(1,0,2)),(0,1))
        mask_FP_vol =  np.flip(np.transpose(mask_FP_vol,(1,0,2)),(0,1))
        mask_FN_vol =  np.flip(np.transpose(mask_FN_vol,(1,0,2)),(0,1))

    # mask_TP_vol = np.round(mask_TP - binary_erosion(mask_TP,iterations=1))
    # mask_FP_vol = np.round(mask_FP - binary_erosion(mask_FP,iterations=1))
    # mask_FN_vol = np.round(mask_FN - binary_erosion(mask_FN,iterations=2))
    for z in range(img.shape[-1]):    
        filename = os.path.join(outfolder,'%s_%s_slice%02d.png' % (CAM_method,img_name,z))
        
        img = img_vol[...,z]
        a = heatmap_vol[...,z]
        DWI = DWI_vol[...,z]
        mask_TP = mask_TP_vol[...,z]
        mask_FP = mask_FP_vol[...,z]
        mask_FN = mask_FN_vol[...,z]
        # mask_TP = np.round(mask_TP - binary_erosion(mask_TP,iterations=1))
        # mask_FP = np.round(mask_FP - binary_erosion(mask_FP,iterations=1))
        # mask_FN = np.round(mask_FN - binary_erosion(mask_FN,iterations=1))
        alpha = np.where(a>low, np.maximum(a-low,0)/(max_alpha_at-low) * (max_alpha-min_alpha)+ min_alpha,0)
        alpha = np.where(alpha>max_alpha,max_alpha,alpha)

        alpha_TP = np.where(mask_TP>0.0,mask_alpha_val,0)
        alpha_FP = np.where(mask_FP>0.0,mask_alpha_val,0)
        alpha_FN = np.where(mask_FN>0.0,mask_alpha_val,0)
        mask_all = mask_TP + mask_FP + mask_FN

        if np.max(img) == 0 or np.max(mask_all) == 0:
            continue
        plt.figure(figsize=(18,6))
        plt.subplot(1,3,1)
        plt.imshow(img,cmap='gray')
        if np.max(a) > low:
            plt.imshow(a,alpha=alpha,vmin=low,vmax=1,cmap=colormap)
        plt.imshow(mask_TP,alpha=alpha_TP,cmap='Greens',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FP,alpha=alpha_FP,cmap='RdPu',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FN,alpha=alpha_FN,cmap='gray',vmin=0,vmax=1)
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(img,cmap='gray')
        plt.imshow(mask_TP,alpha=alpha_TP,cmap='Greens',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FP,alpha=alpha_FP,cmap='RdPu',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FN,alpha=alpha_FN,cmap='gray',vmin=0,vmax=1)
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(DWI,cmap='gray')
        plt.imshow(mask_TP,alpha=alpha_TP,cmap='Greens',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FP,alpha=alpha_FP,cmap='RdPu',vmin=0,vmax=mask_vmax)
        plt.imshow(mask_FN,alpha=alpha_FN,cmap='gray',vmin=0,vmax=1)
        plt.axis('off')

        plt.savefig(filename,dpi=200)
        plt.close()
    pass

def save_montage(img,heatmap,mask,idx_TP,idx_FP,idx_FN,outpath,img_name,CAM_method,
    cam_threhold=0.5,min_alpha=0.5,max_alpha=0.5,CAM_ROI_mask=False,CAM_on_DWI=False,CAM_bag_sub=False,
    target_layer=[2], colormap='plasma'):
    target_layer_str = ' '.join(map(str, target_layer))
    outfolder  = os.path.join(outpath,CAM_method,CAM_method)
    if CAM_ROI_mask:
        outfolder = outfolder + '_ROIMasked'
    if CAM_bag_sub:
        outfolder = outfolder + '_BagSubPre'
    outfolder = outfolder + ' (Layer %s, th %.2f)' % (target_layer_str,cam_threhold)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    filename = os.path.join(outfolder,'%s_%s.png' % (CAM_method,img_name))
    low = cam_threhold
    mask_alpha_val=1.0
    mask_vmax=2.0
    
    img = img[0,...]
    img = montage(np.transpose(img,(2,0,1)))
    a = heatmap[0,...]
    a = montage(np.transpose(a,(2,0,1)))
    alpha = np.where(a>low, np.maximum(a-low,0)/(1-low) * (max_alpha-min_alpha)+ min_alpha,0)
    mask_TP = mask[0,idx_TP,...]
    mask_FP = mask[0,idx_FP,...]
    mask_FN = mask[0,idx_FN,...]
    mask_TP = np.sum(mask_TP,0)
    mask_FP = np.sum(mask_FP,0)
    mask_FN = np.sum(mask_FN,0)
    mask_FN = montage(np.transpose(mask_FN,(2,0,1)))
    mask_TP = montage(np.transpose(mask_TP,(2,0,1)))
    mask_FP = montage(np.transpose(mask_FP,(2,0,1)))
    mask_TP = np.round(mask_TP - binary_erosion(mask_TP,iterations=2))
    mask_FP = np.round(mask_FP - binary_erosion(mask_FP,iterations=2))
    mask_FN = np.round(mask_FN - binary_erosion(mask_FN,iterations=2))

    alpha_TP = np.where(mask_TP>0.0,mask_alpha_val,0)
    alpha_FP = np.where(mask_FP>0.0,mask_alpha_val,0)
    alpha_FN = np.where(mask_FN>0.0,mask_alpha_val,0)
    
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.imshow(a,alpha=alpha,vmin=low,cmap=colormap)
    plt.imshow(mask_TP,alpha=alpha_TP,cmap='Greens',vmin=0,vmax=mask_vmax)
    plt.imshow(mask_FP,alpha=alpha_FP,cmap='RdPu',vmin=0,vmax=mask_vmax)
    plt.imshow(mask_FN,alpha=alpha_FN,cmap='gray',vmin=0,vmax=1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img,cmap='gray')
    plt.imshow(mask_TP,alpha=alpha_TP,cmap='Greens',vmin=0,vmax=mask_vmax)
    plt.imshow(mask_FP,alpha=alpha_FP,cmap='RdPu',vmin=0,vmax=mask_vmax)
    plt.imshow(mask_FN,alpha=alpha_FN,cmap='gray',vmin=0,vmax=1)
    plt.axis('off')

    plt.savefig(filename,dpi=200)
    plt.close()



