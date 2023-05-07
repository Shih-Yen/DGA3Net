from parse import *
import torch
import pandas as pd
import argparse
import numpy as np
import scipy.io
from PIL import *
import os
from glob import glob
from os.path import join
import importlib
from torch.utils.data import DataLoader
import copy
from utils import seed_everything
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
from torch.optim.swa_utils import *
from lossfun import *
from bagged_model import BaggedModels
from performance import *
from dataset import TorchDataset
from training_testing import *
from visualization import visualize, visualize_bagged
from models.DGA3Net import DGA3Net


def get_Kfold_index(dataset,n_splits=10,random_state=None):
    #Statified group k fold based on number of stroke ROIs
    skf = StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    labels = np.stack(dataset.label_list,axis=0)
    subj_idx = np.arange(len(dataset))
    subj_idx = np.tile(subj_idx,(1,*labels.shape[1:])).flatten()
    fold_idx = []
    fold_iter = skf.split(labels.flatten(), labels.flatten(), subj_idx)
    for train_idx, test_idx in fold_iter:
        subj_train = np.unique(subj_idx[train_idx])
        subj_test = np.unique(subj_idx[test_idx])
        fold_idx.append((subj_train,subj_test))
    return fold_idx

def get_bag_performance():
    # Use ensemble of  CV models (using arg.train_data) to predict  arg.valid_data
    test_data =  TorchDataset(image_dir = args.valid_data,
            repeat=1,augment=True if args.num_test_aug > 1 else False,z_crop=args.z_crop,dtype=dtype,pre_align=args.pre_align,
            n_augs=args.num_test_aug,
            aug_Rrange=args.aug_Rrange, aug_Trange=args.aug_Trange,
            hflip=args.aug_hflip,vflip=args.aug_vflip)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)    
    search_str = os.path.join(args.logdir,'vol_checkpoint_*.pt')
    trained_models = glob(search_str)
    bag_model = BaggedModels(DGA3Net,trained_models).cuda()

    
    ## Output CAM
    if args.output_CAM:
        if args.CAM_bag_sub:
            visualize_bagged(args,bag_model, device, class_threshold=args.class_threshold,cam_threhold=args.CAM_threshold)
        else:
            visualize(args,bag_model, device, class_threshold=args.class_threshold,cam_threhold=args.CAM_threshold)
        
    accuracy, total_loss,pred_all, prob_all, label_all,best_th,best_f1 = evaluate(args,bag_model, device, test_loader,Loss,class_threshold=args.class_threshold)

    print(accuracy)
    #
    
    test_pred_filename = join(args.logdir,'ens_test_pred_%s_%s.npz' % (args.model_name,args.timestamp))
    np.savez(test_pred_filename ,test_pred = pred_all,test_label = label_all,test_prob=prob_all)

    test_pred_mat_filename = join(args.logdir,'ens_test_pred_%s_%s.mat' % (args.model_name,args.timestamp))
    scipy.io.savemat(test_pred_mat_filename ,{'pred':np.reshape(pred_all,(-1,10,2)),'label':np.reshape(label_all,(-1,10,2)),'prob': np.reshape(prob_all,(-1,10,2))})
    #
    CR = classification_report_with_ASPECTS(label_all,pred_all,prob_all)

    CR_list_singleR = get_single_region_performance(label_all,pred_all,prob_all)
    CR_list = {
        key: summarize_performance(CR[key])
        for key in CR.keys()}
    CRs_list_single = {
         key: summarize_single_region_performance([CRs[key] for CRs in CR_list_singleR])
         for key in CR_list_singleR[0].keys()}
    CR_list = {**CR_list, **CRs_list_single}

    
        
    summary_filename = join(args.logdir,'ens_summary_%s_%s.xlsx' % (args.model_name,args.timestamp))
    """ 
    for key in CR_list.keys():
        CR_list[key].to_excel(summary_filename,sheet_name=key)
    """
    with pd.ExcelWriter(summary_filename) as writer:  
        for key in CR_list.keys():
            CR_list[key].to_excel(writer,sheet_name=key)


def export_CV_performance():
    
    test_pred_filenames = [join(args.logdir,'test_pred_%s_%s_fold%d.npz' % (args.model_name,args.timestamp,fold)) for fold in range(1,args.num_folds+1)]
    
    npzfiles = [np.load(x) for x in test_pred_filenames]
    CRdict_list = [classification_report_with_ASPECTS(npzfile['test_label'],npzfile['test_pred'],npzfile['test_prob'],ASPECTS_threshold=6) for npzfile in npzfiles]
    CRs_list = {
        key: summarize_fold_performance([CRs[key] for CRs in CRdict_list])
        for key in CRdict_list[0].keys()}
    
    summary_filename = join(args.logdir,'summary_%s_%s.xlsx' % (args.model_name,args.timestamp))
    """ if os.path.isfile(summary_filename):
        os.remove(summary_filename) """

    with pd.ExcelWriter(summary_filename) as writer:  
        for key in CRs_list.keys():
            CRs_list[key].to_excel(writer,sheet_name=key)



if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-train_data" , type = str , default=r'F:\CT_project_SY\wholebrain_CT_pytorch_dataset_ASPECTSCrop_DS_with_trans\train')
    parser.add_argument("-valid_data" , type = str , default=r'F:\CT_project_SY\wholebrain_CT_pytorch_dataset_ASPECTSCrop_DS_with_trans\test')
    parser.add_argument('-pre_align', action='store_true', default=False, help='Rotate the image to center in data prerocessing')
    parser.add_argument('-no_augment', action='store_true', default=False, help='Disable data augmentation')
    parser.add_argument("-use_pretrained", default=False,action='store_true', help="Use pretrained model")
    parser.add_argument('-half_precision', action='store_true', default=False, help='use float16')
    parser.add_argument('-skull_strip', action='store_true', default=False, help='do skull-stripping in preprocessing')


    parser.add_argument("-z_crop", default=False,action='store_true', help="Only retain ASPECTS slices")
    parser.add_argument("-logdir" , type = str , default='cpts')
    parser.add_argument("-trained_model" , type = str , default='vol_checkpoint_WBmodel_ndSE_SEres7altBN_1ch_alt_20220505_2319.pt')
    parser.add_argument("-restart", default=False,action='store_true', help="Redo training using only pretrained model params (no other training states).")
    parser.add_argument("-freeze_enc", default=False,action='store_true', help="Freeze encoder.")
    parser.add_argument("-trained_swa_model" , type = str , default=None)

    
    parser.add_argument("-scheduler", type=str, default='ExponentialLR', help='LR scheduler class')
    parser.add_argument("-cyclic_epochs",type = int , default = 1)
    parser.add_argument("-cyclic_mult",type = int , default = 2)
    parser.add_argument("-seed" , type = int , default = 0)
    parser.add_argument("-num_class" , type = int , default = 2)
    parser.add_argument("-num_folds" , type = int , default = 10)
    parser.add_argument("-start_K" , type = int , default = 1)
    parser.add_argument("-epoch" , type = int , default = 100)
    parser.add_argument("-es_patience" , type = int , default = 300)
    parser.add_argument("-batch_size" , type = int , default = 3)
    parser.add_argument("-loss_beta" , type = float , default = 0.999)
    
    
    parser.add_argument("-aug_Rrange" , type = float , default = 15)
    parser.add_argument("-aug_Trange" , type = float , default = 0.02)
    parser.add_argument('-aug_hflip', action='store_true', default=False, help='random horizontal flip')
    parser.add_argument('-aug_vflip', action='store_true', default=False, help='random vertical flip')
    parser.add_argument('-MC_dropout', action='store_true', default=False, help='random vertical flip')
    parser.add_argument("-batch_per_update" , type = int , default = 1)
    parser.add_argument("-num_test_aug" , type = int , default = 1)
    parser.add_argument('-test_aug_agg', type=str, default='mean', help='voting method')
    parser.add_argument("-Learning_rate" , type = float , default = 5e-4)
    parser.add_argument("-lr_gamma" , type = float , default = 0.977)
    parser.add_argument("-momentum" , type = float , default = 0.9)
    parser.add_argument("-weight_decay" , type = float , default = 5e-9)
    parser.add_argument("-class_threshold" , type = float , default = 0.5)
    
    parser.add_argument('-use_SWA_model', action='store_true', default=False, help='Perform validation/testing through moving average model')
    parser.add_argument('-no_SWA_scheduler', action='store_true', default=False, help='use regular scheduler instead of SWA scheduler')
    parser.add_argument('-epoch_per_SWA', type=int, default=1, help='Epochs for SWA.')
    parser.add_argument('-batch_per_SWA', type=int, default=0, help='Iteration for SWA. If nonzero, update by number of iters instead of epochs')
    
    parser.add_argument('-optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('-swa_scheduler', type=str, default='SWALR', help='SWA scheduler (options: SWALR, CosineAnnealingWarmRestarts)')
    parser.add_argument('-swa_lr', type=float, default=5e-5, help='Initial learning rate for SWA')
    parser.add_argument('-lr_min', type=float, default=1e-6, help='minimum learning rate (for learning rate decay).')
    parser.add_argument('-SWA_start_epoch', type=int, default=175, help='Start of SWA')
    parser.add_argument('-lossfun', type=str, default='BinaryFocalLoss_2', help='Loss Function')
    parser.add_argument('-loss_kwarg_keys', type=str, default=[], nargs='*', help='names for specifying loss arg')
    parser.add_argument('-loss_kwarg_vals', type=float, default=[], nargs='*', help='values for specifying loss arg')
    parser.add_argument('-model_kwarg_keys', type=str, default=[], nargs='*', help='names for specifying model arg')
    parser.add_argument('-model_kwarg_vals', type=int, default=[], nargs='*', help='values for specifying model arg')
    parser.add_argument('-output_CAM', action='store_true', default=False, help='Also output CAM')
    parser.add_argument('-CAM_method', type=str, default='GradCAM', help='GradCAM type (see pytorch_grad_cam_3d. options: GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad)')
    parser.add_argument('-CAM_target_layer', type=int, default=[2], nargs='*', help='target layer')
    parser.add_argument('-CAM_threshold', type=float, default=0.5, help='threshold for CAM visualization')
    parser.add_argument('-CAM_dir', type=str, default='visualization_CV', help='dir for CAM result')
    parser.add_argument('-CAM_ROI_mask', action='store_true', default=False, help='Also output CAM')
    parser.add_argument('-CAM_bag_sub', action='store_true', default=False, help='Only output heatmap from subclassifiers with the target prediction')
    parser.add_argument('-CAM_on_DWI', action='store_true', default=False, help='Draw CAM on DWI (dataset has to contain DWI)')


    
    args = parser.parse_args()
    print(args)

    # create log dir
    args.trained_model = join(args.logdir,args.trained_model )
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    eval_str = 'lossfun = ' + args.lossfun
    exec(eval_str)   

    
    # get timestamp
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M")
    args.timestamp=dt_string
    

    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device:
        print ("GPU is ready")

    # if use args.half_precision (NOT RECOMMANDED)
    if args.half_precision:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # initialize by RNG by  args.seed
    seed_everything(seed=args.seed)

    # initialize dataset
    main_data =  TorchDataset(image_dir = args.train_data,
            repeat=1,augment=not args.no_augment,z_crop=args.z_crop,dtype=dtype,pre_align=args.pre_align,
            aug_Rrange=args.aug_Rrange, aug_Trange=args.aug_Trange,
            hflip=args.aug_hflip,vflip=args.aug_vflip)    
    print("Main dataset:\n", main_data)

    # initialize Cv index
    kfold_idx = get_Kfold_index(main_data,n_splits=args.num_folds,random_state=args.seed)
    #skf = KFold(n_splits=args.num_folds,shuffle=True,random_state=args.seed)
    K = 0

    #### Get checkpoint for the most recent fold
    pretrained_state=None
    if args.use_pretrained:
        search_str = os.path.join(args.logdir,'vol_checkpoint_*.pt')
        trained_models = glob(search_str)
        if len(trained_models) > 0:        
            checkpoint_name = [os.path.split(x)[1] for x in trained_models]
            trained_fold = [int(x.replace('.pt','').split('_')[-1].replace('fold','')) for x in trained_models]

            idx = np.argmax(trained_fold)
            # if there is also test_pred file, fold is already finished. move to the next one,
            search_str = os.path.join(args.logdir,'test_pred_*_fold%d.npz' % trained_fold[idx])
            test_pred = glob(search_str)
            if len(test_pred) > 0:       
                args.trained_model = trained_models[idx]  
                args.start_K = trained_fold[idx] + 1
                a = args.trained_model.replace('.','_').split('_')
                dt_string = '%s_%s' % (a[-4],a[-3])
                args.timestamp=dt_string
            else:
                args.trained_model = trained_models[idx] 
                pretrained_state = torch.load(args.trained_model)
                #if 'fold' in pretrained_state.keys():

                print('Checkpoint_exist. Resuming from Fold %d' % trained_fold[idx])
                args.start_K = trained_fold[idx] 
                        
                a = args.trained_model.replace('.','_').split('_')
                dt_string = '%s_%s' % (a[-4],a[-3])
                args.timestamp=dt_string
        else:
            pretrained_state=None
    else:
        pretrained_state = None

    ###### Start cross validation
    for train_index, test_index in kfold_idx:
        K += 1
        if K < args.start_K: # Skip finished fold
            continue
        print('=====================\tFold %d\t======================='%K)

        # initialize dataset
        train_data = copy.deepcopy(main_data)
        valid_data = copy.deepcopy(main_data)
        train_data.filter(train_index)
        valid_data.filter(test_index)
        if  args.num_test_aug ==1:
            valid_data.no_augment()
        valid_data.n_augs = args.num_test_aug
        
        print("Training dataset:\n", train_data)
        print("Validation dataset:\n", valid_data)
        
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
        if args.num_test_aug > 1:
            valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
        else:
            valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)
        ##########
        print ("Train Start")
        model_kwargs = dict(zip(args.model_kwarg_keys, args.model_kwarg_vals))
        model = DGA3Net(**model_kwargs)
        if args.half_precision:
            model.half()
        
        if args.use_SWA_model:
            global swa_model         
            swa_model = AveragedModel(model,device)
        else:
            swa_model = None


        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('# of params: %.3fM' % (float(num_params)/1e6))

        ############# calculate class balance weights #########################
        beta = [args.loss_beta, args.loss_beta]
        normal_data_num, detect_data_num = train_data.calculate_slice()
        class_balance_weights = [(1-beta[0])/(1-pow(beta[0],detect_data_num)), (1-beta[1])/(1-pow(beta[1],normal_data_num))]
        alpha = 1/(class_balance_weights[0]+class_balance_weights[1])
        data_weight = [class_balance_weights[0]*alpha, class_balance_weights[1]*alpha] ### class weight for CBFL

        ##### Initializing loss function
        loss_kwargs = dict(zip(args.loss_kwarg_keys, args.loss_kwarg_vals))
        #locals()['lossfun'] = args.lossfun
        if 'num_class' not in model_kwargs.keys():
            data_weight = data_weight[1]
        elif model_kwargs['num_class'] == 1:
            data_weight = data_weight[1]
        Loss = lossfun(alpha=data_weight,**loss_kwargs)
        
        
        print ("p",data_weight)
        
        ##### Initializing optimizer and swa_optimizer
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.Learning_rate,weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.Learning_rate,weight_decay=args.weight_decay)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.Learning_rate,weight_decay=args.weight_decay)
        swa_optimizer = torch.optim.SGD(model.parameters(), lr=args.Learning_rate,weight_decay=args.weight_decay)
        ###################
        checkpoint_name =  'vol_checkpoint_%s_%s_fold%d.pt' % ( args.model_name, args.timestamp,K)
        checkpoint_name = join(args.logdir,checkpoint_name)

        ##### Training process
        train_acc , valid_acc, train_loss , valid_loss  = training(args,
                model, train_loader, valid_loader, Loss, data_weight,
                optimizer,args.epoch, device, args.num_class, checkpoint_name,
                class_threshold=args.class_threshold,swa_model=swa_model,pretrained_state=pretrained_state,fold=K)
        pretrained_state = None
        ##### calculate final performance from checkpoint
        if args.epoch > 0:
            checkpoint_state = torch.load(checkpoint_name)
        else:
            checkpoint_state = pretrained_state
        if args.use_SWA_model and checkpoint_state['epoch'] >= args.SWA_start_epoch:
            best_model = swa_model
            
            best_model.load_state_dict(checkpoint_state['swa_dict'])
        else:   
            try:
                if isinstance(checkpoint_state['model'],DGA3Net):
                    best_model	=	checkpoint_state['model']
                else:
                    best_model = DGA3Net(**model_kwargs)
                    best_model.load_state_dict(checkpoint_state['model'])
            except:
                best_model = swa_model
                best_model.load_state_dict(checkpoint_state['model'])

        best_model.to(device)
        best_model.eval()
        accuracy, total_loss,pred_all, prob_all, label_all,best_th,best_f1 = evaluate(args,best_model, device, valid_loader,Loss,class_threshold=args.class_threshold)
        test_pred_filename = join(args.logdir,'test_pred_%s_%s_fold%d.npz' % (args.model_name,args.timestamp,K))
        np.savez(test_pred_filename ,test_pred = pred_all,test_label = label_all,test_prob=prob_all)
        print ("Accuracy : " , accuracy ,"%")
    ## summarize_performance
    Loss = lossfun()
    get_bag_performance()
    export_CV_performance()
