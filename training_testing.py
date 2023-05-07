

import torch
import torch.nn as nn
from os.path import isfile, isdir, join, splitext
from sklearn.metrics import classification_report
from torch.optim.swa_utils import *
import numpy as np
import importlib
from tqdm import tqdm
import copy


def training(args, model, train_loader, test_loader, Loss, data_weight ,optimizer, epochs, device, num_class, checkpoint_name, class_threshold=0.5,swa_model=None,pretrained_state=None,fold=None):

    # import model (in models folder, [args.model_name].py)
    model_import_str = 'models.' + args.model_name
    models = importlib.import_module(model_import_str)
    ## specify variables to store in checkpoints
    backup_vars = [
        'model','epoch', 'best_model_wts','best_evaluated_acc','best_epoch','optimizer',
        'train_acc','test_acc','test_acc_best','test_th_best','train_loss','test_loss',
        'scheduler']            
    if args.use_SWA_model:
        backup_vars += ['swa_scheduler']
    # various initialization
    model.to(device)
    best_model_wts = None #
    best_evaluated_acc = 0
    best_epoch=0
    train_acc = []
    test_acc = []
    test_acc_best = []
    test_th_best = []
    train_loss = []
    test_loss = []
    # initialize scheduler (Type given by args.scheduler)
    if  args.scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.lr_gamma)
    elif  args.scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr_min, args.Learning_rate, step_size_up=args.cyclic_epochs*len(train_loader),cycle_momentum=False)
    elif  args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.cyclic_epochs*len(train_loader), T_mult=args.cyclic_mult, eta_min=args.lr_min, last_epoch=- 1)
    
    # initialize SWA scheduler (if uses SWA)
    if args.use_SWA_model:
        parts = splitext(checkpoint_name)
        if args.swa_scheduler == 'SWALR':
            swa_scheduler = SWALR(optimizer, anneal_epochs=args.epoch_per_SWA, swa_lr= args.swa_lr)      
        elif args.swa_scheduler == 'CosineAnnealingWarmRestarts':
            swa_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epoch_per_SWA*len(train_loader), T_mult=1, eta_min=args.lr_min, last_epoch=- 1)

    # if pretrained_state exist, load from pretrained_state
    start_epoch=1
    if pretrained_state is not None:
        if isinstance(pretrained_state['model'],models.mymodel):
            model	=	pretrained_state['model']
        else:
            model.load_state_dict(pretrained_state['model'])
        model.to(device)
        if args.freeze_enc:
            for param in model.features.parameters():
                param.requires_grad = False
        else:
            for param in model.features.parameters():
                param.requires_grad = True

        # resume training
        #   (if args.restart is true, use model checkpoint only, and retart training)
        if not args.restart:
            exclude_vars = ['epoch']
            load_vars = list(set(backup_vars) - set(exclude_vars))
            start_epoch = pretrained_state['epoch'] + 1
            train_acc	=	pretrained_state['train_acc']
            test_acc	=	pretrained_state['test_acc']
            test_acc_best	=	pretrained_state['test_acc_best']
            test_th_best	=	pretrained_state['test_th_best']
            optimizer	=	pretrained_state['optimizer']
            best_evaluated_acc	=	pretrained_state['best_evaluated_acc']
            scheduler	=	pretrained_state['scheduler']
            best_model_wts	=	pretrained_state['best_model_wts']
            test_loss	=	pretrained_state['test_loss']
            train_loss	=	pretrained_state['train_loss']
            
            #swa_scheduler	=	pretrained_state['swa_scheduler']
            best_epoch	=	pretrained_state['best_epoch']
            
            if  pretrained_state['epoch'] >= args.SWA_start_epoch:
                if args.use_SWA_model:
                    swa_scheduler	=	pretrained_state['swa_scheduler']

                    swa_model.load_state_dict(pretrained_state['swa_dict'])
            if args.use_SWA_model:
                swa_model.to(device)
    ###### start training for
    batch_count = 0  #batch counter
    swa_batch_count = 0  
    #
    for epoch in range(start_epoch, epochs+1):
        print ("------------------Epoch %d --------------------"% (epoch))
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            ##
            batch = tqdm(train_loader,ncols=100)
            pred_all = np.array([])
            label_all = np.array([])
            torch.cuda.empty_cache()

            for idx,(data, label) in enumerate(batch): 
                torch.cuda.empty_cache()

                if batch_count == 0:
                    optimizer.zero_grad()
                data = data.to(device)
                label = label.to(device,dtype=torch.long)
                label = label.view(label.shape[0],-1)
                predict = model(data) 
                if not isinstance(predict,list): # normal case
                    logits = predict 
                    loss = Loss(logits, label.type_as(predict))
                else: # if args.num_test_aug > 1 (for ensemble prediction using test-time augmentation)
                    logits = [x for x in predict]
                    losses = [model.loss_w[i] *  Loss(logits[i], label.type_as(logits[i])) for i in range(len(logits))]
                    losses = torch.stack(losses)
                    loss = torch.sum(losses)
                    logits = torch.stack(logits).mean(dim=0)

                total_loss += loss.item()
                if logits.ndim == 2 or logits.shape[-1] == 1:
                    prob = torch.sigmoid(logits)
                    pred = torch.where(prob>class_threshold,1,0)
                else:
                    prob = torch.softmax(logits,dim=-1)
                    prob, pred = torch.max(prob,dim=-1)
                pred_all = np.concatenate((pred_all,pred.flatten().detach().cpu().numpy()))
                label_all = np.concatenate((label_all,label.flatten().detach().cpu().numpy()))
                loss.backward()
                batch.set_description_str('training loss:\t%.4f' % (total_loss / (idx+1)))
                if batch_count == 0:                    
                    optimizer.step()  #Update every N iteration (given by args.batch_per_update)
                batch_count = (batch_count + 1) % args.batch_per_update
                
                if args.use_SWA_model and epoch >= args.SWA_start_epoch:
                    swa_batch_count = (swa_batch_count + 1) % args.batch_per_SWA
                    if args.batch_per_SWA != 0 and swa_batch_count == 0:
                        swa_model.update_parameters(model)                                    
                # if SWA process has been started, update swa_scheduler
                if args.use_SWA_model and epoch >= args.SWA_start_epoch and not args.no_SWA_scheduler:
                    swa_scheduler.step()

                # some scheduler requires update for every iteration
                elif  args.scheduler == 'CosineAnnealingWarmRestarts' or args.scheduler == 'CyclicLR':
                        scheduler.step()

        # Update SWA if update by number of epochs    
        if args.use_SWA_model and epoch + 1 >= args.SWA_start_epoch : 
            if (epoch + 1 - args.SWA_start_epoch) % args.epoch_per_SWA == 0:
                if args.batch_per_SWA == 0:  # Update SWA if update by number of epochs
                    swa_model.update_parameters(model)
                torch.optim.swa_utils.update_bn(train_loader, swa_model,device=device)

        # calculate, store and show performance
        train_metrics = classification_report(label_all, pred_all,labels=[0,1],output_dict=True,zero_division=0)
        if args.use_SWA_model and epoch >= args.SWA_start_epoch:  
            eval_model = swa_model
        else:
            eval_model = model
        valid_metrics,test_loss_epoch,pred_all, prob_all,label_all,best_th,best_f1 = evaluate(args,
            eval_model, device, test_loader,Loss, class_threshold=args.class_threshold)  
        train_acc.append(train_metrics['0']['f1-score'])
        test_acc.append(valid_metrics['0']['f1-score'])
        test_acc_best.append(best_f1)
        test_th_best.append(best_th)
        train_loss.append(total_loss/(idx+1))
        test_loss.append(test_loss_epoch)
        accuracy = valid_metrics['0']['f1-score']
        
        print('training metrics:')
        print("\tClass 0:", train_metrics['0'])
        print("\tClass 1:", train_metrics['1'])
        print('testing metrics:')
        print("\tClass 0:", valid_metrics['0'])
        print("\tClass 1:", valid_metrics['1'])
        print ("---------------------------------------------------------")
        # Update scheduler for every epoch 
        #  (if scheduler is not CosineAnnealingWarmRestarts or CyclicLR, becasue they should be updated every iteration)
        if args.use_SWA_model and epoch >= args.SWA_start_epoch:
            pass
        elif not(  args.scheduler == 'CosineAnnealingWarmRestarts' or args.scheduler == 'CyclicLR'):
            if scheduler.get_last_lr()[0] >= args.lr_min:
                scheduler.step()      
        #save training curve info
        if fold is not None:
            traing_curve_filename = join(args.logdir,'training_curve_%s_%s_fold%d.npz' % (args.model_name,args.timestamp,fold))
        else:
            traing_curve_filename = join(args.logdir,'training_curve_%s_%s.npz' % (args.model_name,args.timestamp))
        np.savez(traing_curve_filename,        
            args = args,
            train_acc=train_acc,test_acc=test_acc,
            train_loss=train_loss,test_loss=test_loss,
            test_pred=pred_all, test_label=label_all,
            test_prob=prob_all,
            test_acc_best=test_acc_best,test_th_best=test_th_best,
            train_metrics=train_metrics['0'],test_metrics=valid_metrics['0'])
        if accuracy >= best_evaluated_acc:
            best_evaluated_acc = accuracy
            best_epoch = epoch
             
            if args.use_SWA_model and epoch >= args.SWA_start_epoch:
                best_model_wts = copy.deepcopy(swa_model.state_dict())
            else:
                best_model_wts = copy.deepcopy(model.state_dict())

        elif epoch - best_epoch > args.es_patience:
            print('No improvement for %d epochs. Stopping Early at epoch %d.' % (args.es_patience, epoch))
            break
            
        ## save checkpoint for fail-proofing
        state = {'args':args}
        for varname in backup_vars:
            exec("state[\"%s\"] = %s" %(varname,varname))
            #state[varname] = locals()[varname]               
        if args.use_SWA_model:
            state['swa_dict'] = swa_model.state_dict()    
        torch.save(state, checkpoint_name)
            

    return train_acc, test_acc, train_loss, test_loss



def evaluate(args, model, device, test_loader, Loss, class_threshold=0.5,N_thresholds=500):
    total_loss = 0
    batch = tqdm(test_loader,ncols=100)

    prob_all = []
    pred_all = []
    label_all = []
    thresholds = np.linspace(0,1,N_thresholds)


    with torch.set_grad_enabled(False):
        model.eval()
        if args.num_test_aug >1 and args.MC_dropout:
            for m in model.modules():
                if isinstance(m,nn.Dropout):
                    m.train()
        for idx,(data,label) in enumerate(batch):
            torch.cuda.empty_cache()
            if args.num_test_aug >1:
                data = torch.cat(data,dim=0)
                label = label[0].unsqueeze(0)
            data = data.to(device)
            label = label.to(device,dtype=torch.long)

            label = label.view(label.shape[0],-1)
            
            predict = model(data)
            
            
            if not isinstance(predict,list):
                logits = predict
            else:
                logits = predict[-1]

            if args.num_test_aug >1:
                if args.test_aug_agg == 'mean':
                    logits = torch.mean(logits,dim=0,keepdim=True)
                elif args.test_aug_agg == 'median':
                    logits = torch.quantile(logits,0.5,dim=0,keepdim=True)

            loss = Loss(logits, label.type_as(logits))

            total_loss += loss.item()
            if logits.ndim == 2 or logits.shape[-1] == 1:
                prob = torch.sigmoid(logits)
                pred = torch.where(prob>class_threshold,1,0)
            else:
                prob = torch.softmax(logits,dim=-1)
                prob, pred = torch.max(prob,dim=-1)
            pred_all.append(pred.view(-1,10,2).detach().cpu().numpy())
            prob_all.append(prob.view(-1,10,2).detach().cpu().numpy())
            label_all.append(label.view(-1,10,2).detach().cpu().numpy())
            batch.set_description_str('testing loss:\t%.4f' % (total_loss / (idx+1)))
        pred_all = np.concatenate(pred_all,axis=0).flatten()
        prob_all = np.concatenate(prob_all,axis=0).flatten()
        label_all = np.concatenate(label_all,axis=0).flatten()
        

        
        metrics = classification_report(label_all, pred_all,labels=[0,1],output_dict=True,zero_division=0)
        #calculate threshold with max F1
        th_metrics = np.array([classification_report(
            label_all, np.where(prob_all>th,1,0),
            labels=[0,1],output_dict=True,zero_division=0)['0']['f1-score']
            for th in thresholds])
        idx_best_th = np.argmax(th_metrics)
        best_f1 = th_metrics[idx_best_th]
        best_th = thresholds[idx_best_th]

    return  metrics, total_loss/(idx+1),pred_all, prob_all, label_all,best_th,best_f1

