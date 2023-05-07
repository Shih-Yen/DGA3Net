
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, cohen_kappa_score
from  scipy.stats import pearsonr
import pandas as pd

ROIs = ['Insula',  'Internal_Capsule', 'Caudate', 'Putamen','M1', 'M2', 'M3', 'M4', 'M5', 'M6']
def exclude_fields(CR,exclude_keys = ['accuracy', 'micro avg']):
    keys = list(CR.keys())
    for key in keys:
        remove = [kw in key for kw in exclude_keys]
        if True in remove:
            CR.pop(key)
    return CR

def classification_report_ASPECTS(label,prob,ASPECTS_threshold=6):
    #label = np.sum(label,axis=-1)
    label = np.where(label >=ASPECTS_threshold,1,0)
    pred = np.where(prob >=ASPECTS_threshold,1,0)
    CR = classification_report(
        label.flatten(), pred.flatten(),
        labels=[0,1],output_dict=True,zero_division=0,digits=4)

    try:
        auc = roc_auc_score(label.flatten(),prob.flatten())
    except: # if only one class in one side
        auc = np.nan
    kappa = cohen_kappa_score(label.flatten(),pred.flatten())
    R = pearsonr(label.flatten(),prob.flatten())
    CR['auc'] = auc
    CR['kappa'] = kappa
    CR['PCorr'] = R[0]
    return CR

def classification_report_regional(label,pred,prob):
    CR = classification_report(
        label.flatten(), pred.flatten(),
        labels=[0,1],output_dict=True,zero_division=0)
    try:
        auc = roc_auc_score(label.flatten(),prob.flatten())
    except: # if only one class in one side
        auc = np.nan
    kappa = cohen_kappa_score(label.flatten(),pred.flatten())
    #R = pearsonr(label.flatten(),prob.flatten())
    CR['auc'] = auc
    CR['kappa'] = kappa
    return CR


def classification_report_single_region(label,pred,prob,region=0,both_size=True):
    
    prob = prob.reshape((-1,10,2))
    pred = pred.reshape((-1,10,2))
    label = label.reshape((-1,10,2))
    prob = prob[:,region,:]
    pred = pred[:,region,:]
    label = label[:,region,:]

    CR = classification_report_regional(label,pred,prob)

    CR_R = classification_report_regional(label[:,0],pred[:,0],prob[:,0])
    CR_L = classification_report_regional(label[:,1],pred[:,1],prob[:,1])
    CR_dict = {'SingleRegions': CR, 'SingleRegions_L': CR_L, 'SingleRegions_R': CR_R,  }
    return CR_dict


def get_known_side(label_all,pred_all,prob_all):
    prob = prob_all.reshape((-1,10,2))
    pred = pred_all.reshape((-1,10,2))
    label = label_all.reshape((-1,10,2))
    # get side
    idx_side = np.sum(pred,axis=1)
    idx_side = np.argmin(idx_side,axis=-1)
    pred_side = np.zeros(pred.shape[:2])
    prob_side = np.zeros(prob.shape[:2])
    label_side = np.zeros(pred.shape[:2])
    for i in range(pred.shape[0]):
        pred_side[i,:] = pred[i,:,idx_side[i]]
        label_side[i,:] = label[i,:,idx_side[i]]
        prob_side[i,:] = prob[i,:,idx_side[i]]
    label_ASPECTS_side = np.sum(label_side,axis=1)
    pred_ASPECTS_side = np.sum(pred_side,axis=1)
    return label_side, pred_side, prob_side, label_ASPECTS_side, pred_ASPECTS_side

def get_single_region_performance(label_all,pred_all,prob_all,both_size=True):
    

    CRs = [classification_report_single_region(label_all,pred_all,prob_all,region=i) for i in range(len(ROIs))]
    #CR_single = summarize_single_region_performance(CRs)
    return CRs


def get_regional_performance(label_all,pred_all,prob_all):
    
    label_side, pred_side, prob_side, label_ASPECTS_side, pred_ASPECTS_side = get_known_side(label_all,pred_all,prob_all)

    CR = classification_report_regional(label_all,pred_all,prob_all)
    CR_side = classification_report_regional(label_side, pred_side, prob_side)
    
    CR_dict = {
        'Regional': CR,
        'KnownSideRegional': CR_side,
    }
    return CR_dict


def summarize_performance(CR):
    CR =  pd.DataFrame(CR)
    return CR
def summarize_fold_performance(CRs):
    CR_dict = {}
    for key in CRs[0].keys():
        CR_dict[key] = pd.DataFrame([CR[key] for CR in CRs]) 
    CRs = pd.concat(CR_dict,axis=1)
    index_labels=['%d'%i for i in range(len(CRs))] + ['mean', 'std']
    CRs = CRs.append(CRs.mean(),ignore_index=True).append(CRs.std(),ignore_index=True)
    CRs.index = index_labels
    return CRs

def summarize_single_region_performance(CRs):
    CR_dict = {}
    for key in CRs[0].keys():
        CR_dict[key] = pd.DataFrame([CR[key] for CR in CRs]) 
    CRs = pd.concat(CR_dict,axis=1)
    index_labels= ROIs + ['mean', 'std']
    CRs = CRs.append(CRs.mean(),ignore_index=True).append(CRs.std(),ignore_index=True)
    CRs.index = index_labels
    return CRs



def get_ASPECT_performance(label_all,pred_all,prob_all,ASPECTS_threshold=6):
    
    label_LR = np.sum(np.reshape(label_all,(-1,10,2)),axis=1)
    prob_LR = np.reshape(prob_all,(-1,10,2))
    pred_LR = np.reshape(pred_all,(-1,10,2))
    pred_ASPECT_LR =  np.sum(pred_LR,axis=1)
    label_side, pred_side, prob_side,label_ASPECTS_side, pred_ASPECTS_side = get_known_side(label_all,pred_all,prob_all)

    CR_LR = classification_report_ASPECTS(label_LR,pred_ASPECT_LR,ASPECTS_threshold=ASPECTS_threshold)
    CR_R = classification_report_ASPECTS(label_LR[:,0],pred_ASPECT_LR[:,0],ASPECTS_threshold=ASPECTS_threshold)
    CR_L = classification_report_ASPECTS(label_LR[:,1],pred_ASPECT_LR[:,1],ASPECTS_threshold=ASPECTS_threshold)
    CR_side = classification_report_ASPECTS(label_ASPECTS_side,pred_ASPECTS_side,ASPECTS_threshold=ASPECTS_threshold)
    
    CR_dict = {
        'ASPECTS_th%d' %ASPECTS_threshold: CR_LR,
        'LeftASPECTS_th%d' %ASPECTS_threshold: CR_L,
        'RightASPECTS_th%d' %ASPECTS_threshold: CR_R,
        'KnownSideASPECTS_th%d' %ASPECTS_threshold: CR_side,
    }
    return CR_dict

def classification_report_with_ASPECTS(label_all,pred_all,prob_all,ASPECTS_threshold=6):
    exclude_keys = ['accuracy', 'micro avg']
    CRs_regional = get_regional_performance(label_all,pred_all,prob_all)
    CRs_ASPECTS = get_ASPECT_performance(label_all,pred_all,prob_all,ASPECTS_threshold=ASPECTS_threshold)
    CRs = {**CRs_regional, **CRs_ASPECTS}

    for CRkey in CRs.keys():
        CRs[CRkey] = exclude_fields(CRs[CRkey] ,exclude_keys = exclude_keys)
    return  CRs
    