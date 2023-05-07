
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from os.path import isfile, isdir, join, splitext
from glob import glob
from torch.utils.data import Dataset
import time
import numpy as np
import scipy.io



def zero_padding(img,size=(512,512,20)):
    def compute(a , max):
        if ( (max - a) % 2 )== 0 :
            front =int( (max - a) / 2 )
            end =int( (max - a) / 2 ) 
        else :
            front = math.floor((max - a) / 2 )
            end =  math.ceil((max - a) / 2 )

        return front ,end

    d1_front , d1_end = compute(img.shape[-3] , size[0])
    d2_front , d2_end = compute(img.shape[-2] , size[1])
    d3_front , d3_end = compute(img.shape[-1] , size[2]) 
    #img = np.pad(img , ((d1_front,d1_end),(d2_front,d2_end),(d3_front,d3_end)) , 'constant' , constant_values=(0,0) )
    img = F.pad(img , (d3_front,d3_end, d2_front,d2_end, d1_front,d1_end) , mode='constant'  )
    return img


def affine_resample(img, M):
    img = img.unsqueeze(0)
    M = M.unsqueeze(0)
    
    grid = F.affine_grid(M[:,range(3),:], img.shape,align_corners=False)
    
    img_w = F.grid_sample(img, grid,padding_mode='zeros',align_corners=False)
    img_w = img_w.view(img_w.shape[1:])
    return img_w


class TorchDataset(Dataset):
    def __init__(self, image_dir,
        repeat = 1, augment=True,pre_align=False,
        hflip=False, vflip=False,
        n_augs=1,crop_ratio = 1,pad_size=(256,256,25),
        aug_Rrange=15,
        aug_Trange=0.02,
        skull_strip=False,
        z_crop=False,dtype=torch.float32):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.image_dir = image_dir
        self.pre_align = pre_align
        self.image_list = self.read_file(image_dir)
        self.label_list = [self.load_label(x,image_dir) for x in  self.image_list]
        self.len = len(self.image_list)
        self.repeat = repeat
        self.crop_ratio = crop_ratio
        self.n_augs = n_augs
        self.z_crop = z_crop
        self.dtype = dtype
        self.skull_strip = skull_strip
        #self.toTensor = crop_ratio.ToTensor()
        ''''''
        data_augmentation = []
        data_preprocess = transforms.Lambda(lambda img: zero_padding(img,pad_size))
        if augment:
            if hflip:
                data_augmentation.append(transforms.RandomVerticalFlip(p=0.5))  #we are [Z x W x H], hflip are [... x H x W]
            if vflip:
                data_augmentation.append(transforms.RandomHorizontalFlip(p=0.5))
            data_augmentation.append(transforms.RandomAffine(aug_Rrange,translate=(aug_Trange,aug_Trange),scale=(0.9,1.1)))
        self.preprocess = data_preprocess
        self.transformations=transforms.Compose(data_augmentation)        
        '''
        self.transformations = CustomAugmentation(
            flip_p=0.5, rotate=5,
            translate=(0.05,0.05),scale=(0.95,1.05),
            crop_ratio = crop_ratio,pad_size=pad_size )
        '''
        
    def getitem_base(self, i):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        start = time.time()
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_mat = self.image_list[index]

        img,label,transform = self.load_data(image_mat , self.image_dir)

        img = self.data_preproccess(img,transform)
        #img = torch.unsqueeze(img,0)

        #label = self.load_label(image_mat , self.image_dir)

        end = time.time()
        etime = end - start        
        return img, label

    
    def __getitem__(self, idx):
        if self.n_augs == 1:
            return self.getitem_base(idx)
        else:
            data_list = [self.getitem_base(idx) for i in range(self.n_augs) ]
            
            return [x[0] for x in data_list], [x[1] for x in data_list]

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        if self.repeat == None:
            data_len = 10000000000
        else:
            data_len = len(self.image_list) * self.repeat
        return data_len

    def read_file(self , image_dir):
        file_list = sorted(glob(join(image_dir,'*.mat')))
        return file_list
    def load_mat(self, image_mat , image_dir):
        path = join(image_dir , image_mat)
        mat = scipy.io.loadmat(path)
    def load_label(self , image_mat , image_dir):
        path = join(image_dir , image_mat)
        mat = scipy.io.loadmat(path)
        label = np.reshape(mat.get('label'),(10,2))
        return label
        
    def load_data(self , image_mat , image_dir):
        path = join(image_dir , image_mat)
        mat = scipy.io.loadmat(path)
        data = mat.get('data')
        mask = mat.get('mask')
        brain = mat.get('brain')        
        label = np.reshape(mat.get('label'),(10,2))
        if self.skull_strip:
            data = data * brain
        if 'transform' in mat.keys():
            transform = mat.get('transform')    
        else:    
            transform = None
        '''
        if label == 2:
            label = 0
        '''
        mask = np.transpose(mask,(3,0,1,2))
        data = np.expand_dims(data,axis=0)
        brain = np.expand_dims(brain,axis=0)
        data = np.concatenate([data,brain,mask],axis=0)
        if(self.z_crop):
            z_mask = np.max(mask,axis=0)
            z_mask = np.max(z_mask,axis=0)
            z_mask = np.max(z_mask,axis=0)
            data_masked = data[...,z_mask==1]

        return data, label, transform
     


    def filter(self,idx):

        self.image_list = [self.image_list[i] for i in idx]
        self.label_list = [self.label_list[i] for i in idx]
        self.len = len(self.image_list)

    def load_img(self , image_mat , image_dir):
        path = join(image_dir , image_mat)
        mat = scipy.io.loadmat(path)
        data = mat.get('data')
        mask = mat.get('mask')
        brain = mat.get('brain')

        return np.stack([data,mask,brain],axis=0)
        
    def calculate_slice(self):
        label = np.concatenate(self.label_list)
        label = label.flatten()
        normal_data_num = len( np.argwhere(label == 1))
        detect_data_num = len( np.argwhere(label == 0))
        return normal_data_num , detect_data_num

    def data_preproccess(self , data,transform=None):
        #data = self.toTensor(data)
        data = torch.tensor(data,dtype=self.dtype).cuda()
        data = self.preprocess(data)
        if self.pre_align:
            transform = torch.tensor(transform,dtype=self.dtype).cuda()
            data = affine_resample(data, transform)
        data = data.permute(0,3,1,2)
        data = self.transformations(data)
        data = data.permute(0,2,3,1)
        return data
    
    def __str__(self):
        normal_data_num, detect_data_num = self.calculate_slice()
        data_num = normal_data_num + detect_data_num
        affect_ratio = float(detect_data_num) / float(data_num)
        repr_str = "TorchDataset\n\t%d volumes,\t%d ROIs\n\t%d normal ROIs,\t%d defect ROIs (%.2f%% affected)" %(len(self),data_num,normal_data_num,detect_data_num,affect_ratio*100)
        return repr_str
        
    def no_augment(self):
        self.transformations=transforms.Compose([])   

class TorchDatasetWithDWI(TorchDataset):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args,**kwargs)
          
    def getitem_base(self, i):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        start = time.time()
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_mat = self.image_list[index]

        img,label,transform = self.load_data(image_mat , self.image_dir)

        img = self.data_preproccess(img,transform)
        #img = torch.unsqueeze(img,0)

        #label = self.load_label(image_mat , self.image_dir)

        end = time.time()
        etime = end - start        
        return img, label

    def load_data(self , image_mat , image_dir):
        path = join(image_dir , image_mat)
        mat = scipy.io.loadmat(path)
        data = mat.get('data')
        mask = mat.get('mask')
        brain = mat.get('brain')        
        DWI = mat.get('DWI')        
        label = np.reshape(mat.get('label'),(10,2))
        if self.skull_strip:
            data = data * brain
        if 'transform' in mat.keys():
            transform = mat.get('transform')    
        else:    
            transform = None
        '''
        if label == 2:
            label = 0
        '''
        mask = np.transpose(mask,(3,0,1,2))
        data = np.expand_dims(data,axis=0)
        brain = np.expand_dims(brain,axis=0)
        DWI = np.expand_dims(DWI,axis=0)
        data = np.concatenate([data,brain,mask,DWI],axis=0)
        if(self.z_crop):
            z_mask = np.max(mask,axis=0)
            z_mask = np.max(z_mask,axis=0)
            z_mask = np.max(z_mask,axis=0)
            data_masked = data[...,z_mask==1]

        return data, label, transform
     

class HBMDataset(TorchDataset):        
    def crop(self,x,mx,my,mz=None,roi=0):
        mx = mx[roi,:]
        my = my[roi,:]
        xi = x[:,mx,:,:]
        xi = xi[:,:,my,:].unsqueeze(0)
        #xi = xi[:,:,:,mz]
        x = self.pool(xi)
        return x
    
    def get_bounding_box(self,x,mask):
        
        mx = mask.max(dim=-1)[0].max(dim=-1)[0].type(torch.bool)
        my = mask.max(dim=-1)[0].max(dim=-2)[0].type(torch.bool)
        mz = mask.max(dim=-2)[0].max(dim=-2)[0].type(torch.bool)
        n_slices = torch.sum(mz,dim=-1)
        out_list = []
        for i in range(mask.shape[0]):
            xi = x[i,...] # 1 * C * X * Y * Z
            xx = [ self.crop(xi,mx[i,...],my[i,...],roi=i_roi)  for i_roi in range(mask.shape[1])]
            xx = torch.stack(xx,dim=-2)
            mbox = [ self.crop(mask[i,i_roi,...].unsqueeze(0),mx[i,...],my[i,...],roi=i_roi)  for i_roi in range(mask.shape[1])]
            mbox = torch.stack(mbox,dim=-2)
            xx *= mbox
            
            out_list.append(xx)
            
    def get_ROI(self , data):
        out = torch.cat(out_list,dim=0)
        #data = self.toTensor(data)
        data = torch.tensor(data,dtype=self.dtype).cuda()
        data = self.preprocess(data)
        if self.pre_align:
            transform = torch.tensor(transform,dtype=self.dtype).cuda()
            data = affine_resample(data, transform)
        data = data.permute(0,3,1,2)
        data = self.transformations(data)
        data = data.permute(0,2,3,1)
        return data
    