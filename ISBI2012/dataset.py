import os
import numpy as np
import torch
import random
import elasticdeform
from tifffile import *
import albumentations as A

class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, transform=None, pad_size = 48):
        self.transform = transform
        self.mode = mode
        if self.mode in ["train","val",'val_in_train','train+val']:
            self.img = imread("data/train-volume.tif")
            self.label = imread("data/train_labels.tif")
        elif self.mode == "test":
            self.img = imread("data/test-volume.tif")
        self.pad_size = pad_size
    def __len__(self):
        if self.mode in ['train','train+val']:
            return 16
        if self.mode in ['val','val_in_train']:
            return 1*4*4
        if self.mode == 'test':
            return 6*4*4
    
    def __getitem__(self, index):

        if self.mode == "train":
            img = self.img[:25,:,:]
            label = self.label[:25,:,:]
            ret = {
            'img': img, # (25,512,512)
            'label': label, # (25,512,512)
            }
            ret = self.transform(ret)

        elif self.mode == "train+val":
            img = self.img
            label = self.label
            ret = {
            'img': img, # (30,512,512)
            'label': label, # (30,512,512)
            }
            ret = self.transform(ret)

        elif self.mode == "val_in_train":
            img = self.img[:5,:,:]
            label = self.label[:5,:,:]
            
            current_address = os.path.dirname(os.path.abspath(__file__))

            imwrite(current_address+'/result/val_in_train_label.tif',label.astype(np.float32)/255.)

            ret = {
            'img': img, # (5,512,512)
            'label': label, # (5,512,512)
            }
            ret = self.transform(ret) # (5,512+16*2, 512+16*2)
            img, label = ret["img"], ret['label']

            
            i = index // 16
            j = index % 16 // 4
            k = index % 16 % 4
            img = ret["img"][:,i*5:i*5+5,j*128:j*128+160,k*128:k*128+160]
            label = ret["label"][:,i*5:i*5+5,j*128:j*128+160,k*128:k*128+160]
            ret = {
            "img": img, "label": label
            }            
            
            
        elif self.mode == "val":
            img = self.img[25:,:,:]
            label = self.label[25:,:,:]
            
            current_address = os.path.dirname(os.path.abspath(__file__))

            imwrite(current_address+'/result/val_label.tif',label.astype(np.float32)/255.)

            ret = {
            'img': img, # (5,512,512)
            'label': label, # (5,512,512)
            }
            ret = self.transform(ret) # (5,512+16*2, 512+16*2)
            img, label = ret["img"], ret['label']

            
            i = index // 16
            j = index % 16 // 4
            k = index % 16 % 4
            img = ret["img"][:,i*5:i*5+5,j*128:j*128+128+self.pad_size*2,k*128:k*128+128+self.pad_size*2]
            label = ret["label"][:,i*5:i*5+5,j*128:j*128+128+self.pad_size*2,k*128:k*128+128+self.pad_size*2]
            ret = {
            "img": img, "label": label
            }

        elif self.mode == "test":
            img = self.img
            ret = {
            'img': img, # (512,512,30)
            }
            ret = self.transform(ret)
            i = index // 16
            j = index % 16 // 4
            k = index % 16 % 4
            ret = {
                "img": ret["img"][:, i*5:i*5+5,j*128:j*128+128+self.pad_size*2,k*128:k*128+128+self.pad_size*2]
            }
        
        return ret # (5,160,160)

class ReflectPadding:
    def __init__(self,pad_size=16):
        self.pad_size = pad_size
    def __call__(self, data):
        if "label" in data.keys():
            img, label = data['img'], data['label']
        
            img = np.pad(img,((0,0),(self.pad_size,self.pad_size),(self.pad_size,self.pad_size)),mode="reflect")
            label = np.pad(label,((0,0),(self.pad_size,self.pad_size),(self.pad_size,self.pad_size)),mode="reflect")
            
            ret = {
                'img': img,
                'label': label,
            }
        else:
            img = data['img']
            img = np.pad(img,((0,0),(self.pad_size,self.pad_size),(self.pad_size,self.pad_size)),mode="reflect")
            ret = {
                'img': img,
            }
        return ret

class RandomElasticDeform:
    def __init__(self,channel_crop_size=5,spatial_crop_size=160,sigma=25,points=3):
        self.channel_crop_size = channel_crop_size
        self.spatial_crop_size = spatial_crop_size
        self.sigma = sigma
        self.points = points
        
    def __call__(self, data):
#         channel crop
        img, label = data['img'], data['label']
        channel, img_size, _ = img.shape
        first_channel = random.randint(0,channel - self.channel_crop_size)
        img = img[first_channel:first_channel + self.channel_crop_size, :, :]
        label = label[first_channel:first_channel + self.channel_crop_size, :, :]     
        left_crop = random.randint(0,img_size-self.spatial_crop_size)
        up_crop = random.randint(0,img_size-self.spatial_crop_size)
        crop = (slice(left_crop,left_crop+self.spatial_crop_size),
                slice(up_crop,up_crop+self.spatial_crop_size))   
        img_deformed, label_deformed = elasticdeform.deform_random_grid([img,label],axis=[(1,2),(1,2)],
                                                                       sigma=25,points=self.points,
                                                                       order=[3,0],crop=crop)
        ret = {
            'img': img_deformed,
            'label': label_deformed,
        }
        return ret

class RandomFlip:
    def __call__(self, data):
        img, label = data['img'], data['label']
        
        if np.random.rand() > 0.5:
            img = img[::-1,:,:]
            label = label[::-1,:,:]
            
        if np.random.rand() > 0.5:
            img = img[:,::-1,:]
            label = label[:,::-1,:]

        if np.random.rand() > 0.5:
            img = img[:,:,::-1]
            label = label[:,:,::-1]
            
        ret = {
            'img': img,
            'label': label,
        }
        return ret

    
class ToTensor:
    def __call__(self, data):
        if "label" in data.keys():
            img, label = data['img'], data['label']

            img = img[np.newaxis,...].astype(np.float32)/255.
            label = label[np.newaxis,...].astype(np.float32)/255.

            ret = {
                'img': torch.from_numpy(img),
                'label': torch.from_numpy(label),
            }
        else:
            img = data['img']
            img = img[np.newaxis,...].astype(np.float32)/255.
            ret = {
                'img': torch.from_numpy(img),
            }
            
        return ret


class GaussianNoise:
    def __init__(self,std):
        self.std = std
    def __call__(self, data):
        img, label = data['img'], data['label']
        
        img += torch.randn(img.shape) * self.std
            
        ret = {
            'img': img,
            'label': label,
        }
        return ret


class RandomCrop(object):
    def __init__(self, channel_crop_size=5, spatial_crop_size=160):
        self.channel_crop_size = channel_crop_size
        self.spatial_crop_size = spatial_crop_size

    def __call__(self, data):
        img, label = data['img'], data['label']
        channel, img_size, _ = img.shape
        first_channel = random.randint(0,channel - self.channel_crop_size)
        img = img[first_channel:first_channel + self.channel_crop_size, :, :]
        label = label[first_channel:first_channel + self.channel_crop_size, :, :]

        # === Not using albumentaions library ===
        print('in call')
        print(img_size,self.spatial_crop_size)
        left_crop = random.randint(0, img_size-self.spatial_crop_size)
        up_crop = random.randint(0, img_size-self.spatial_crop_size)
        crop = (slice(left_crop, left_crop+self.spatial_crop_size),
                slice(up_crop, up_crop+self.spatial_crop_size))
        img = img[:, crop[0], crop[1]]
        label = label[:, crop[0], crop[1]]
        ret = {
            'img': img,
            'label': label,
        }

        return ret

class Rotate(object):
    def __init__(self, limit=180):
        self.limit = limit

    def __call__(self, data):
        img, label = data['img'], data['label']
        channel, img_size, _ = img.shape
        img = np.transpose(img, (1, 2, 0))
        label = np.transpose(label, (1, 2, 0))  # CHW --> HWC
        rotate = A.Rotate(limit=self.limit, interpolation=1, border_mode=4, p=0.5)
        transformed = rotate(image=img, mask=label)
        ret = {
            'img': np.transpose(transformed['image'], (2, 0, 1)),
            'label': np.transpose(transformed['mask'], (2, 0, 1)),  # .long()
        }
        return ret
