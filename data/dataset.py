import os
from torch.utils.data import DataLoader
import torch as t
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision as tv
class DogCat(data.Dataset):
    def __init__(self,path,transform=None,train=True,test=False):
        super(DogCat,self).__init__()
        """获取训练数据"""
        self.test = test
        imgs=[os.path.join(path,img) for img in os.listdir(path)]
        if self.test:

            imgs=sorted(imgs,key=lambda x: (int(x.split('.')[-2].split('\\')[-1])))
        else :
            imgs=sorted(imgs,key=lambda x: int(x.split('.')[-2]))

        imgs_num=len(imgs)
        #训练集，测试集验证集划分验证：训练=3:7
        if self.test:
            self.imgs=imgs
        elif train:
            self.imgs=imgs[:int(0.7*imgs_num)]
        else :
            self.imgs=imgs[int(0.7*imgs_num):]
        if transform is None:
            normalize=tv.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            if self.test or not train:
                self.transforms=tv.transforms.Compose([
                    tv.transforms.Scale(224),
                    tv.transforms.CenterCrop(224),
                    tv.transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = tv.transforms.Compose([
                    tv.transforms.Scale(256),
                    tv.transforms.RandomResizedCrop(224),
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        # 返回一张图片的数据
        img_path = self.imgs[index]
        print('img_path:\n',img_path)
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('\\')[-1])
            print("label:", label)
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
            print("label:",label)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        # 返回数据集中的所有图片张数
        return len(self.imgs)
datapath='D:/workspace/code/pytorch/kaggle/train/cat/'
batch_size=8
num_workers=0
train_dataset=DogCat(datapath,test=True)
train_dataloader=DataLoader(train_dataset,batch_size,shuffle=True,num_workers=num_workers)
for ii,(data,label) in enumerate(train_dataloader):
    print('\n')





