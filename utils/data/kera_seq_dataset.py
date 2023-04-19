
import torch
from torch.utils.data import Dataset, dataloader,Subset
import glob
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
from itertools import combinations
from random import randint,shuffle
import pandas as pd


def pil_loader(imgname):
    # maybe faster
    with open(imgname,'rb') as f:
        return Image.open(imgname).convert('RGB')

def default_transform(a):
    return ToTensor()(a)

colorjit = [transforms.RandomRotation(10), transforms.ColorJitter(0.02,0.02,0.01)]


def single_transform(imgsize=256):
    actualsize = imgsize*8//7
    return transforms.Compose(
        [
        transforms.Resize(actualsize),
        transforms.RandomApply(colorjit),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((imgsize,imgsize)),
        transforms.ToTensor()]
    )

def test_transform(imgsize=256):
    actualsize = imgsize*8//7
    return transforms.Compose(
        [
        transforms.Resize(actualsize),
        transforms.CenterCrop((imgsize,imgsize)),
        transforms.CenterCrop(imgsize),
        transforms.ToTensor()]
    )

def collate_fn(mode='single', normal_set=None):
    if mode=='single':
        def f(datas):
            return {
                'path':[d['path'] for d in datas],
                'gt': torch.tensor([d['gt'] for d in datas]),
                'img':torch.stack([d['img'] for d in datas],0)
            }
        return f
    elif mode=='double':
        def f(datas):
            return {
                'p1':[d['p1'] for d in datas],
                'p2':[d['p2'] for d in datas],
                'gt': torch.tensor([d['gt'] for d in datas]),
                'img1':torch.stack([d['img1'] for d in datas]),
                'img2':torch.stack([d['img2'] for d in datas])
            }
        return f
    elif mode=='with_normal':
        N = len(normal_set)
        def f(datas):
            batchsize=len(datas)
            n_normal = batchsize//5
            select=list(range(N))
            shuffle(select)
            select = select[:n_normal]
            return {
                'path':[d['path'] for d in datas]+[normal_set[i]['path'] for i in select],
                'gt': torch.tensor([d['gt'] for d in datas]+[-1 for i in select]).long(),
                'img':torch.stack([d['img'] for d in datas]+[normal_set[i]['img'] for i in select],0)
            }
        return f

class Keratitis_sequence(Dataset):
    # single mode return all imgs in (path,gt) tuple
    # double mode return items depends on training status:
    #   -- training: return all possible pairs in (path, path, gt) tuple
    #   -- testing: only return the first and the second path of each patient in (path, path, format)
    def __init__(self, path, transform=default_transform, mode='single', is_train=False, cross_validation_fold=-1) -> None:
        super().__init__()
        self.categories=['Amoeba','Bact', 'Fungal','Hsk','GCD','GDCD','HealthyCornea','LCD','MCD','TMD']
        self.mode=mode
        self.transform = transform
        self.filenames = glob.glob(f"{path}/*/*/*.jpg")
        self.patients={}
        self.imgs = []
        self.is_train = is_train
        self.cross_validation = None
        self.cross_validation_fold = cross_validation_fold
        for i,f in enumerate(self.filenames):
            gt = self.categories.index(f.split(os.sep)[-3])
            pid = f.split('/')[-2]
            self.imgs.append((f,gt))
            if pid in self.patients.keys():
                self.patients[pid].append(i)
            else:
                self.patients[pid]=[i]
        self.sample_pair = []
        for k in self.patients:
            if is_train:
                self.sample_pair.extend(list(combinations(self.patients[k],2)))
            else:
                _ = [self.sample_pair.append(tuple(self.patients[k][i:i+2])) for i in range(0,len(self.patients[k]),2) if i<len(self.patients[k])-1]
        
    def __getitem__(self, index):
        if self.mode=='single':
            path,gt = self.imgs[index]
            return {
                'path':path,
                'gt':gt,
                'img':self.transform(pil_loader(path))
            }
        else:
            (p1,gt1),(p2,gt2)= self.sample_pair[index]
            return {
                'p1':p1,
                'p2':p2,
                'gt':gt1,
                'img1':self.transform(pil_loader(p1)),
                'img2':self.transform(pil_loader(p2))
            }

    def cross_validation_split(self, fold_id):
        idx = list(range(self.__len__()))
        pids = list(self.patients.keys())
        if not self.cross_validation:
            n =len(pids)//self.cross_validation_fold
            shuffle(pids)
            self.pids_split = [pids[i:i+n] for i in range(self.cross_validation_fold)]
            self.cross_validation  = [ [i for p in ps for i in self.patients[p]] for ps in self.pids_split]
        return Subset(self,list(set(idx) - set(self.cross_validation[fold_id]))),Subset(self,self.cross_validation[fold_id])

    def __len__(self):
        return len(self.imgs) if self.mode=='single' else len(self.sample_pair)

    def get_distribution(self):      
          return {k:len(list(filter(lambda x: x[0][1]==i, self.sample_pair))) for i,k in enumerate(self.categories)}

    def get_random_sample(self, unsqueeze=False):
        if self.mode=='single':
            index = randint(0,len(self.imgs))
            path,gt = self.imgs[index]
            return {
                'path':path,
                'gt':gt,
                'img':self.transform(pil_loader(path))[None,:,:,:] if unsqueeze else self.transform(pil_loader(path))
            }
        else:
            index = randint(0,len(self.sample_pair))
            (p1,gt1),(p2,gt2)= self.sample_pair[index]
            return {
                'p1':p1,
                'p2':p2,
                'gt':gt1,
                'img1':self.transform(pil_loader(p1))[None,:,:,:] if unsqueeze else self.transform(pil_loader(p1)),
                'img2':self.transform(pil_loader(p2))[None,:,:,:] if unsqueeze else self.transform(pil_loader(p2)),
            }

class Keratitis_human_evaluation(Dataset):
    # single mode return all imgs in (path,gt) tuple
    # double mode return items depends on training status:
    #   -- training: return all possible pairs in (path, path, gt) tuple
    #   -- testing: only return the first and the second path of each patient in (path, path, format)
    CLS = ['ak','bk','fk','hsk','others','normal']
    def __init__(self, path, transform=default_transform) -> None:
        super().__init__()
        self.transform = transform
        self.imgs = glob.glob(f"{path}/*.jpg")
        csvpath = '/data/home/fangzhengqing/Data/human_evaluation_dataset_selected.csv'
        df = pd.read_csv(open(csvpath,"r"))
        gtdict = dict(zip(df.iloc[:,1],df.iloc[:,2]))
        # print(gtdict.keys())
        self.gts = [self.CLS.index(gtdict[k.split('/')[-1]]) for k in self.imgs]

    def __getitem__(self, index):
            path = self.imgs[index]
            return {
                'path':path,
                'gt':self.gts[index],
                'img':self.transform(pil_loader(path))
            }

    def __len__(self):
        return len(self.imgs)

