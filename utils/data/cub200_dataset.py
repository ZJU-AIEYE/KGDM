import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import imageio

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg


def pil_loader(imgname, crop=None):
    # maybe faster
    if crop is None:
        with open(imgname, 'rb') as f:
            return Image.open(imgname).convert('RGB')
    else:
        box = (crop[0].item(), crop[1].item(), crop[2].item()+crop[0].item(), crop[3].item()+crop[1].item())
        with open(imgname, 'rb') as f:
            img = Image.open(imgname).convert('RGB')
        im = img.crop(box)

        return im


def single_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor()]
    )


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224,scale=(0.8,1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])


def test_transform():
    return transforms.Compose(
        [
            transforms.Resize((224,224)),

            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])


def default_transform(a):
    return ToTensor()(a)


def collate_fn_CUB(mode='single'):
    if mode == 'single':
        def f(datas):
            return {
                'name': [d['name'] for d in datas],
                'box': [d['box'] for d in datas],
                'gt': torch.from_numpy(np.array([d['gt'] for d in datas])).long(),
                'img':torch.stack([d['img1'] for d in datas],0)
            }
        return f


# ： from https://github.com/TACJu/TransFG
class CUB():
    def __init__(self, root, is_train=True, data_len=None, transform=default_transform):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        bounding_box_file = open(os.path.join(self.root, 'bounding_boxes.txt'))

        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])

        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)

        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        bounding_box_list = []
        for line in bounding_box_file:
            data = line[:-1].split(' ')
            bounding_box_list.append([int(float(data[1])), int(float(data[2])),
                                      int(float(data[3])), int(float(data[4]))])

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.train_box = torch.tensor([x for i, x in zip(train_test_list, bounding_box_list) if i])
        self.test_box = torch.tensor([x for i, x in zip(train_test_list, bounding_box_list) if not i])

        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]

        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]

    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname, box = self.train_img[index], self.train_label[index], self.train_imgname[index], \
                                        self.train_box[index]

            img1 = self.transform(pil_loader(img))
            img2 = self.transform(pil_loader(img, crop=box))

        else:
            img, target, imgname, box = self.test_img[index], self.test_label[index], self.test_imgname[index], \
                                        self.test_box[index]

            img1 = self.transform(pil_loader(img))
            img2 = self.transform(pil_loader(img, crop=box))

        return {
            'name': imgname,
            'box': box,
            'img1': img1,
            'gt': target,
            'img2': img2
        }

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)
