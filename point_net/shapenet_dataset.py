'''
    Dataset for shapenet provides functionality for both Classification and Segmentation

can be downloaded in Colab using the following lines
# !wget -nv https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
# !unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
# !rm shapenetcore_partanno_segmentation_benchmark_v0.zip

This dataloader is based on:
    https://github.com/intel-isl/Open3D-PointNet

'''

import os
import json
import numpy as np
import open3d as o3
from PIL import Image
import torch
from torch.utils.data import Dataset

class ShapenetDataset(Dataset):

    def __init__(self, root, split, npoints=2500, classification=False, class_choice=None, 
                 image=False, normalize=True):

        self.root = root
        self.split = split.lower()
        self.npoints = npoints
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = classification
        self.image = image
        self.normalize = normalize

        # Open the Category File and Map Folders to Categories
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        
        # select specific categories from the dataset. 
        # ex: Call in parameters "class_choice=["Airplane"].
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        
        # for each category, assign the point, segmentation, and image.
        self.meta = {}        
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_seg_img = os.path.join(self.root, self.cat[item], 'seg_img')
            
            # get train, valid, test splits from json files
            if self.split == 'train':
                split_file = os.path.join(self.root, 
                    r'train_test_split\shuffled_train_file_list.json')
            elif self.split == 'test':
                split_file = os.path.join(self.root, 
                    r'train_test_split\shuffled_test_file_list.json')
            elif (self.split == 'valid') or (self.split == 'val'):
                split_file = os.path.join(self.root, 
                    r'train_test_split\shuffled_val_file_list.json')
                
            with open(split_file, 'r') as f:
                split_data = json.load(f)

            # get point cloud file (.pts) names for current split
            pts_names = []
            for token in split_data:
                if self.cat[item] in token:
                    pts_names.append(token.split('/')[-1] + '.pts')
                

            # FOR EVERY POINT CLOUD FILE
            for fn in pts_names:
                token = (os.path.splitext(os.path.basename(fn))[0])
                # add point cloud, segmentations, and image to class metadata dict
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), 
                                        os.path.join(dir_seg, token + '.seg'),
                                        os.path.join(dir_seg_img, token + '.png')))
       
        # create list containing (item, points, segmentation points, segmentation image)
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        
        self.num_seg_classes = 0
        if not self.classification: # Take the Segmentation Labels
            for i in range(len(self.datapath)//50):
                # get number of seg classes in current item
                l = len(np.unique(np.loadtxt(self.datapath[i][2]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)


    def __getitem__(self, index):
        '''
        Each element has the format "class, points, segmentation labels, segmentation image"
        '''
        # Get one Element
        fn = self.datapath[index]
        
        # get its Class
        cls_ = self.classes[fn[0]]
        
        # Read the Point Cloud
        point_set = np.asarray(o3.io.read_point_cloud(fn[1], format='xyz').points,dtype=np.float32)
        
        # Read the Segmentation Data
        seg = np.loadtxt(fn[2]).astype(np.int64)
        
        # Read the Segmentation Image
        image = Image.open(fn[3])
        
        # down sample the pont cloud
        if len(seg) > self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            # case when there are less points than the desired number
            choice = np.random.choice(len(seg), self.npoints, replace=True)

        point_set = point_set[choice, :]        
        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls_ = torch.from_numpy(np.array([cls_]).astype(np.int64))
        
        # add Gaussian noise to point set if not testing
        if self.split != 'test':
            # add N(0, 1/100) noise
            point_set += torch.randn(point_set.shape)/100

            # add random rotation to the point cloud
            point_set = self.random_rotate(point_set)

        # Normalize Point Cloud to (0, 1)
        if self.normalize:
            point_set = self.normalize_points(point_set)
        
        if self.classification:
            if self.image:
                return point_set, cls_, image
            else:
                return point_set, cls_

        else:
            if self.image:
                return point_set, seg, image
            else:
                return point_set, seg


    @staticmethod
    def random_rotate(points):
        ''' randomly rotates point cloud about vertical axis.
            Code is commented out to rotate about all axes
            '''
        # construct a randomly parameterized 3x3 rotation matrix
        # phi = torch.FloatTensor(1).uniform_(-torch.pi, torch.pi)
        theta = torch.FloatTensor(1).uniform_(-torch.pi, torch.pi)
        # psi = torch.FloatTensor(1).uniform_(-torch.pi, torch.pi)

        # rot_x = torch.Tensor([
        #     [1,              0,                 0],
        #     [0, torch.cos(phi), -torch.sin(phi)],
        #     [0, torch.sin(phi), torch.cos(phi) ]])

        rot_y = torch.Tensor([
            [torch.cos(theta),  0, torch.sin(theta)],
            [0,                 1,                0],
            [-torch.sin(theta), 0, torch.cos(theta)]])

        # rot_z = torch.Tensor([
        #     [torch.cos(psi), -torch.sin(psi), 0],
        #     [torch.sin(psi), torch.cos(psi),  0],
        #     [0,              0,                 1]])

        # rot = torch.matmul(rot_x, torch.matmul(rot_y, rot_z))
        
        return torch.matmul(points, rot_y)

    @staticmethod
    def normalize_points(points):
        ''' Perform min/max normalization on points
            Same as:
            (x - min(x))/(max(x) - min(x))
            '''
        points = points - points.min(axis=0)[0]
        points /= points.max(axis=0)[0]

        return points


    def __len__(self):
        return len(self.datapath)

