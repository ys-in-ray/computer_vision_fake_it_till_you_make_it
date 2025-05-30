import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 
import pickle5 
from glob import glob

from torchvision.transforms import transforms, functional
from PIL import Image
import json 
import random
import copy
import math

def get_dataset(train_data_root, valid_data_root, test_data_root, p_dict=None, img_size=384):
    # print(pickle.format_version)
    # get all the images and the corresponding 68 facial landmarks
    with open(train_data_root, 'rb') as f:
        train_annot = pickle5.load(f)

    with open(valid_data_root, 'rb') as f:
        valid_annot = pickle5.load(f)
    
    # train_annot = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    # print(train_annot[0][0])
    # print(train_annot[1][0])
    N = len(train_annot)
    

    # apply shuffle to generate random results 
    # train_annot = np.random.shuffle(np.array(train_annot))
    
    # all_images, all_landmarks = train_annot[0], train_annot[1]
    train_images = train_annot[0]
    valid_images = valid_annot[0]
    test_images = glob(f"{test_data_root}/*")
    train_landmarks = []
    valid_landmarks = []
    # print(train_annot[1][0])
    for landmark in train_annot[1]:
        temp = []
        for X, Y in landmark:
            temp.extend([X, Y])
        train_landmarks.append(temp)

    for landmark in valid_annot[1]:
        temp = []
        for X, Y in landmark:
            temp.extend([X, Y])
        valid_landmarks.append(temp)  
        
    # print(train_landmarks[0])
    # train_images = all_images[:x]
    # valid_images = all_images[x:] + valid_annot[:,0]
    # train_landmarks = all_landmarks[:x] 
    # valid_landmarks = all_landmarks[x:] + valid_annot[:,1]
    
    
    ## TO DO ## 
    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    
   
    ## TO DO ##
    # Complete class cifiar10_dataset
    
    train_set, valid_set, test_set = dataset(images=train_images, landmarks=train_landmarks, prefix = './data/synthetics_train', train=True, p_dict=p_dict, img_size=img_size), \
                        dataset(images=valid_images, landmarks=valid_landmarks, prefix = './data/aflw_val', train=False, img_size=img_size), \
                        dataset(images=test_images, landmarks=None, prefix = './data/aflw_test', train=False, img_size=img_size)


    return train_set, valid_set, test_set

def horizontalfip_vector(num_landmark=68,img_size=384):
    
    v = np.zeros((num_landmark*2))
    for index in np.arange(0,num_landmark*2-1,2):
        v[index] = img_size-1
    
    return v
    
def horizontalfip_permutation(vector,num_landmark=68):
    
    horizontal_landmark = np.zeros((num_landmark*2))
    for index in range(0,num_landmark):
        if index <= 16:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(16-index)*2:np.abs(16-index)*2+2]
        elif index <= 26:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(43-index)*2:np.abs(43-index)*2+2]
        elif index <= 30:
            horizontal_landmark[index*2:index*2+2] = vector[index*2:index*2+2]
        elif index <= 35:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(66-index)*2:np.abs(66-index)*2+2]
        elif index <= 39:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(81-index)*2:np.abs(81-index)*2+2]
        elif index <= 41:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(87-index)*2:np.abs(87-index)*2+2]
        elif index <= 45:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(81-index)*2:np.abs(81-index)*2+2]
        elif index <= 47:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(87-index)*2:np.abs(87-index)*2+2]    
        elif index <= 54:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(102-index)*2:np.abs(102-index)*2+2]
        elif index <= 59:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(114-index)*2:np.abs(114-index)*2+2]
        elif index <= 64:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(124-index)*2:np.abs(124-index)*2+2]
        else:
            horizontal_landmark[index*2:index*2+2] = vector[np.abs(132-index)*2:np.abs(132-index)*2+2]
                     
    return horizontal_landmark.astype(np.float32)
    
def horizontalfip(image,landmark,img_size=384):
    v = horizontalfip_vector(img_size=img_size)
    HorizontalFlip_trans = transforms.RandomHorizontalFlip(p=1)
    image = HorizontalFlip_trans(image)
    vector = np.abs(v - landmark).astype(np.float32)
    landmark = horizontalfip_permutation(vector)
                
    return image, landmark

## TO DO ##
# Define your own cifar_10 dataset
class dataset(Dataset):
    def __init__(self, images, landmarks=None , prefix = './data/synthetics_train', train=False, p_dict=None, img_size=384):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.landmarks = landmarks 
        
        
        # prefix of the files' names
        self.prefix = prefix
        self.train = train
        self.img_size = img_size
        self.p_dict = p_dict

        if p_dict is None:
            self.p_dict = {
                    'RandomGrayscale':0.2,
                    'random_scale_p':0.2,
                    'random_scale_scale':0.8,
                    'random_rotate_p':0.5,
                    'random_rotate_degree':30,
                    'random_blur_p': 0.2,
                    'random_blur_min': 128,
                    'random_horizontalfip_p':0.5,
                    'random_erasemargin_p':0.2,
                    'random_erasemargin_ratio':0.7,
                    'RandomErasing_p':0.15,
                    'RandomErasing_scale_min':0.02,
                    'RandomErasing_scale_max':0.1,
                }
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self): 
        return len(self.images)


    def random_horizontalfip(self, img, landmark, p=0.2,img_size=384):
        rd_seed = random.uniform(0.0, 1.0)
        if rd_seed>p:
            return img, landmark
        else:
            return horizontalfip(img,landmark,img_size=img_size)

    def random_erasemargin(self, img, p=0.2, ratio=0.7, img_size=384):
        '''
        p: probability of erase
        ratio: minimum size to retain
        '''
        rd_seed = random.uniform(0.0, 1.0)
        if rd_seed>p:
            return img
        else:
            h_retain = random.uniform(ratio, 1.0)
            h_offset = random.uniform(0, 1-h_retain)
            w_retain = random.uniform(ratio, 1.0)
            w_offset = random.uniform(0, 1-w_retain)

            h1 = round(img_size*h_offset)
            h2 = round(img_size*h_retain)
            h3 = img_size-h1-h2
            w1 = round(img_size*w_offset)
            w2 = round(img_size*w_retain)
            w3 = img_size-w1-w2


            functional.erase(img,i=0,j=0,h=h1,w=img_size,v=0, inplace=True) # 上方
            functional.erase(img,i=0,j=w1+w2,h=img_size,w=w3,v=0, inplace=True) # 右方
            functional.erase(img,i=h1+h2,j=0,h=h3,w=img_size,v=0, inplace=True) # 下方
            functional.erase(img,i=0,j=0,h=img_size,w=w1,v=0, inplace=True) # 左方

            return img

    def random_rotate(self, img, landmark, p=0.5, degree=[-30,30], img_size=384):
        half_img_size = int(img_size/2)
        def rotate(p, degrees, origin=(112,112)):
            angle = np.deg2rad(degrees)
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle),  np.cos(angle)]])
            o = np.atleast_2d(origin)
            p = np.atleast_2d(p)
            return (R @ (p.T-o.T) + o.T).T.flatten()


        rd_seed = random.uniform(0.0, 1.0)
        if rd_seed>p:
            return img, landmark
        else:
            rotate_angle = random.randint(degree[0],degree[1])
            img = functional.rotate(img, rotate_angle,center=[half_img_size,half_img_size])
            # 實作上不知道為什麼degree要*-1才會吻合
            landmark = rotate(np.squeeze(landmark).reshape((68,2)), -1*rotate_angle,origin=(half_img_size,half_img_size))
            # img.show()
            return img, landmark

    def random_scale(self, img, landmark, p, scale=0.85, img_size=384):
        rd_seed = random.uniform(0.0, 1.0)
        if rd_seed>p:
            return img, landmark
        else:
            width_scale = random.uniform(scale,1)     # randomly resize width with scale between 0.9 and 1.0
            height_scale = random.uniform(scale,1)    # randomly resize height with scale between 0.9 and 1.0
            new_width = int(img_size * width_scale)
            new_height = int(img_size * height_scale)

            landmark[::2] *= width_scale
            landmark[1::2] *= height_scale

            pad_right = random.randint(0,img_size - new_width)
            pad_left = img_size-pad_right-new_width
            landmark[::2] += pad_left


            pad_top = random.randint(0,img_size - new_height)
            pad_bottom = img_size-pad_top-new_height
            landmark[1::2] += pad_top

            Resize = transforms.Resize((new_height, new_width))
            Pad = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill = 0, padding_mode = "constant")
            img = Resize(img)
            img = Pad(img)
            return img, landmark

    
    def random_blur(self, img, p, min_size=128):
        rd_seed = random.uniform(0.0, 1.0)
        if rd_seed>p:
            return img
        else:
            downsize = random.randint(min_size,self.img_size)
            img = transforms.Resize((downsize,downsize))(img)
            img = transforms.Resize((self.img_size,self.img_size))(img)
            return img

    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
                
        if not self.landmarks:
            imgpath = self.images[idx]
            image = Image.open(f"{imgpath}").convert('RGB')

            valid_transform = transforms.Compose([
                        transforms.Resize((self.img_size,self.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(means, stds),
                    ])

            transformed_image = valid_transform(image)
            return transformed_image, imgpath.split("/")[-1]
        else:
            imgpath = self.images[idx]
            image = Image.open(f"{self.prefix}/{imgpath}").convert('RGB')
            landmark = np.array(self.landmarks[idx])

            # # # 顯示結果的 對比
            # original_image = image.copy()
            # original_landmark = copy.deepcopy(landmark)
            

            if self.train:
                # self.p_dict = {
                #     'ColorJitter_b':0.3,
                #     'ColorJitter_c':0.3,
                #     'ColorJitter_s':0.3,
                #     'ColorJitter_h':0.1,
                #     'RandomGrayscale':0,
                #     'random_scale_p':0,
                #     'random_scale_scale':0.8,
                #     'random_rotate_p':0,
                #     'random_rotate_degree':30,
                #     'random_blur_p': 0,
                #     'random_blur_min': 128,
                #     'random_horizontalfip_p':0,
                #     'random_erasemargin_p':0,
                #     'random_erasemargin_ratio':0.7,
                #     'RandomErasing_p':0,
                #     'RandomErasing_scale_min':0.02,
                #     'RandomErasing_scale_max':0.1,
                # }

                train_transform = transforms.Compose([ 
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.ColorJitter(brightness=self.p_dict['ColorJitter_b'], contrast = self.p_dict['ColorJitter_c'], saturation = self.p_dict['ColorJitter_s'], hue = self.p_dict['ColorJitter_h']),
                            transforms.RandomGrayscale(p=self.p_dict['RandomGrayscale']),
                        ])

                
                landmark = landmark/384*self.img_size
                image = train_transform(image)
                
                image, landmark = self.random_scale(image, landmark, p=self.p_dict['random_scale_p'], scale=self.p_dict['random_scale_scale'], img_size=self.img_size)
                image, landmark = self.random_rotate(image, landmark, p=self.p_dict['random_rotate_p'], degree=[-1*self.p_dict['random_rotate_degree'],self.p_dict['random_rotate_degree']], img_size=self.img_size)
                image = self.random_blur(image, p=self.p_dict['random_blur_p'],min_size=self.p_dict['random_blur_min'])
                image = transforms.ToTensor()(image)

                image, landmark = self.random_horizontalfip(image, landmark, p=self.p_dict['random_horizontalfip_p'], img_size=self.img_size)
                image = self.random_erasemargin(image, p=self.p_dict['random_erasemargin_p'], ratio=self.p_dict['random_erasemargin_ratio'], img_size=self.img_size)
                image= transforms.RandomErasing(p=self.p_dict['RandomErasing_p'], scale=(self.p_dict['RandomErasing_scale_min'], self.p_dict['RandomErasing_scale_max']), ratio=(0.3, 3.3), value=(0.5, 0.5, 0.5))(image) # 因為是在圖片中，感覺pad 0.5(灰色)比較好

                
                # # 顯示結果
                # import matplotlib.pyplot as plt
                # from show import plot_dots
                # original_image = transforms.ToTensor()(original_image)

                # f, axarr = plt.subplots(1,2) 
                # axarr[0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
                # plot_dots(axarr[0], np.squeeze(landmark).reshape((68,2)),face_dot_color='red',plot_bez_curve=False,show_num=True,plot_aux=False)
                # axarr[1].imshow(np.transpose(original_image.cpu().numpy(), (1, 2, 0)))
                # plot_dots(axarr[1], np.squeeze(original_landmark).reshape((68,2)),face_dot_color='red',plot_bez_curve=False,show_num=True,plot_aux=False)
                # plt.show()

                
                # train 的 transform沒有normalize
                image = transforms.Normalize(means, stds)(image)
            else:
                valid_transform = transforms.Compose([
                                transforms.Resize((self.img_size,self.img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(means, stds),
                            ])
                landmark = landmark/384*self.img_size
                image = valid_transform(image)
            
            # landmark 從self.img_size rescale到-0.5~0.5
            landmark = (landmark-int(self.img_size/2))/self.img_size
            landmark = landmark.astype('float32')
                
            return image, landmark

