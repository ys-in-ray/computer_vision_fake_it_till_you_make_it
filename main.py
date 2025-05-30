import torch
import os
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
from myDatasets import  get_dataset
from tool import train, test, fixed_seed, NMSELoss
from torchinfo import summary
import numpy as np 
import random

# Modify config if you are conducting different models



""" input argumnet """

def train_interface(cfg,hyper,p_dict, version=None):
    train_data_root = cfg['train_data_root']
    valid_data_root = cfg['valid_data_root']
    test_data_root = cfg['test_data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    
    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)
    
    
    with open(log_path, 'w'):
        pass
    
    ## training setting ##

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    batch_size = hyper['batch_size']
    num_epoch = hyper['num_epoch']
    lr = hyper['lr']
    img_size = hyper['img_resize']
    cos_T_0 = hyper['cos_T_0']
    cos_T_multi = hyper['cos_T_multi']
    
    
    ## Modify here if you want to change your model ##

    model = models.mobilenet_v2(width_mult= 1.25,pretrained=False,num_classes=136)
    
    # print(model)
    # summary(model, input_size=(batch_size, 3, img_size, img_size))

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
      
    train_set, valid_set, _ =  get_dataset(train_data_root, valid_data_root, test_data_root, p_dict=p_dict, img_size=img_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True,worker_init_fn=seed_worker,generator=g)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8,pin_memory=False,worker_init_fn=seed_worker,generator=g)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # define your loss function and optimizer to unpdate the model's parameters.

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cos_T_0, T_mult = cos_T_multi)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, threshold=0.1, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    #criterion = NMSELoss
    criterion = nn.MSELoss()
    #criterion =  ASMLoss(accuracy=90)
    model = model.to(device)
    #summary(model,(3,384,384))
    
    model = train(model=model, train_loader=train_loader, valid_loader=valid_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer,scheduler = scheduler, version=version)

    return model

def p_dict_tuner(cfg,hyper,now_p_dict):
    p_dict_possible = {
            'ColorJitter_b':[0.2,0.3,0.4],
            'ColorJitter_c':[0.2,0.3,0.4],
            'ColorJitter_s':[0.2,0.3,0.4],
            'ColorJitter_h':[0.1],
            'RandomGrayscale':[0.0,0.3,0.6,1.0],
            'random_scale_p':[0.0,0.2,0.4,0.6],
            'random_scale_scale':[0.6,0.8],
            'random_rotate_p':[0.0,0.2,0.4,0.6],
            'random_rotate_degree':[15,30,40],
            'random_blur_p': [0.0,0.2,0.4,0.6],
            'random_blur_min': [128],
            'random_horizontalfip_p':[0,0.5],
            'random_erasemargin_p':[0.0,0.2,0.4,0.6],
            'random_erasemargin_ratio':[0.6,0.8],
            'RandomErasing_p':[0.0,0.2,0.4],
            'RandomErasing_scale_min':[0.02],
            'RandomErasing_scale_max':[0.05,0.1,0.15],
        }

    order = ['ColorJitter_b','ColorJitter_c','ColorJitter_s','RandomGrayscale','random_horizontalfip_p']
    order2 =['random_blur_p',\
        'random_erasemargin_p','random_erasemargin_ratio',\
        'random_scale_p','random_scale_scale',\
        'random_rotate_p','random_rotate_degree',\
        'RandomErasing_p','RandomErasing_scale_max']

    for p_dict_key in order:
        best_NMSE=10
        best_value = 0
        for value in p_dict_possible[p_dict_key]:
            print("Trying "+p_dict_key+": "+str(value))
            now_p_dict[p_dict_key] = value
            cfg['model_type'] = 'Tune_hyperv3_'+p_dict_key+str(now_p_dict[p_dict_key])
            this_best_NMSE, _, _ = train_interface(cfg,hyper,now_p_dict)
            if this_best_NMSE<best_NMSE:
                best_NMSE = this_best_NMSE
                best_value = value
        now_p_dict[p_dict_key] = best_value
        print("~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~")
        print("Find best "+p_dict_key+": "+str(best_value))
        print(now_p_dict)
        print("~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~")

    return now_p_dict

if __name__ == '__main__':
    ## 也可以在這邊tune
    from cfg import cfg, p_dict, hyper
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['CUDA_VISIBLE_DEVICES']
    # hyper = {
    #     'batch_size': 16,
    #     'img_resize': 384,
    #     'lr':5e-3,
    #     'num_epoch': 44,
    #     'cos_T_0':9,
    #     'cos_T_multi':1,
    # }

    # p_dict = {
    #     'ColorJitter_b':0.3,
    #     'ColorJitter_c':0.3,
    #     'ColorJitter_s':0.3,
    #     'ColorJitter_h':0.1,
    #     'RandomGrayscale':0.4,
    #     'random_scale_p':0.2,
    #     'random_scale_scale':0.8,
    #     'random_rotate_p':0.4,
    #     'random_rotate_degree':30,
    #     'random_blur_p': 0.2,
    #     'random_blur_min': 128,
    #     'random_horizontalfip_p':0.5,
    #     'random_erasemargin_p':0.4,
    #     'random_erasemargin_ratio':0.7,
    #     'RandomErasing_p':0.1,
    #     'RandomErasing_scale_min':0.02,
    #     'RandomErasing_scale_max':0.1,
    # }

    # hyper['batch_size'] = 16
    # hyper['num_epoch'] = 1

    cfg['model_type'] = 'Tune_hyperv4'
    best_NMSE, best_train_NMSE, best_NMSE_epoch = train_interface(cfg,hyper,p_dict)

    print("\n\n=====================result======================")
    print(hyper)
    print(p_dict)
    print('model='+cfg['model_type'])
    print('best_NMSE_epoch = '+str(best_NMSE_epoch))
    print('best_train_NMSE = '+str(best_train_NMSE))
    print('best_NMSE = '+str(best_NMSE))




    