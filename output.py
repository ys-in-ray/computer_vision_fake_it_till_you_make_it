import torch
import os
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn

from myDatasets import  get_dataset
from tool import train, test, fixed_seed, NMSELoss

# Modify config if you are conducting different models
from cfg import cfg, hyper, p_dict

""" input argumnet """

def test_interface():
    train_data_root = cfg['train_data_root']
    valid_data_root = cfg['valid_data_root']
    test_data_root = cfg['test_data_root']
    
    num_out = cfg['num_out']
    num_epoch = hyper['num_epoch']

    seed = cfg['seed']
    batch_size = hyper['batch_size']
    loaded_model = cfg['loaded_model']
    img_size = hyper['img_resize']
    cfg['model_type'] = 'Tune_hyperv2_bs'+str(hyper['batch_size'])+'_lr'+str(hyper['lr'])+'_t0'+str(hyper['cos_T_0'])+'_tm'+str(hyper['cos_T_multi'])
    #cfg['model_type'] = "Tune_hyperv2_bs16_lr0.005_t08_tm1"
    model_type = cfg['model_type']

    # fixed random seed
    fixed_seed(seed)
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu') 
    
    ## Modify here if you want to change your model ##
    #model = models.resnet18(pretrained=False)
    #model.fc = nn.Linear(512, 136) 
    
    #cnn = CNNModel() 
    #model = cnn.get_model(arch= model_type, output_len=num_out) 
    # print model's architecture
    model = models.mobilenet_v2(width_mult= 1.25,pretrained=False,num_classes=136)

    
    PATH = "./save_dir/"+ model_type + "/"+ loaded_model
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)
    
    _, _, test_set =  get_dataset(train_data_root, valid_data_root, test_data_root, img_size=img_size)    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(model_type)
    print(loaded_model)
    test(model = model, test_loader = test_loader, device=device)
    
if __name__ == '__main__':
    test_interface()