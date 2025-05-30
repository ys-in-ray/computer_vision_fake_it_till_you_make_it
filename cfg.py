## You could add some configs to perform other training experiments...

cfg = {

    # 'model_type': 'mobileNetV2_1_25_dev2.5', #寫在main中
    # 'model_type': 'mobileNetV2_1_25_erasing_v2',
    'train_data_root' : './data/synthetics_train/annot.pkl',
    'valid_data_root' : './data/aflw_val/annot.pkl',
    'test_data_root' : './data/aflw_test',
    # ratio of training images and validation images 
    'split_ratio': 1.0,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    # training hyperparameters
    
    'num_out': 136,
    #'loaded_model': "epoch_13.pt",
    'loaded_model': "best_model.pt",
    'CUDA_VISIBLE_DEVICES': '2',
}

hyper = {
    'batch_size': 16,
    'img_resize': 384,
    'lr':5e-3,
    'num_epoch': 48,
    'cos_T_0':8,
    'cos_T_multi':1,
}

p_dict = {
    'ColorJitter_b':0.4,
    'ColorJitter_c':0.2,
    'ColorJitter_s':0.3,
    'ColorJitter_h':0.1,
    'RandomGrayscale':0.15,
    'random_scale_p':0.1,
    'random_scale_scale':0.8,
    'random_rotate_p':0.3,
    'random_rotate_degree':30,
    'random_blur_p': 0.0,
    'random_blur_min': 128,
    'random_horizontalfip_p':0.5,
    'random_erasemargin_p':0.375,
    'random_erasemargin_ratio':0.65,
    'RandomErasing_p':0.2,
    'RandomErasing_scale_min':0.02,
    'RandomErasing_scale_max':0.1,
}