
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(25, 138384), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        out = self.relu(x)
        return out


        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=136):
        super(myResnet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),

                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        self.conv1_with_res = residual_block(16,16)
        self.conv2_with_res = residual_block(16,16)


        self.fc1 = nn.Sequential(nn.Linear(138384, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=136, kernel_size=3, padding=1)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
    
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1_with_res(x)
        x = self.conv1_with_res(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

# class myCNNModel:
#     def get_model(self, arch=ASMNet, output_len):
#         if arch == 'ASMNet':
#             model = self.create_ASMNet(inp_shape=[224, 224, 3], output_len=output_len)

#         elif arch == 'mobileNetV2':
#             model = self.create_mobileNet(inp_shape=[224, 224, 3], output_len=output_len)

#         return model

#     def create_ASMNet(self, output_len, inp_tensor=None, inp_shape=None):

#         block_15_add_mpool = GlobalAveragePooling2D()(block_15_add)


class ASMNet(nn.Module):
    def __init__(self, num_out=136):
        super(ASMNet, self).__init__()
        self.num_out = num_out
        self.mobilenet_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        # block_1_project_BN = self.mobilenet_model.features[4]  # 56*56*24
        # pool_1 = torch.nn.AdaptiveAvgPool2d(24)
        # block_3_project_BN = self.mobilenet_model.features[7]  # 28*28*32
        # pool_3 = torch.nn.AdaptiveAvgPool2d(12)
        # block_6_project_BN = self.mobilenet_model.features[11]  # 14*14*64
        # pool_6 = torch.nn.AdaptiveAvgPool2d(6)
        # block_10_project_BN = self.mobilenet_model.features[14]  # 14*14*96
        # pool_10 = torch.nn.AdaptiveAvgPool2d(3)
        # block_13_project_BN = self.mobilenet_model.features[15]  # 7*7*160
        # pool_13 = torch.nn.AdaptiveAvgPool2d(2)
        # block_15_project_BN = self.mobilenet_model.features[17]  # 7*7*160
        # pool_15 = torch.nn.AdaptiveAvgPool2d(1)
        
        # self.model = nn.Sequential(self.mobilenet_model.features[0],self.mobilenet_model.features[1],self.mobilenet_model.features[2],block_1_project_BN,pool_1,block_3_project_BN,pool_3,block_6_project_BN,pool_6,block_10_project_BN,pool_10,block_13_project_BN,pool_13,block_15_project_BN,pool_15,torch.nn.Dropout(0.3),torch.nn.Flatten())
        self.mobilenet_model.classifier[1] = nn.Linear(1280, 136)
        # self.fc= nn.Linear(1000, 136)

        mobilenet_model_2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        block_1_project_BN_2 = mobilenet_model_2.features[4]  # 56*56*24
        pool_1_2 = torch.nn.AdaptiveAvgPool2d(24)
        block_3_project_BN_2 = mobilenet_model_2.features[7]  # 28*28*32
        pool_3_2 = torch.nn.AdaptiveAvgPool2d(12)
        block_6_project_BN_2 = mobilenet_model_2.features[11]  # 14*14*64
        pool_6_2 = torch.nn.AdaptiveAvgPool2d(6)
        block_10_project_BN_2 = mobilenet_model_2.features[14]  # 14*14*96
        pool_10_2 = torch.nn.AdaptiveAvgPool2d(3)
        block_13_project_BN_2 = mobilenet_model_2.features[15]  # 7*7*160
        pool_13_2 = torch.nn.AdaptiveAvgPool2d(2)
        block_15_project_BN_2 = mobilenet_model_2.features[17]  # 7*7*160
        pool_15_2 = torch.nn.AdaptiveAvgPool2d(1)
        
        self.model_2 = nn.Sequential(mobilenet_model_2.features[0],mobilenet_model_2.features[1],mobilenet_model_2.features[2],block_1_project_BN_2,pool_1_2,block_3_project_BN_2,pool_3_2,block_6_project_BN_2,pool_6_2,block_10_project_BN_2,pool_10_2,block_13_project_BN_2,pool_13_2,block_15_project_BN_2,pool_15_2,torch.nn.Dropout(0.3),torch.nn.Flatten())
        self.fc_2= nn.Linear(320, 32)

    def forward(self,x):
        x_face = self.mobilenet_model(x)
        # x_face = self.fc(x_face)
        x_mouth = self.model_2(x)
        x_mouth = self.fc_2(x_mouth)

        out = torch.cat([x_mouth,x_face],1)
        x_face[:,:32]=(x_face[:,:32]+x_mouth)/2
        # print(out.shape)
        return x_face
