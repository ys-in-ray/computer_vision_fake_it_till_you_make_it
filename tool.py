import torch
import torch.nn as nn
import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
from cfg import cfg
import math
#from show_tool import default_face

class Mean_counter:
    def __init__(self, start=0, sum=None):
        self.count=start
        self.sum = sum

    def add(self,x):
        if self.count==0:
            self.sum=x
            self.count=1
        else: 
            self.count+=1
            self.sum+=x

    def mean(self):
        return self.sum/self.count

    def normalized(self):
        mean = self.sum/self.count
        return mean/torch.sum(mean)

class Curve_plotter:
    def __init__(self, num_curve=1, title_name=None, curve_name=None):
        self.num_curve=num_curve
        self.len = 0
        self.data = []
        self.curve_name = curve_name
        self.title_name = title_name

    def add(self,*args):
        if self.len==0:
            self.data = [[arg] for arg in args]
        else: 
            for i in range(self.num_curve):
                self.data[i].append(args[i])
        self.len+=1

    def plot(self, fig=None, ax=None):
        x = [i for i in range(self.len)]
        if ax is None:
            for j in range(self.num_curve):
                if self.curve_name is None:
                    plt.plot(x, self.data[j])
                else:
                    plt.plot(x, self.data[j], label=self.curve_name[j])

                for k,l in zip(x,self.data[j]):
                    ax.annotate(f"{l:.2E}" if (l>=100 or l<0.01) else f"{l:.3f}",xy=(k,l), fontsize=5)

            if self.curve_name is not None:
                plt.legend()

            if self.title_name is not None:
                plt.title(self.title_name)

            plt.xlabel('epoch')

            os.makedirs( './save_figs', exist_ok=True)
            plt.savefig(os.path.join('./save_figs', 'no_title.jpg' if self.title_name is None else self.title_name))
            plt.close()
        else:
            for j in range(self.num_curve):
                if self.curve_name is None:
                    ax.plot(x, self.data[j])
                else:
                    ax.plot(x, self.data[j], label=self.curve_name[j])

                for k,l in zip(x,self.data[j]):
                    ax.annotate(f"{l:.2E}" if (l>=100 or l<0.01) else f"{l:.3f}",xy=(k,l), fontsize=5)

            if self.curve_name is not None:
                ax.legend()

            if self.title_name is not None:
                ax.set_title(self.title_name)

            



def test(model, test_loader, device):
    model.eval()
    # Initialize a list to store the predictions.
    predictions = []

    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):
        # A batch consists of image data and corresponding labels.
        # But here the variable "labels" is useless since we do not have the ground-truth.
        # If printing out the labels, you will find that it is always 0.
        # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
        # so we have to create fake labels to make it work normally.
        imgs, imgpaths = batch

        # We don't need gradient in testing, and we don't even have labels to compute loss.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            predicted_landmarks = model(imgs.to(device))
            predicted_landmarks = predicted_landmarks*384+192
        # Take the class with greatest logit as prediction and record it.
        # print(type(imgpaths))
        # print(imgpaths)
        for imgpath, predicted_landmark in zip(imgpaths, predicted_landmarks):
            predictions.append([imgpath] + predicted_landmark.cpu().numpy().tolist())
    # Save predictions into the file.
    with open("solution.txt", "w") as f:
        # For the rest of the rows, each image id corresponds to a predicted class.
        for pred in predictions:
            f.write("{}\n".format(" ".join([str(i) for i in pred])))


def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
        
        
def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")


def NMSELoss(landmark, predicted_landmark):
    '''
    landmark, predicted_landmark: batchsize by num_landmark*2
    
    loss = (landmark - predicted_landmark)
    loss = torch.sqrt(torch.sum(torch.pow(loss, 2), 1))
    loss = torch.mean(loss)
    return loss/384
    '''
    loss = (landmark.reshape((landmark.shape[0],int(landmark.shape[1]/2),2)) - predicted_landmark.reshape((landmark.shape[0],int(landmark.shape[1]/2),2)))
    loss = torch.sqrt(torch.sum(torch.pow(loss, 2), 2))
    loss = torch.mean(loss,axis=1)
    return torch.mean(loss / 384)*100
    
def WingLoss(landmark, predicted_landmark, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmark, predicted_landmark: float tensors with shape [batch_size, num_landmarks*2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    x = landmark - predicted_landmark
    c = w * (1.0 - math.log(1.0 + w/epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(
    torch.gt(torch.tensor(w), absolute_x),
    w * torch.log(1.0 + absolute_x/epsilon),
    absolute_x - c
    )
    loss = torch.mean(torch.sum(losses, axis=1), axis=0)
    return loss



def loss_each_landmark(landmark, predicted_landmark):
    loss = (landmark.reshape((landmark.shape[0],int(landmark.shape[1]/2),2)) - predicted_landmark.reshape((landmark.shape[0],int(landmark.shape[1]/2),2)))
    loss = torch.sqrt(torch.sum(torch.pow(loss, 2), 2))
    # loss = torch.mean(loss,axis=1)
    return torch.mean(loss,axis=0)

'''
def plot_loss_each_landmark(each_landmark, save_path):
    fig, ax, pos = default_face()
    for i in range(68):
        draw_circle = plt.Circle((pos[i][0], pos[i][1]), each_landmark[i],fill=False)
        ax.add_artist(draw_circle)

    os.makedirs( './save_results', exist_ok=True)
    fig.savefig(os.path.join('./save_results', save_path))
    plt.close(fig)
'''
        

def train(model, train_loader, valid_loader, num_epoch, log_path, save_path, device, criterion, scheduler, optimizer, version=None):
    start_train = time.time()
    print('start training: '+cfg['model_type'])
    print('run on CUDA: '+cfg['CUDA_VISIBLE_DEVICES'])

    overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_NMSE = np.zeros(num_epoch ,dtype = np.float32)
    overall_valid_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_valid_NMSE = np.zeros(num_epoch ,dtype = np.float32)

    best_NMSE = 1000
    best_train_NMSE = 1000
    best_NMSE_epoch = 0
    lr_list = []

    lr_plotter = Curve_plotter(num_curve=1, title_name='learning rate')
    loss_plotter = Curve_plotter(num_curve=2, title_name='loss', curve_name=['train loss', 'valid loss'])
    NMSE_plotter = Curve_plotter(num_curve=2, title_name='NMSE', curve_name=['train NMSE', 'valid NMSE'])

    for i in range(num_epoch):
        print(f'epoch = {i}')
        print('lr: ',end='')
        this_epoch_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_plotter.add(this_epoch_lr)
        print(this_epoch_lr)
        # epcoch setting
        start_time = time.time()
        train_loss = Mean_counter()
        train_NMSE = Mean_counter()
        each_landmark=Mean_counter()

        # training part
        # start training
        model.train()

        pbar = tqdm(train_loader)
        for batch_idx, ( data, landmark,) in enumerate(pbar):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            landmark = landmark.to(device)
            # pass forward function define in the model and get output 
            predicted_landmark = model(data) 

            landmark = landmark*384+192
            predicted_landmark = predicted_landmark*384+192
            #print(predicted_landmark.shape)
            # calculate the loss between output and ground truth

            loss = criterion(landmark, predicted_landmark)

            # discard the gradient left from former iteration 
            optimizer.zero_grad()

            # calcualte the gradient from the loss function 
            loss.backward()
            
            # if the gradient is too large, we dont adopt it
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # Update the parameters according to the gradient we calculated
            optimizer.step()
            train_loss.add(loss.item())

            # here we calculate the NMSE
            this_time_MSE = NMSELoss(landmark, predicted_landmark).detach().item()
            # print(this_time_MSE)
            train_NMSE.add(this_time_MSE)

            pbar.set_description("NMSE %.4f" % train_NMSE.mean())

            each_landmark.add(loss_each_landmark(landmark, predicted_landmark).detach())

            # if batch_idx>len(train_loader)/600:
            #     pbar.close()
            #     break

        # scheduler += 1 for adjusting learning rate later
        scheduler.step()

        # scheduler.step(train_NMSE.mean())

        # visualize
        #plot_loss_each_landmark(each_landmark.mean(), cfg['model_type']+'_epoch'+str(i)+'.jpg')
        
        ## TO DO ##

        model.eval()

        valid_loss = Mean_counter()
        valid_NMSE = Mean_counter() 
        for valid_data, valid_landmark in tqdm(valid_loader):
            valid_data = valid_data.to(device)
            valid_landmark = valid_landmark.to(device)
            with torch.no_grad():
                predicted_landmark = model(valid_data) 

            valid_landmark = valid_landmark*384+192
            predicted_landmark = predicted_landmark*384+192

            loss = criterion(valid_landmark, predicted_landmark)
            valid_loss.add(loss.item())

            # here we calculate the NMSE
            valid_NMSE.add(NMSELoss(valid_landmark, predicted_landmark).detach().item())


        loss_plotter.add(train_loss.mean(), valid_loss.mean())
        NMSE_plotter.add(train_NMSE.mean(), valid_NMSE.mean())

        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        #print(f'epoch = {i}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss.mean():.4f} ', f' train NMSE = {train_NMSE.mean():.4f}' )
        print(f'valid loss : {valid_loss.mean():.4f} ', f' valid NMSE = {valid_NMSE.mean():.4f}' )
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('lr = {:.3f}\n'.format(this_epoch_lr) )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss.mean()}  train NMSE = {train_NMSE.mean()}\n' )
            f.write(f'valid loss : {valid_loss.mean()}  valid NMSE = {valid_NMSE.mean()}\n' )
            f.write('============================\n')

        # save model for every epoch 
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{i}.pt'))
        
        # save the best model if it gain performance on validation set
        if  best_NMSE > valid_NMSE.mean():
            best_NMSE = valid_NMSE.mean()
            best_train_NMSE = train_NMSE.mean()
            best_NMSE_epoch = i
            print("Best model is at epoch:",best_NMSE_epoch)
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))

        fig, axs = plt.subplots(1,3,figsize=(20,5))
        lr_plotter.plot(fig, axs[0])
        loss_plotter.plot(fig, axs[1])
        NMSE_plotter.plot(fig, axs[2])
        axs[0].set_xlabel('epoch')
        axs[1].set_xlabel('epoch')
        axs[2].set_xlabel('epoch')
        fig.suptitle(version)
        os.makedirs( './save_figs', exist_ok=True)
        fig.savefig(os.path.join('./save_figs', cfg['model_type']+'.jpg'))
        plt.close()

    return best_NMSE, best_train_NMSE, best_NMSE_epoch

