import torch
import os
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
# import tensorflow as tf
from myDatasets import  get_dataset
from tool import train, test, fixed_seed, NMSELoss

# Modify config if you are conducting different models
from cfg import cfg,hyper
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from torchinfo import summary
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm
import pickle5 
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['CUDA_VISIBLE_DEVICES']

""" input argumnet """


def predict_from_img(img_path):
    cfg['model_type'] = 'Tune_hyperv2_bs'+str(hyper['batch_size'])+'_lr'+str(hyper['lr'])+'_t0'+str(hyper['cos_T_0'])+'_tm'+str(hyper['cos_T_multi'])
    #cfg['model_type'] = 'Tune_hyper_lr'+str(hyper['lr'])+'_t0_'+str(hyper['cos_T_0'])+'_tm_'+str(hyper['cos_T_multi'])
    model_type = cfg['model_type']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    batch_size = hyper['batch_size']
    loaded_model = cfg['loaded_model']
    img_size = hyper['img_resize']
    
    # fixed random seed
    fixed_seed(seed)
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # print model's architecture
    model = models.mobilenet_v2(width_mult= 1.25,pretrained=False,num_classes=136)

    # summary(model, input_size=(batch_size, 3, 384, 384))

    PATH = "./save_dir/"+ model_type + "/"+ loaded_model
    model.load_state_dict(torch.load(PATH,map_location='cuda'))
    model = model.to(device)

    # load image
    image = Image.open(img_path).convert('RGB')

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    valid_transform = transforms.Compose([
                                transforms.Resize((img_size,img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(means, stds),
                            ])
    transformed_image = valid_transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        predicted_landmarks = model(transformed_image.to(device))

    predicted_landmarks = predicted_landmarks.cpu().numpy()*384+192
    predicted_landmarks = np.squeeze(predicted_landmarks).reshape((68,2))
    return predicted_landmarks

def imshow_from_path(ax,img_path):
    im = plt.imread(img_path)
    implot = ax.imshow(im)

    

def plot_dots(ax, landmark,degree=[8,3,3,3],face_dot_color=None,plot_bez_curve=True,show_num=True,plot_aux=True):
    x = [i[0] for i in landmark]
    y = [i[1] for i in landmark]
    #print(ax)
    plot_face_dot(x,y,ax,color=face_dot_color)

    if plot_bez_curve:
        plot_face_outline(x,y,ax,degree=degree,plot_aux=plot_aux)

    if show_num:
        for i in range(68):
            ax.annotate(str(i), (x[i], y[i]))



def plot_face_dot(x,y,ax,color=None):
    if color is None:
        ax.scatter(x[:17], y[:17],s=3,c="red")
        ax.scatter(x[17:27], y[17:27],s=3,c="yellow")
        ax.scatter(x[27:36], y[27:36],s=3,c="blue")
        ax.scatter(x[36:48], y[36:48],s=3,c="orange")
        ax.scatter(x[48:], y[48:],s=3,c="gray")
    else:
        ax.scatter(x, y,s=3,c=color)

def plot_face_outline(x,y,ax,color='white',degree=[6,3,3,3],plot_aux=True):
    
    plot_bez(x,y,0,17,ax,degree[0],color=color,plot_aux=plot_aux)
    plot_bez(x,y,17,22,ax,degree[1],color=color,plot_aux=False)
    plot_bez(x,y,22,27,ax,degree[1],color=color,plot_aux=False)
    plot_bez(x,y,27,31,ax,degree[2],color=color,plot_aux=False)
    plot_bez(x,y,31,36,ax,degree[3],color=color,plot_aux=False)

def plot_bez(x,y,start,end,ax,degree,color='blue',nTimes=1000,plot_aux=True):
    data = get_bezier_parameters(x[start:end],y[start:end], degree=degree)
    # data is a list, the length of data is (degree+1).
    x_val = [x[0] for x in data]
    y_val = [x[1] for x in data]

    if plot_aux:
        ax.plot(x_val,y_val,'k--o', label='Control Points')

    xvals, yvals = bezier_curve(data, nTimes=nTimes)
    ax.plot(xvals, yvals, '-', label='B Curve',c=color)
    # ax.plot(xvals, yvals, 'o', label='B Curve',c=color)


def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bezier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bezier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bezier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))

    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return np.array(final).astype(np.float32)

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """
    nPoints = len(points)
    #print(nPoints)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)
    
    # polynomial_array shape: ( len(points),len(t) )
    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    # xvals, yvals shape : (len(t),)
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    
    
    return xvals, yvals


def show_val_result(num, plot=True,plot_threshold=0.0,save=False, plot_bez_curve=False, plot_aux=False):
    
    root_type = 'valid_data_root'
    data_root = cfg[root_type]
    data_dir = os.path.dirname(data_root)
    with open(data_root, 'rb') as f:
        file_path, annot = pickle5.load(f)

    img_name = file_path[num]
    predicted_landmarks = predict_from_img(os.path.join(data_dir,img_name))
    predicted_landmarks = predicted_landmarks
    ground_landmarks = annot[num]

    # count NMSE
    gt = torch.Tensor(np.array(annot[num]))
    pred = torch.Tensor(predicted_landmarks)
    nmse = NMSELoss(gt,pred).item()

    if plot and nmse>plot_threshold:
        fig, ax = plt.subplots()
        imshow_from_path(ax, os.path.join(data_dir,img_name))
        plot_dots(ax, predicted_landmarks,face_dot_color='blue',plot_bez_curve=False,show_num=False,plot_aux=False)
        plot_dots(ax, annot[num],face_dot_color='red',plot_bez_curve=plot_bez_curve,show_num=False,plot_aux=plot_aux)
        fig.suptitle(str(num)+' ['+img_name+'], NMSE='+f"{nmse:.5f}", fontsize=8)

        if save:
            os.makedirs( './save_results', exist_ok=True)
            fig.savefig(os.path.join('./save_results', str(num)+'.jpg'))
            plt.close(fig)
        #else:
        #    plt.show()

    return nmse

def find_img(imgae_path):
    root_type = 'valid_data_root'
    data_root = cfg[root_type]
    data_dir = os.path.dirname(data_root)
    with open(data_root, 'rb') as f:
        file_path, annot = pickle5.load(f)

    for i in range(199):
        if file_path[i]==imgae_path:
            return i


if __name__ == '__main__':
    
   
    
    ## 2. 
    nmse_list = []
    for num in tqdm(range(199)):
        nmse = show_val_result(num, plot=False)
        nmse_list.append(nmse)
    
    n, bins, patches = plt.hist(nmse_list, 20, color='g')
    plt.xlabel('nmse')
    plt.ylabel('Frequency')
    plt.title('Histogram of NMSE on val')
    plt.xlim(0.0, 20.0)
    plt.ylim(0, 40)
    plt.grid(True)
    #plt.show()
    os.makedirs( './save_results', exist_ok=True)
    plt.savefig(os.path.join('./save_results','NMSE.jpg'))
    
    #plt.close()

    
    ## 3. 
    for num in tqdm(range(199)):
        show_val_result(num, plot=True,plot_threshold=1.8,save=True, plot_bez_curve=False, plot_aux=False)
    

    



    