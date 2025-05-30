import torch
import os
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
# import tensorflow as tf
from myDatasets import  get_dataset
from myModels import ASMNet
# Modify config if you are conducting different models
from cfg import cfg
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

def imshow_from_path(ax,img_path):
    im = plt.imread(img_path)
    implot = ax.imshow(im)

    

def plot_dots(ax, landmark,degree=[8,3,3,3],face_dot_color=None,plot_bez_curve=True,show_num=True,plot_aux=True):
    x = [i[0] for i in landmark]
    y = [i[1] for i in landmark]

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
    # degree分別是: 臉、眉毛、鼻樑、鼻底
    plot_bez(x,y,0,17,ax,degree[0],color=color,plot_aux=plot_aux)
    plot_bez(x,y,17,22,ax,degree[1],color=color,plot_aux=False)
    plot_bez(x,y,22,27,ax,degree[1],color=color,plot_aux=False)
    plot_bez(x,y,27,31,ax,degree[2],color=color,plot_aux=False)
    plot_bez(x,y,31,36,ax,degree[3],color=color,plot_aux=False)

def plot_bez(x,y,start,end,ax,degree,color='blue',nTimes=1000,plot_aux=True):
    data = get_bezier_parameters(x[start:end],y[start:end], degree=degree)
    x_val = [x[0] for x in data]
    y_val = [x[1] for x in data]

    if plot_aux:
        ax.plot(x_val,y_val,'k--o', label='Control Points')

    xvals, yvals = bezier_curve(data, nTimes=nTimes)
    ax.plot(xvals, yvals, 'b-', label='B Curve',c=color)
    # ax.plot(xvals, yvals, 'o', label='B Curve',c=color)


def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
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
        """ Bernstein matrix for Bézier curves. """
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
    return final

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
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals



def show_face(num, plot_bez_curve=False, plot_aux=False, root_type = 'valid_data_root'):
    data_root = cfg[root_type]
    data_dir = os.path.dirname(data_root)
    with open(data_root, 'rb') as f:
        file_path, annot = pickle5.load(f)

    img_name = file_path[num]
    fig, ax = plt.subplots()
    imshow_from_path(ax, os.path.join(data_dir,img_name))
    plot_dots(ax, annot[num],face_dot_color='red',plot_bez_curve=plot_bez_curve,show_num=True,plot_aux=plot_aux)
    return fig, ax, annot[num]

def find_img(imgae_path, root_type='valid_data_root'):
    # root_type = 'train_data_root'
    data_root = cfg[root_type]
    data_dir = os.path.dirname(data_root)
    with open(data_root, 'rb') as f:
        file_path, annot = pickle5.load(f)

    for i in range(100000):
        if file_path[i]==imgae_path:
            return i

def default_face():
    # num = find_img('065568.jpg',root_type='train_data_root')
    # print(num)
    return show_face(65420, plot_bez_curve=False, plot_aux=False,root_type='train_data_root')

if __name__ == '__main__':
    
    '''
    ## 1. 畫出每張圖
    for num in tqdm(range(199)):
        show_val_result(num, plot_bez_curve=True,plot_aux=True)
    
    '''

    '''
    ## 2. 畫loss的分布
    nmse_list = []
    for num in tqdm(range(199)):
        nmse_list.append(show_val_result(num, plot=False))
        # print(nmse_list)
    n, bins, patches = plt.hist(nmse_list, 20, color='g')
    plt.xlabel('nmse')
    plt.ylabel('Frequency')
    plt.title('Histogram of NMSE on val')
    plt.xlim(0.0, 20.0)
    plt.ylim(0, 40)
    plt.grid(True)
    plt.show()
    '''

    '''
    ## 3. 如果loss>5 存檔
    for num in tqdm(range(199)):
        show_val_result(num, plot=True,plot_threshold=5.0,save=True, plot_bez_curve=False, plot_aux=False)
    '''

    # ## 4. 畫特定名稱的圖片
    # show_val_result(find_img('image00523.jpg'), plot_bez_curve=True, plot_aux=False)
    
    
    



    