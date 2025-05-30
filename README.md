# CV-final

[![hackmd-github-sync-badge](https://hackmd.io/-DW5a4A2S5a6y3BHbvnFSQ/badge)](https://hackmd.io/-DW5a4A2S5a6y3BHbvnFSQ)

![](https://hackmd.io/_uploads/S1f0Zaftq.png)



目錄
[TOC]

## 環境
(從0開始，不過@蕭的結果會不一樣)
```=
conda update -n base -c defaults conda
conda create -n CVfinal python=3.8.8
source activate CVfinal
conda install -c anaconda numpy
conda install -c anaconda pillow
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge torchinfo
conda install -c conda-forge matplotlib
conda install -c anaconda scipy
```
(!!! 推薦)或者直接安裝requirements
```=
conda create --name CVfinal --file requirement/spec-file.txt
conda env update --name CVfinal --file requirement/environment.yml
pip install -r requirement/requirements.txt
```




## 連結&資源
- [最新版投影片](https://cool.ntu.edu.tw/courses/14814/files/2341311?module_item_id=611515)
- [GitHub連結](https://github.com/hsin-ray-yang/CV-Final-Project)
- [(本篇)HackMD連結](https://hackmd.io/-DW5a4A2S5a6y3BHbvnFSQ?both)
- [(下方)reference](#reference)
- [Colab (帳號:YCHsiao)](https://codalab.lisn.upsaclay.fr/competitions/5118)
- [Score&方法紀錄(google試算表)](https://docs.google.com/spreadsheets/d/1T3hRyc_lFPZMFFxLW0YNCqX08Zh-z6GU2rfbYl1vJXA/edit?fbclid=IwAR0W23ctYJiNz4WcBVa5wkv3XrIbNLa0bd9xpgwgkKkqp-Djn3sZuxe5u0I#gid=0)

## abstract
We aim to build a model to predict 68 2D facial landmarks in a single cropped face image with __high accuracy__, __high efficiency__, and __low computational costs__.

## Dataset
- training set
    - 100000張++生成的++圖片，每張68個點
- val set
    - 199張++真實的++圖片，每張68個點
- test set
    - 1790張++真實的++圖片，每張68個點

## preprocess
![](https://hackmd.io/_uploads/ByGpPcGY5.png)
![](https://hackmd.io/_uploads/S1SgDiGt9.png)
![](https://hackmd.io/_uploads/Hyiwuozt5.png)



---
## reference
### dataset
- [ICCV 2021] [Fake It Till You Make It](https://microsoft.github.io/FaceSynthetics/)
### Codes 
- [Facial Landmark Detection](https://paperswithcode.com/task/facial-landmark-detection/latest?fbclid=IwAR3WGqjxO8wp0twFb88YMBY7mnvY228eJZxPvloM1JkYWNDLDnGfy_MwsWk)
### works
- [ECCV 2020] [Towards Fast, Accurate and Stable 3D Dense Face Alignment](https://github.com/cleardusk/3DDFA_V2)
- [CVPRW 2019] [Accurate 3D Face Reconstruction with Weakly-Supervised Learning](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
- [ICCV 2017] [How far are we from solving the 2D & 3D Face Alignment problem?](https://github.com/1adrianb/2D-and-3D-face-alignment)




###### tags: `CV`













































