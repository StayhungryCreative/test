######### 根据现有代码进行的精简 ########
### 1、去除了 crop，保存的图片是无噪声的真值图
### 2、初步理解了 rgb2cam 的过程


import os
import h5py
import numpy as np
import torch
from torchvision import transforms
from torchvision import utils
from dataloader import process

# result_dir = '/home/amberlwq/Music/UnprocessDenoising_PyTorch/amber_test/result'
# filename = '/home/amberlwq/Music/UnprocessDenoising_PyTorch/dnd_2017/images_raw/0002.mat' 

result_dir = r'E://20240118//amber_test//result_1'
filename = r'E://20240118//0045.mat'
xyz2cam = torch.tensor([[ 1.0234, -0.2969, -0.2266],
        [-0.5625,  1.6328, -0.0469],
        [-0.0703,  0.2188,  0.6406]])
rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])
rgb2cam = torch.mm(xyz2cam, rgb2xyz)
rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
cam2rgb = torch.inverse(rgb2cam)

red_gain = torch.tensor([2.0475])
blue_gain = torch.tensor([1.6752])
ccm       = torch.unsqueeze(cam2rgb, dim=0)
red_g     = torch.unsqueeze(red_gain, dim=0)
blue_g    = torch.unsqueeze(blue_gain, dim=0)


img = h5py.File(filename, 'r')
noisy =np.float32(np.array(img['Inoisy']).T)
noisy = np.flipud(noisy) # 干嘛？？

height, width = noisy.shape
noisy_bayer   = []
for yy in range(2):
    for xx in range(2):
        noisy_c = noisy[yy:height:2, xx:width:2].copy()
        noisy_bayer.append(noisy_c)
noisy_bayer = np.stack(noisy_bayer, axis=-1)

totensor_   = transforms.ToTensor()
noisy_bayer = torch.unsqueeze(totensor_(noisy_bayer), dim=0)

noisy_bayer = torch.flip(noisy_bayer, dims=[2])
noisy_RGB     = process.process(noisy_bayer, red_g, blue_g, ccm)

utils.save_image(noisy_RGB, os.path.join(result_dir, 'test.png' ))


### 中间测试工作
# import os
# import h5py
# import numpy as np
# input_dir = '/home/amberlwq/Music/UnprocessDenoising_PyTorch/dnd_2017'
# result_dir = '/home/amberlwq/Music/UnprocessDenoising_PyTorch/unprocess_denoising/result'

# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

# info = h5py.File(os.path.join(input_dir, 'info.mat'), 'r')['info']
# # print(info)
# # print(info['boundingboxes'])
# # print(info[info['boundingboxes'][0][1]])  # <HDF5 dataset "c": shape (4, 20), type "<f8">

# boxes = np.array(info[info['boundingboxes'][0][1]]).T
# # print(boxes.shape) # (20, 4)

# # print(info['camera'])
# # print(info['camera'][0][1])
# # print(info[info['camera'][0][1]])
# # print(info[info['camera'][0][1]]['ColorMatrix2'])



### 失败尝试
# import scipy.io as sio
# datafile='/home/amberlwq/Music/UnprocessDenoising_PyTorch/dnd_2017/images_raw/0001.mat'   #绝对路径
# dataAll=sio.loadmat(datafile)
# print(dataAll)
# print(dataAll['y'],'\n',dataAll['data'],dataAll['y'].shape,dataAll['data'].shape)
# # NotImplementedError: Please use HDF reader for matlab v7.3 files