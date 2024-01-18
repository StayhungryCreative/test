# Denoises RAW iamges from the Darmstadt dataset.

import os
import h5py
import numpy as np
import torch
from torchvision import transforms
from torchvision import utils
from dataloader import process
from models import *
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, default='./run_20190711/checkpoint_400.tar',
                        help="Location from which any pre-trained model needs to be loaded.")
    parser.add_argument("--data_dir", type=str, default='./dnd_2017/',
                        help="Directory containing the Darmstadt RAW images.")
    parser.add_argument("--results_dir", type=str, default='./results_amber/',
                        help="Directory to store the results in.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Select the args.gpu_id to run the code on')
    return parser.parse_args()
# python dnd_denoise.py --load_model ./run_20190711/checkpoint_400.tar  --data_dir ./dnd_2017/ --results_dir ./results/       


args = get_arguments()
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# Loads image information and bounding boxes.
# bounding boxes 是什么？？？
info = h5py.File(os.path.join(args.data_dir, 'info.mat'), 'r')['info']
bb = info['boundingboxes']

# Create generator model
torch.cuda.set_device(args.gpu_id)
model = Generator().cuda()
model = nn.DataParallel(model, device_ids=[args.gpu_id]).cuda()

if args.load_model is not None:
    print('Loading pre-trained checkpoint %s' % args.load_model)
    model_psnr = torch.load(args.load_model, map_location='cuda:0')['avg_psnr']
    model_ssim = torch.load(args.load_model, map_location='cuda:0')['avg_ssim']
    print('Avg. PSNR and SSIM values recorded from the checkpoint: %f, %f' % (model_psnr, model_ssim))
    model_state_dict = torch.load(args.load_model, map_location='cuda:0')['state_dict']
    model.load_state_dict(model_state_dict)
    
# Denoise each image.
##### 把整组处理改成只处理一张图片，好看处理过程
# for i in range(0, 50):
i = 0
# Loads the noisy image.
filename = os.path.join(args.data_dir, 'images_raw', '%04d.mat' % (i + 1))
print('Processing file: %s' % filename)
img = h5py.File(filename, 'r')
noisy =np.float32(np.array(img['Inoisy']).T)

# Loads raw Bayer color pattern.
# bayer_pattern = np.asarray(info[info['camera'][0][i]]['pattern']).tolist()
"""
5th Feb, 2021
Suggestion from: https://github.com/aasharma90/UnprocessDenoising_PyTorch/issues/6
Using transpose of the bayer pattern fixes the previously observed visualisation/white-balance problem
"""
bayer_pattern = np.asarray(np.array(info[info['camera'][0][i]]['pattern']).T).tolist() 
# print(bayer_pattern)  # [[2.0, 3.0], [1.0, 2.0]]

# Load the camera's (or image's) ColorMatrix2 
xyz2cam = torch.FloatTensor(np.reshape(np.asarray(info[info['camera'][0][i]]['ColorMatrix2']), (3, 3)))
# print('bayer_pattern: ',bayer_pattern)
print('xyz2cam: ',xyz2cam)

# Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])
rgb2cam = torch.mm(xyz2cam, rgb2xyz)
# Normalizes each row.
rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
cam2rgb = torch.inverse(rgb2cam)
# print('cam2rgb: ',cam2rgb)
# print(cam2rgb.size())

# Specify red and blue gains here (for White Balancing)
asshotneutral = info[info['camera'][0][i]]['AsShotNeutral']
# print(asshotneutral[1]/asshotneutral[0], asshotneutral[1]/asshotneutral[2]) # [2.04754165] [1.67519947] 绿色和蓝色的增益？？？
red_gain  =  torch.FloatTensor(asshotneutral[1]/asshotneutral[0])
blue_gain =  torch.FloatTensor(asshotneutral[1]/asshotneutral[2])

# Denoises each bounding box in this image.
boxes = np.array(info[bb[0][i]]).T
# print(boxes.shape) # (20, 4)  为什么一张图有 20 个boxes

# for k in range(20):
for k in range(2):
  # Crops the image to this bounding box.
    idx = [
        int(boxes[k, 0] - 1),
        int(boxes[k, 2]),
        int(boxes[k, 1] - 1),
        int(boxes[k, 3])
    ]
    print(boxes[k, 0])
    print(idx)

    noisy_crop = noisy[idx[0]:idx[1], idx[2]:idx[3]].copy()

    # Flips the raw image to ensure RGGB Bayer color pattern.
    if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
    elif (bayer_pattern == [[2, 1], [3, 2]]):
        noisy_crop = np.fliplr(noisy_crop)
    elif (bayer_pattern == [[2, 3], [1, 2]]):
        noisy_crop = np.flipud(noisy_crop)
    else:
        print('Warning: assuming unknown Bayer pattern is RGGB.')

    # Loads shot and read noise factors.  什么情况？ shot、read噪声参数都有，如果我的数据没法给出相应的参数信息会怎样？
    nlf_h5 = info[info['nlf'][0][i]]
    shot_noise = nlf_h5['a'][0][0]
    read_noise = nlf_h5['b'][0][0]
    # print(shot_noise, read_noise)

    # Extracts each Bayer image plane.
    denoised_crop = noisy_crop.copy()
    height, width = noisy_crop.shape
    noisy_bayer   = []
    for yy in range(2):
        for xx in range(2):
            noisy_crop_c = noisy_crop[yy:height:2, xx:width:2].copy()
            noisy_bayer.append(noisy_crop_c)
    noisy_bayer = np.stack(noisy_bayer, axis=-1)
    # print(np.shape(noisy_bayer))
    variance    = shot_noise * noisy_bayer + read_noise  # 这是什么？
    # print(variance.shape)  # (256, 256, 4)

    totensor_   = transforms.ToTensor()
    noisy_bayer = torch.unsqueeze(totensor_(noisy_bayer), dim=0)
    variance    = torch.unsqueeze(totensor_(variance), dim=0)
    # print(variance.size())   # torch.Size([1, 4, 256, 256])

    # DENOISING THE BAYER IMAGES HERE !
    model.eval()
    raw_image_in = Variable(torch.FloatTensor(noisy_bayer)).cuda()
    raw_image_var= Variable(torch.FloatTensor(variance)).cuda()
    with torch.no_grad():
        raw_image_out = model(raw_image_in, raw_image_var)
    noisy_bayer   = raw_image_in.detach().cpu()
    denoised_bayer= raw_image_out.detach().cpu()
    # DENOISING THE BAYER IMAGES HERE !

    # Flips noisy and denoised bayer images back to original Bayer color pattern.
    if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
    elif (bayer_pattern == [[2, 1], [3, 2]]):
        noisy_bayer = torch.flip(noisy_bayer, dims=[3])
        denoised_bayer = torch.flip(denoised_bayer, dims=[3])
    elif (bayer_pattern == [[2, 3], [1, 2]]):
        noisy_bayer = torch.flip(noisy_bayer, dims=[2])
        denoised_bayer = torch.flip(denoised_bayer, dims=[2])

    # Post-Processing for saving the results
    ccm       = torch.unsqueeze(cam2rgb, dim=0) # torch.Size([1, 3, 3])
    red_g     = torch.unsqueeze(red_gain, dim=0)
    blue_g    = torch.unsqueeze(blue_gain, dim=0)
    # print(noisy_bayer.size())
    noisy_RGB     = process.process(noisy_bayer, red_g, blue_g, ccm)
    denoised_RGB  = process.process(denoised_bayer, red_g, blue_g, ccm)

    # 在这里生成了有噪和去噪的RGB图，为什么？
    out_save = torch.cat((noisy_RGB, denoised_RGB), 3)
    utils.save_image(out_save, args.results_dir + '%04d_%02d.png' % (i + 1, k + 1))
