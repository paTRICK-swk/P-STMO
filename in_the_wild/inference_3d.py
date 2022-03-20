# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import hashlib
import os
import pathlib
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable

def get_varialbe(target):
    num = len(target)
    var = []

    for i in range(num):
        temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
        var.append(temp)

    return var
def input_augmentation(input_2D, input_2D_flip, model_trans, joints_left, joints_right):
    B, T, J, C = input_2D.shape

    input_2D_flip = input_2D_flip.view(B, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D.view(B, T, J, C, 1).permute(0, 3, 1, 2, 4)

    output_3D_flip, output_3D_flip_VTE = model_trans(input_2D_flip)

    output_3D_flip_VTE[:, 0] *= -1
    output_3D_flip[:, 0] *= -1

    output_3D_flip_VTE[:, :, :, joints_left + joints_right] = output_3D_flip_VTE[:, :, :, joints_right + joints_left]
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]

    output_3D_non_flip, output_3D_non_flip_VTE = model_trans(input_2D_non_flip)

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D, output_3D_VTE

def step(opt, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']

    model_trans.eval()

    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    epoch_cnt=0
    out = []
    for _, batch, batch_2d, batch_2d_flip in dataLoader.next_epoch():
        #[gt_3D, input_2D] = get_varialbe([batch, batch_2d])
        #input_2D = Variable(batch_2d).contiguous().cuda().type(torch.cuda.FloatTensor)
        input_2D = torch.from_numpy(batch_2d.astype('float32'))
        input_2D_flip = torch.from_numpy(batch_2d_flip.astype('float32'))
        if torch.cuda.is_available():
            input_2D = input_2D.cuda()
            input_2D_flip = input_2D_flip.cuda()

        N = input_2D.size(0)

        # out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels)
        # out_target[:, :, 0] = 0
        # gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)
        #
        # if out_target.size(1) > 1:
        #     out_target_single = out_target[:, opt.pad].unsqueeze(1)
        #     gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        # else:
        #     out_target_single = out_target
        #     gt_3D_single = gt_3D


        input_2D, output_3D, output_3D_VTE = input_augmentation(input_2D, input_2D_flip, model_trans, joints_left, joints_right)


        output_3D_VTE = output_3D_VTE.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)
        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)

        output_3D_single = output_3D


        pred_out = output_3D_single

        input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)

        pred_out[:, :, 0, :] = 0

        if epoch_cnt == 0:
            out = pred_out.squeeze(1).cpu()
        else:
            out = torch.cat((out, pred_out.squeeze(1).cpu()), dim=0)
        epoch_cnt +=1
    return out.numpy()

def val(opt, val_loader, model):
    with torch.no_grad():
        return step(opt, val_loader, model)