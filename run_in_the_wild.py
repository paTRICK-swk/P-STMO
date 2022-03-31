import os
import glob
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_hm36_tds_in_the_wild import Fusion
from common.h36m_dataset import Human36mDataset
from model.block.refine import refine
from model.stmo import Model
from model.stmo_pretrain import Model_MAE

from thop import clever_format
from thop.profile import profile

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test',  opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']
    model_MAE = model['MAE']

    if split == 'train':
        model_trans.train()
        model_refine.train()
        model_MAE.train()
    else:
        model_trans.eval()
        model_refine.eval()
        model_MAE.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()

    action_error_sum = define_error_list(actions)
    action_error_sum_post_out = define_error_list(actions)
    action_error_sum_MAE = define_error_list(actions)

    joints_left = [4, 5, 6, 11, 12, 13]  
    joints_right = [1, 2, 3, 14, 15, 16]

    for i, data in enumerate(tqdm(dataLoader, 0)):

        if opt.MAE:
            batch_cam, input_2D, action, subject, scale, bb_box, cam_ind = data
            [input_2D, batch_cam, scale, bb_box] = get_varialbe(split,[input_2D, batch_cam, scale, bb_box])

            N = input_2D.size(0)
            f = opt.frames

            mask_num = int(f*opt.temporal_mask_rate)
            mask = np.hstack([
                np.zeros(f - mask_num),
                np.ones(mask_num),
            ]).flatten()

            np.random.seed()
            np.random.shuffle(mask)

            mask = torch.from_numpy(mask).to(torch.bool).cuda()

            spatial_mask = np.zeros((f, 17), dtype=bool)
            for k in range(f):
                ran = random.sample(range(0, 16), opt.spatial_mask_num)
                spatial_mask[k, ran] = True


            if opt.test_augmentation and split == 'test':
                input_2D, output_2D = input_augmentation_MAE(input_2D, model_MAE, joints_left, joints_right, mask, spatial_mask)

            else:
                input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(
                    torch.cuda.FloatTensor)
                output_2D = model_MAE(input_2D, mask, spatial_mask)


            input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)
            output_2D = output_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)
            #a = input_2D[:, mask]


            loss = mpjpe_cal(output_2D, torch.cat((input_2D[:, ~mask], input_2D[:, mask]), dim=1))
            #my_loss_one = torch.mean(torch.norm(output_2D[20,180]-a[20,180], dim=1))


        else:
            batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
            [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split,
                                                                       [input_2D, gt_3D, batch_cam, scale, bb_box])

            N = input_2D.size(0)

            out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels)
            out_target[:, :, 0] = 0
            gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

            if out_target.size(1) > 1:
                out_target_single = out_target[:, opt.pad].unsqueeze(1)
                gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
            else:
                out_target_single = out_target
                gt_3D_single = gt_3D

            if opt.test_augmentation and split =='test':
                input_2D, output_3D, output_3D_VTE = input_augmentation(input_2D, model_trans, joints_left, joints_right)
            else:
                input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor)
                output_3D, output_3D_VTE = model_trans(input_2D)

            output_3D_VTE = output_3D_VTE.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)
            output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)

            output_3D_VTE = output_3D_VTE * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D_VTE.size(1),opt.out_joints, opt.out_channels)
            output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.out_joints, opt.out_channels)
            output_3D_single = output_3D

            if split == 'train':
                pred_out = output_3D_VTE

            elif split == 'test':
                pred_out = output_3D_single

            input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints ,2)

            if opt.refine:
                pred_uv = input_2D
                uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
                xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
                xyz[:, :, 0, :] = 0
                post_out = model_refine(output_3D_single, xyz)
                loss = mpjpe_cal(post_out, out_target_single)
            else:
                loss = mpjpe_cal(pred_out, out_target) + mpjpe_cal(output_3D_single, out_target_single)

        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not opt.MAE:

                if opt.refine:
                    post_out[:,:,0,:] = 0
                    joint_error = mpjpe_cal(post_out, out_target_single).item()
                else:
                    pred_out[:,:,0,:] = 0
                    joint_error = mpjpe_cal(pred_out, out_target).item()

                error_sum.update(joint_error*N, N)

        elif split == 'test':
            if opt.MAE:
                action_error_sum_MAE = test_calculation(output_2D, torch.cat((input_2D[:, ~mask], input_2D[:, mask]), dim=1), action, action_error_sum_MAE, opt.dataset,
                                                    subject,MAE=opt.MAE)
            else:
                pred_out[:, :, 0, :] = 0
                action_error_sum = test_calculation(pred_out, out_target, action, action_error_sum, opt.dataset, subject)

                if opt.refine:
                    post_out[:, :, 0, :] = 0
                    action_error_sum_post_out = test_calculation(post_out, out_target, action, action_error_sum_post_out, opt.dataset, subject)

    if split == 'train':
        if opt.MAE:
            return loss_all['loss'].avg
        else:
            return loss_all['loss'].avg, error_sum.avg*1000
    elif split == 'test':
        if opt.MAE:
            p1, p2 = print_error(opt.dataset, action_error_sum_MAE, opt.train)
            return p1, p2, loss_all['loss'].avg
        if opt.refine:
            p1, p2 = print_error(opt.dataset, action_error_sum_post_out, opt.train)
        else:
            p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)

        return p1, p2

def input_augmentation_MAE(input_2D, model_trans, joints_left, joints_right, mask, spatial_mask=None):
    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)

    output_2D_flip = model_trans(input_2D_flip, mask, spatial_mask)

    output_2D_flip[:, 0] *= -1

    output_2D_flip[:, :, :, joints_left + joints_right] = output_2D_flip[:, :, :, joints_right + joints_left]

    output_2D_non_flip = model_trans(input_2D_non_flip, mask, spatial_mask)

    output_2D = (output_2D_non_flip + output_2D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_2D

def input_augmentation(input_2D, model_trans, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape 

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)   
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) 

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

if __name__ == '__main__':
    opt.manualSeed = 1

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if opt.train == 1:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
            
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path, MAE=opt.MAE, tds=opt.t_downsample)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    if opt.test:
        test_data = Fusion(opt=opt, train=False,dataset=dataset, root_path =root_path, MAE=opt.MAE, tds=opt.t_downsample)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    opt.out_joints = dataset.skeleton().num_joints()

    print(torch.cuda.is_available())

    model_test=Model(opt)
    dsize = (1, 2, 243, 17, 1)
    inputs = torch.randn(dsize)
    total_ops, total_params = profile(model_test, (inputs,), verbose=False)
    macs, params = clever_format([total_ops, total_params], "%.3f")
    print('MACs:', macs)
    print('Paras:', params)

    model = {}
    model['trans'] = nn.DataParallel(Model(opt)).cuda()
    model['refine'] = nn.DataParallel(refine(opt)).cuda()
    model['MAE'] = nn.DataParallel(Model_MAE(opt)).cuda()

    model_params = 0
    for parameter in model['trans'].parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)


    if opt.MAE_reload == 1:
        model_dict = model['trans'].state_dict()

        MAE_path = opt.previous_dir

        pre_dict = torch.load(MAE_path)

        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}

        model_dict.update(state_dict)
        model['trans'].load_state_dict(model_dict)

        # cnt = 0
        # log_path = os.path.join(opt.checkpoint, 'pretrain.txt')
        # log_path_cur = os.path.join(opt.checkpoint, 'network.txt')
        # f1 = open(log_path, mode='a')
        # f2 = open(log_path_cur, mode='a')
        # for k, v in pre_dict.items():
        #     f1.write('%d\n' % cnt)
        #     f1.write(k+'\n')
        #     cnt+=1
        # f1.close()
        # cnt = 0
        # for k in model_dict.keys():
        #     f2.write('%d\n' % cnt)
        #     f2.write(k+'\n')
        #     cnt+=1
        # f2.close()

    model_dict = model['trans'].state_dict()
    if opt.reload == 1:

        no_refine_path = opt.previous_dir

        pre_dict = torch.load(no_refine_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['trans'].load_state_dict(model_dict)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload == 1:

        refine_path = opt.previous_refine_name

        pre_dict_refine = torch.load(refine_path)
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine[name]
        model['refine'].load_state_dict(refine_dict)

    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    for epoch in range(1, opt.nepoch):
        if opt.train == 1:
            if not opt.MAE:
                loss, mpjpe = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
            else:
                loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
        if opt.test == 1:
            if not opt.MAE:
                p1, p2 = val(opt, actions, test_dataloader, model)
            else:
                p1, p2, loss_test = val(opt, actions, test_dataloader, model)
            data_threshold = p1

            if opt.train and data_threshold < opt.previous_best_threshold:
                if opt.MAE:
                    opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold,
                                                   model['MAE'], 'MAE')

                else:
                    opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model['trans'], 'no_refine')

                    if opt.refine:
                        opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                              data_threshold, model['refine'], 'refine')
                opt.previous_best_threshold = data_threshold

            if opt.train == 0:
                print('p1: %.2f, p2: %.2f' % (p1, p2))
                break
            else:
                if opt.MAE:
                    logging.info('epoch: %d, lr: %.7f, loss: %.4f, loss_test: %.4f, p1: %.2f, p2: %.2f' % (
                    epoch, lr, loss, loss_test, p1, p2))
                    print('e: %d, lr: %.7f, loss: %.4f, loss_test: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, loss_test, p1, p2))
                else:
                    logging.info('epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, mpjpe, p1, p2))
                    print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, mpjpe, p1, p2))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay








