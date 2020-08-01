# coding: utf-8
# author: wx
# for train dataset

import os
# import sys
# print(sys.path)
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from visdom import Visdom

from datasets import Dataset

from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss

from net.ResUNet import net

import parameter as para

# print("import ok")

if __name__ == '__main__':
    # 清除缓存
    # print(torch.cuda.device_count())
    torch.cuda.empty_cache()
    print("Train dataset: cache clean.")

    # 设置visdom
    viz = Visdom(port=8097)
    step_list = [0]
    win = viz.line(X=np.array([0]), Y=np.array([1.0]), opts=dict(title='loss'))
    print("Train dataset: visdom set.")

    # 设置显卡相关
    os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
    cudnn.benchmark = para.cudnn_benchmark
    print("Train dataset: cuda and cudnn done.")

    # 定义网络
    net = torch.nn.DataParallel(net).cuda()
    net.train()
    print("Train dataset: net done.")

    # 定义dataset
    train_ds = Dataset(os.path.join(para.training_set_path, 'ct'),  os.path.join(para.training_set_path, 'seg'))
    print("Train datset: dataset done, at", os.path.join(para.training_set_path, 'ct'),  os.path.join(para.training_set_path, 'seg'))

    # 定义数据加载
    train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)

    # 挑选损失函数
    loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
    loss_func = loss_func_list[5]
    print("Train dataset: pick", loss_func, "as loss func")

    # 定义优化器
    opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

    # 学习率衰减
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

    # 深度监督衰减系数
    alpha = para.alpha
    
    if not os.path.exists(para.save_module_path):
        os.mkdir(para.save_module_path)

    epochs = 1
    # 恢复训练
    if para.recovery:
        module = os.listdir(para.save_module_path)[-2]
        print("Train dataset: recover module", module)
        epochs = int(module.split('-')[0][3:])
        net.load_state_dict(torch.load(para.save_module_path+module))
        
    # 训练网络
    print("Train dataset: start train.")
    start = time()
    for epoch in range(epochs, para.Epoch+1):
        torch.cuda.empty_cache()
        mean_loss = []

        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.cuda()
            seg = seg.cuda()

            outputs = net(ct)

            loss1 = loss_func(outputs[0], seg)
            loss2 = loss_func(outputs[1], seg)
            loss3 = loss_func(outputs[2], seg)
            loss4 = loss_func(outputs[3], seg)

            loss = (loss1+loss2+loss3)*alpha+loss4

            mean_loss.append(loss4.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%5 is 1:
                step_list.append(step_list[-1]+1)
                viz.line(X=np.array([step_list[-1]]), Y=np.array([loss4.item()]), win=win, update='append')
                print("Train dataset:\n\tepoch:{}, step:{}, loss:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}. time:{:.3f}min"
                    .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time()-start)/60))

        mean_loss = sum(mean_loss) / len(mean_loss)
        if epoch%50 is 0:
            # 网络模型命名方式: epoch轮数 + minibatch loss+ 本轮epoch平均loss
            torch.save(net.state_dict(), para.save_module_path+'net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))
            print("Train dataset: module save to "+para.save_module_path+"/net{}-{:.3f}-{:.3f}.pth".format(epoch, loss, mean_loss))

        if epoch%40 is 0:
            alpha *= 0.8
            print("Train dataset: alpha decay to", alpha)
            
        lr_decay.step()

