# coding: utf-8
# author: wx
# for help

import os
import sys
sys.path.append('\\'.join(os.path.split(sys.path[0])))
 
import matplotlib
matplotlib.use('TkAgg')
 
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

import parameter as para


def showNii(path):
    img = img = nib.load(path)
    print(path)
    width, height, queue=img.dataobj.shape
    OrthoSlicer3D(img.dataobj).show()
    # num = 1
    # for i in range(0,queue,10):
    #     img_arr = img.dataobj[:,:,i]
    #     plt.subplot(5,4,num)
    #     plt.imshow(img_arr,cmap='gray')
    #     num +=1
    # plt.show()


def compare(num):
    volume = para.test_ct_path + 'volume-' + str(num) + '.nii'
    gold = para.test_seg_path + 'segmentation-' + str(num) + '.nii'
    pred = para.pred_path + 'pred-' + str(num) + '.nii'



if __name__ == '__main__':
    if len(sys.argv)==2:
        showNii(sys.argv[1])
    else:
        showNii('E:\\deep_learning\\pytorch\\lits2017\\test_pred\\pred-125.nii')
        # showNii('E:\\deep_learning\\pytorch\\lits2017\\train\\seg\\segmentation-130.nii.gz')
        # showNii('E:\\deep_learning\\pytorch\\lits2017\\test_pred\\pred-28.nii')
        # compare(28)
        # print()