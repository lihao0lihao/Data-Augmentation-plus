from __future__ import absolute_import
 
from torchvision.transforms import *
import argparse
from PIL import Image
import random
import math
import numpy as np
import torch
import cv2
import os
from torchvision import transforms
 
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    probability：执行Random Erasing的概率，默认是0.5；
    sl：最小的擦除面积，这里是相对原图的面积比例，默认是0.02；
    sh：最大的擦除面积，默认是0.4；
    r1：最小的长宽比，默认是0.3，实际长宽比是[0.3,1/0.3]之间的随机值；
    mean：擦除的块中填充的数值，默认是ImageNet的像素归一化均值[0.4914, 0.4822, 0.4465]。
    ————————————————

                                版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                            
原文链接：https://blog.csdn.net/u013685264/article/details/122564323
    -------------------------------------------------------------------------------------
    '''
 
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
 
    def __call__(self, img):
 
        if random.uniform(0, 1) > self.probability:
            return img
 
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
 
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
 
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
 
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
 
        return img
    

    def get_surround(self,img,h,w):
        if w < img.size()[2] and h < img.size()[1]:
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            if img.size()[0] == 3:
                # img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                # img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                # img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                patch=img[:,x1:x1+h,y1:y1+w]
            else:
                patch=img[:,x1:x1+h,y1:y1+w]
            return patch
    def patch(self,img):
        if random.uniform(0, 1) > self.probability:
            return img
 
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
 
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
 
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            patch=self.get_surround(img,h,w)
            # print(patch)
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[:, x1:x1 + h, y1:y1 + w] = patch
                    
                else:
                    img[:, x1:x1 + h, y1:y1 + w] = patch
                return img
    



def parse_args():
    parser=argparse.ArgumentParser(description="parser args for random earse")
    parser.add_argument('--input_dir', type=str, help='origin picture dir')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--mode', type=str,choices=["origin", "patch"], default="origin", help='erase mode')
    args= parser.parse_args()
    return args
if __name__=='__main__':
    args=parse_args()
    mode=args.mode
    dir=args.input_dir
    output_dir=args.output_dir
    pic_dir=os.path.join('./origin_pic',dir)
    files=os.listdir(pic_dir)
    for file in files:
        name=os.path.basename(file)
        if name.split('.')[-1]!='jpg':
            continue
        img_path=os.path.join(pic_dir,name)
        img = cv2.imread(img_path)
        img = transforms.ToTensor()(img)
        re = RandomErasing(probability=1,sl=0.02,sh=0.1,r1=0.2,mean=[0,0,0])
        if mode == 'origin':
            img = re(img)
        elif mode =='patch':
            img = re.patch(img)
        else:
            continue
        # Random Erasing图像写入本地
        img = img.mul(255).byte()
        img = img.numpy().transpose((1, 2, 0))
        root_dir=os.path.dirname(os.path.realpath(__file__))
        out_dir=os.path.join(root_dir,'output',output_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        cv2.imwrite(os.path.join(out_dir,name), img)
# ————————————————

#版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
# 原文链接：https://blog.csdn.net/u013685264/article/details/122564323
    
# cutout和cutmix数据增强方法详见：https://blog.csdn.net/qq_35914625/article/details/108697476