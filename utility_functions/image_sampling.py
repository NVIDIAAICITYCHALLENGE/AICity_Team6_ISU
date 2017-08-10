# Written by Shuo Wang
# --------------------------------------------------------

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from shutil import copyfile

sample_size =500

dataset_dir = '/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/VOC1080'
if not os.path.exists(os.path.join(dataset_dir,'samples')):
    os.makedirs(os.path.join(dataset_dir,'samples'))

with open(os.path.join(dataset_dir,'ImageSets/Main/all.txt')) as f:
    contents_train = f.readlines()
trainlist = [line.strip() for line in contents_train]

stevens = [name for name in trainlist if name[0:7] == 'stevens']
stevens_num = len(stevens)
stevens_sampleIDs = np.sort(np.random.choice(stevens_num,sample_size,replace=False))
if not os.path.exists(os.path.join(dataset_dir,'samples','stevens')):
    os.makedirs(os.path.join(dataset_dir,'samples','stevens'))
if not os.path.exists(os.path.join(dataset_dir,'samples','stevens','images')):
    os.makedirs(os.path.join(dataset_dir,'samples','stevens','images'))
if not os.path.exists(os.path.join(dataset_dir,'samples','stevens','labels')):
    os.makedirs(os.path.join(dataset_dir,'samples','stevens','labels'))
for ID in stevens_sampleIDs:
    name = stevens[ID]
    src_image = os.path.join(dataset_dir,'JPEGImages',name+'.jpeg')
    dst_image = os.path.join(dataset_dir,'samples','stevens','images',name+'.jpeg')
    copyfile(src_image,dst_image)
    src_label = os.path.join(dataset_dir,'Annotations',name+'.txt')
    dst_label = os.path.join(dataset_dir,'samples','stevens','labels',name+'.txt')
    copyfile(src_label,dst_label)

walsh = [name for name in trainlist if name[0:5] == 'walsh']
walsh_num = len(walsh)
walsh_sampleIDs = np.sort(np.random.choice(walsh_num,sample_size,replace=False))
if not os.path.exists(os.path.join(dataset_dir,'samples','walsh')):
    os.makedirs(os.path.join(dataset_dir,'samples','walsh'))
if not os.path.exists(os.path.join(dataset_dir,'samples','walsh','images')):
    os.makedirs(os.path.join(dataset_dir,'samples','walsh','images'))
if not os.path.exists(os.path.join(dataset_dir,'samples','walsh','labels')):
    os.makedirs(os.path.join(dataset_dir,'samples','walsh','labels'))
for ID in walsh_sampleIDs:
    name = walsh[ID]
    src_image = os.path.join(dataset_dir,'JPEGImages',name+'.jpeg')
    dst_image = os.path.join(dataset_dir,'samples','walsh','images',name+'.jpeg')
    copyfile(src_image,dst_image)
    src_label = os.path.join(dataset_dir,'Annotations',name+'.txt')
    dst_label = os.path.join(dataset_dir,'samples','walsh','labels',name+'.txt')
    copyfile(src_label,dst_label)


with open(os.path.join(dataset_dir,'ImageSets/Main/test.txt')) as f:
    contents_test = f.readlines()
testlist = [line.strip() for line in contents_test]

san_tomas = [name for name in testlist if name[0:9] == 'san_tomas']
san_tomas_num = len(san_tomas)
san_tomas_sampleIDs = np.sort(np.random.choice(san_tomas_num,sample_size,replace=False))
if not os.path.exists(os.path.join(dataset_dir,'samples','san_tomas')):
    os.makedirs(os.path.join(dataset_dir,'samples','san_tomas'))
if not os.path.exists(os.path.join(dataset_dir,'samples','san_tomas','images')):
    os.makedirs(os.path.join(dataset_dir,'samples','san_tomas','images'))
#if not os.path.exists(os.path.join(dataset_dir,'samples','san_tomas','labels')):
#    os.makedirs(os.path.join(dataset_dir,'samples','san_tomas','labels'))
for ID in san_tomas_sampleIDs:
    name = san_tomas[ID]
    src_image = os.path.join(dataset_dir,'JPEGImages',name+'.jpeg')
    dst_image = os.path.join(dataset_dir,'samples','san_tomas','images',name+'.jpeg')
    copyfile(src_image,dst_image)
    #src_label = os.path.join(dataset_dir,'Annotations',name+'.txt')
    #dst_label = os.path.join(dataset_dir,'samples','san_tomas','labels',name+'.txt')
    #copyfile(src_label,dst_label)



