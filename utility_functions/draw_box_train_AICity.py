# Written by Shuo Wang
# --------------------------------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

dataset = 'Outputs_FromUADETRAC'

folderOut = os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/AICity_1080','LabeledImages_'+dataset)
if not os.path.exists(folderOut):
    os.makedirs(folderOut)

with open('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/VOC1080/ImageSets/Main/train.txt') as f:
    contents_train = f.readlines()
trainlist = [line.strip() for line in contents_train]

for image in trainlist:
    """
    folderIn = os.path.join('TrainSequences',trainSeq)
    
    images = os.listdir(folderIn)
    """

    print 'processing image:',image
    images_name = os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/VOC1080/JPEGImages',image+'.jpeg')
    img1 = cv2.imread(images_name)
    img2 = np.copy(img1)

    with open(os.path.join(dataset,image+'.txt')) as f:
	contents_boxes = f.readlines()
    boxes_dt = [line.strip() for line in contents_boxes]


    XMLname = os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/VOC1080/Annotations',image+'.txt')
    tree = ET.parse(XMLname)
    boxes_gt = tree.findall('object')

    for box_gt in boxes_gt:
	bbox = box_gt.find('bndbox')
	cls_gt = box_gt.find('name').text.lower().strip()

	x1 = int(bbox.find('xmin').text) - 1
	y1 = int(bbox.find('ymin').text) - 1
	x2 = int(bbox.find('xmax').text) - 1
	y2 = int(bbox.find('ymax').text) - 1

	cv2.rectangle(img1,(x1,y1),(x2,y2),(0,0,255),3)
	cv2.putText(img1,'{:s}'.format(cls_gt),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)#color='black'
     
    boxes = [box for box in boxes_dt if float(box.split(' ')[-1])>0.1]

    for box in boxes:
	cls_dt = box.split(' ')[0]
	conf = float(box.split(' ')[-1])
	x1 = int(round(float(box.split(' ')[1])))
	y1 = int(round(float(box.split(' ')[2])))
	x2 = int(round(float(box.split(' ')[3])))
	y2 = int(round(float(box.split(' ')[4])))
	cv2.rectangle(img2,(x1,y1),(x2,y2),(0,255,0),3)
	cv2.putText(img2,'{:s} {:.3f}'.format(cls_dt, conf),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)#color='white'

    img = np.hstack([img1,img2])
    cv2.imwrite(os.path.join(folderOut,image+'.jpeg'),img)

	    
