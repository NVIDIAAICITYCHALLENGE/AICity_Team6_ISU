# Written by Shuo Wang
# --------------------------------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

dataset1 = 'Outputs_self'
dataset2 = 'Outputs_FromUADETRAC'
dataset3 = 'Outputs_FreezeUADETRAC'
dataset4 = 'Outputs_FromCOCO'
dataset5 = 'Outputs_FreezeCOCO'

folderIn = os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/AICity_1080','LabeledImages_'+dataset1)
folderOut =os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/AICity_1080','random_vis') 
im_list = os.listdir(folderIn)

im_num = len(im_list)

while 1:
    im_name = im_list[np.random.randint(im_num)]
    image1 = cv2.imread(os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/AICity_1080','LabeledImages_'+dataset1,im_name))
    image2 = cv2.imread(os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/AICity_1080','LabeledImages_'+dataset2,im_name))
    image3 = cv2.imread(os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/AICity_1080','LabeledImages_'+dataset3,im_name))
    image4 = cv2.imread(os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/AICity_1080','LabeledImages_'+dataset4,im_name))
    image5 = cv2.imread(os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/AICity_1080','LabeledImages_'+dataset5,im_name))
    
    img = np.vstack([image1,image2,image3,image4,image5])
    #plt.imshow(img)
    #plt.show()
    cv2.imwrite(os.path.join(folderOut,'crosscomp_'+im_name),img)


