# Written by Shuo Wang
# --------------------------------------------------------

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

list = os.listdir('JPEGImages')

list1 = [img.split('.')[0] for img in list]

thefile = open('all.txt','a')

for image in list1:

    thefile.write(image+'\n')

thefile.close()

