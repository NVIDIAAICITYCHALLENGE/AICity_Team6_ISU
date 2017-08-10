# Written by Shuo Wang
# --------------------------------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

with open('Sequences/testlist_det_experienced.txt') as f:
    contents_test = f.readlines()
testlist = [line.strip() for line in contents_test]

for testSeq in testlist:
    folderIn = os.path.join('TrainSequences',testSeq)
    folderOut = os.path.join('AnnotatedImages_test',testSeq)
    if not os.path.exists(folderOut):
	os.makedirs(folderOut)
    images = os.listdir(folderIn)

    with open(os.path.join('Outputs_test',testSeq+'_Det_DFCN.txt')) as f:
	contents_boxes = f.readlines()
    boxes_all = [line.strip() for line in contents_boxes]

    for image in images:
	print 'processing image:',testSeq,image
	images_name = os.path.join(folderIn,image)
	img = cv2.imread(images_name)
	frame_id = int(image[3:8])

	boxes = [box for box in boxes_all if (int(box.split(',')[0])==frame_id) & (float(box.split(',')[-1])>0.5)]

	for box in boxes:
	    x1 = int(round(float(box.split(',')[2])))
	    y1 = int(round(float(box.split(',')[3])))
	    width = int(round(float(box.split(',')[4])))
	    height = int(round(float(box.split(',')[5])))
	    x2 = x1 + width
	    y2 = y1 + height
	    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

	cv2.imwrite(os.path.join(folderOut,image),img)
	    
