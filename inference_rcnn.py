# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import argparse
import pprint
import logging
import time
import os
import mxnet as mx
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester_Shuo import Predictor, pred_eval, im_detect
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def inference_rcnn_UADETRAC(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        #roidb = imdb.gt_roidb_Shuo()
	roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rfcn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb_Shuo()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    print 'len(roidb):',len(roidb)
    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)
    print 'inferring: ',prefix,' epoch: ',epoch
    
    """# write parameters to file
    print 'type(arg_params):',type(arg_params)
    print 'type(aux_params):',type(aux_params)
    thefile1 = open('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/UADETRAC/arg_params.txt','w')
    thefile2 = open('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/UADETRAC/aux_params.txt','w')
    for item_arg in arg_params.items():
	thefile1.write(item_arg[0] + str(type(item_arg[1])) + str(item_arg[1].shape)+'\n')
    for item_aux in aux_params.items():
	thefile2.write(item_aux[0] + str(type(item_aux[1])) + str(item_aux[1].shape)+'\n')
    """
 
    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    nms = gpu_nms_wrapper(cfg.TEST.NMS, 0)
    # start detection
    # pred_eval(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)
    print 'test_data.size',test_data.size
    print 'test_data:',test_data
    print 'data_names:',data_names
    print 'test_data.provide_data:',test_data.provide_data
    print 'test_data.provide_label:',test_data.provide_label
    nnn = 0
    classes = ['__background','vehicle']
    #num_classes = 10
    #classes = ['__DontCare__','Car','Suv','SmallTruck','MediumTruck','LargeTruck','Pedestrian','Bus','Van','GroupofPeople']
    for im_info, data_batch in test_data:
	print nnn
	#print 'roidb[nnn]:',roidb[nnn]['image']
	image_name = roidb[nnn]['image']
	tic()
        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_detect(predictor, data_batch, data_names, scales, cfg)
	boxes = boxes_all[0].astype('f')
        scores = scores_all[0].astype('f')
	dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 4:8] if cfg.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            #cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
            dets_nms.append(cls_dets)
        print 'testing {} {:.4f}s'.format(image_name, toc())
        # visualize
        im = cv2.imread(image_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

	#print 'cls_dets:',cls_dets
        #show_boxes(im, dets_nms, classes, 1)
	nnn = nnn + 1
	image_name_length = len(image_name.split('/'))
	sequence_name = image_name.split('/')[image_name_length-2]
	output_file = os.path.join('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/UADETRAC', 'Outputs', sequence_name + '_Det_DFCN.txt')
	frame_id = int(image_name.split('/')[image_name_length-1][3:8])
	
        thefile = open(output_file,'a')
	
 	det_id = 0
	for x_small,y_small,x_large,y_large,prob in dets_nms[0]:
	    det_id += 1
	    thefile.write(str(frame_id)+','+str(det_id)+','+str(x_small)+','+str(y_small)+','+str(max(x_large-x_small,0.001))+','+str(max(y_large-y_small,0.001))+','+str(prob)+'\n')
	    #cv2.rectangle(im,(x_small,y_small),(x_large,y_large),(0,255,0),2)
        #plt.imshow(im)
	#plt.show()

def inference_rcnn_AICity(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb_Shuo()
	#roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rfcn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb_Shuo()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    print 'len(roidb):',len(roidb)
    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)
    print 'inferring: ',prefix,' epoch: ',epoch
    
    """# write parameters to file
    print 'type(arg_params):',type(arg_params)
    print 'type(aux_params):',type(aux_params)
    thefile1 = open('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/UADETRAC/arg_params.txt','w')
    thefile2 = open('/raid10/home_ext/Deformable-ConvNets/data/data_Shuo/UADETRAC/aux_params.txt','w')
    for item_arg in arg_params.items():
	thefile1.write(item_arg[0] + str(type(item_arg[1])) + str(item_arg[1].shape)+'\n')
    for item_aux in aux_params.items():
	thefile2.write(item_aux[0] + str(type(item_aux[1])) + str(item_aux[1].shape)+'\n')
    """
 
    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    nms = gpu_nms_wrapper(cfg.TEST.NMS, 0)
    # start detection
    # pred_eval(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)
    print 'test_data.size',test_data.size
    print 'test_data:',test_data
    print 'data_names:',data_names
    print 'test_data.provide_data:',test_data.provide_data
    print 'test_data.provide_label:',test_data.provide_label
    nnn = 0
    #classes = ['__background','vehicle']
    classes = ['Car','SUV','SmallTruck','MediumTruck','LargeTruck','Pedestrian','Bus','Van','GroupOfPeople','Bicycle', 'Motorcycle','TrafficSignal-Green', 'TrafficSignal-Yellow', 'TrafficSignal-Red']
    #,'Pedestrian', 'GroupOfPeople','Bicycle', 'Motorcycle','TrafficSignal-Green', 'TrafficSignal-Yellow', 'TrafficSignal-Red'
    for im_info, data_batch in test_data:
	print nnn
	#print 'roidb[nnn]:',roidb[nnn]['image']
	image_name = roidb[nnn]['image']
	tic()
        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_detect(predictor, data_batch, data_names, scales, cfg)
	boxes = boxes_all[0].astype('f')
        scores = scores_all[0].astype('f')
	dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 4:8] if cfg.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > 0.001, :]
            dets_nms.append(cls_dets)
        print 'testing {} {:.4f}s'.format(image_name, toc())
        # visualize
        im = cv2.imread(image_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	
	#print 'cls_dets:',cls_dets
        #show_boxes(im, dets_nms, classes, 1)
	nnn = nnn + 1
	image_name_length = len(image_name.split('/'))
	imagefile_name = image_name.split('/')[image_name_length-1]
	image_name_lean = imagefile_name.split('.')[0]
    
    if not os.path.exists(os.path.join('data', 'output')):
        os.makedirs(os.path.join('data', 'output'))
    
	output_file = os.path.join('data', 'output', image_name_lean+ '.txt')
	
        thefile = open(output_file,'a')
	
 	#det_id = 0
	#for x_small,y_small,x_large,y_large,prob in dets_nms[0]:
	    #det_id += 1

	for cls_idx, cls_name in enumerate(classes):
            cls_dets = dets_nms[cls_idx]
            for x_small,y_small,x_large,y_large,prob in cls_dets:
        	thefile.write(cls_name+' '+str(x_small)+' '+str(y_small)+' '+str(max(x_small+0.01,x_large))+' '+str(max(y_small+0.01,y_large))+' '+str(prob)+'\n')


