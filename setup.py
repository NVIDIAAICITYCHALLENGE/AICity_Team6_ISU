from shutil import copyfile
import os

# path to the Deformable-ConvNets
dst_dir = '../Deformable-ConvNets'


#install codes of inference pipeline to output detection .txt files one per image
copyfile('inference_rcnn.py',os.path.join(dst_dir,'rfcn/function','inference_rcnn.py'))
copyfile('Inference_Shuo_AICity.py',os.path.join(dst_dir,'rfcn','Inference_Shuo_AICity.py'))
copyfile('Inference_Shuo_UADETRAC.py',os.path.join(dst_dir,'rfcn','Inference_Shuo_UADETRAC.py'))
copyfile('rfcn_Inference_Shuo_AICity.py',os.path.join(dst_dir,'experiments/rfcn','rfcn_Inference_Shuo_AICity.py'))
copyfile('rfcn_Inference_Shuo_UADETRAC.py',os.path.join(dst_dir,'experiments/rfcn','rfcn_Inference_Shuo_UADETRAC.py'))

#install codes of transfer learning pipeline
copyfile('transfer_learning_end2end.py',os.path.join(dst_dir,'rfcn','transfer_learning_end2end.py'))
copyfile('rfcn_end2end_train_Shuo_UADETRAC.py',os.path.join(dst_dir,'experiments/rfcn','rfcn_end2end_train_Shuo_UADETRAC.py'))
copyfile('rfcn_transfer_learning_train_Shuo_AICity.py',os.path.join(dst_dir,'experiments/rfcn','rfcn_transfer_learning_train_Shuo_AICity.py'))
copyfile('arg_params.txt',os.path.join(dst_dir,'rfcn/symbols','arg_params.txt'))
copyfile('resnet_v1_101_rfcn_dcn.py',os.path.join(dst_dir,'rfcn/symbols','resnet_v1_101_rfcn_dcn.py'))

#install AICity and UADETRAC data reader
copyfile('AICity.py',os.path.join(dst_dir,'lib/dataset','AICity.py'))
copyfile('UA_DETRAC.py',os.path.join(dst_dir,'lib/dataset','UA_DETRAC.py'))
copyfile('__init__.py',os.path.join(dst_dir,'lib/dataset','__init__.py'))

#install sample experiments
copyfile('resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal.yaml',os.path.join(dst_dir,'experiments/rfcn/cfgs','resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal.yaml'))
copyfile('RDFCN_ISU_Shuo_UADETRAC_end2end.yaml',os.path.join(dst_dir,'experiments/rfcn/cfgs','RDFCN_ISU_Shuo_UADETRAC_end2end.yaml'))

#install other changes
copyfile('tester_Shuo.py',os.path.join(dst_dir,'rfcn/core','tester_Shuo.py'))
copyfile('loader.py',os.path.join(dst_dir,'rfcn/core','loader.py'))
