from shutil import copyfile
import os

dst_dir = '../Deformable-ConvNets'


#copy inference_rcnn.py in Deformable-ConvNets/rfcn/function/
copyfile('inference_rcnn.py',os.path.join(dst_dir,'rfcn/functions','inference_rcnn.py'))

#copy Inference_Shuo_AICity.py in Deformable-ConvNets/rfcn/
copyfile('Inference_Shuo_AICity.py',os.path.join(dst_dir,'rfcn','Inference_Shuo_AICity.py'))

#copy Inference_Shuo_UADETRAC.py in Deformable-ConvNets/rfcn/
copyfile('Inference_Shuo_UADETRAC.py',os.path.join(dst_dir,'rfcn','Inference_Shuo_UADETRAC.py'))

#copy transfer_learning_end2end.py in Deformable-ConvNets/rfcn/
copyfile('transfer_learning_end2end.py',os.path.join(dst_dir,'rfcn','transfer_learning_end2end.py'))



#copy rfcn_Inference_Shuo_AICity.py in Deformable-ConvNets/experiments/rfcn/
copyfile('rfcn_Inference_Shuo_AICity.py',os.path.join(dst_dir,'experiments/rfcn','rfcn_Inference_Shuo_AICity.py'))

#copy rfcn_Inference_Shuo_UADETRAC.py in Deformable-ConvNets/experiments/rfcn/
copyfile('rfcn_Inference_Shuo_UADETRAC.py',os.path.join(dst_dir,'experiments/rfcn','rfcn_Inference_Shuo_UADETRAC.py'))

#copy rfcn_end2end_train_inference_Shuo_UADETRAC.py in Deformable-ConvNets/experiments/rfcn/
copyfile('rfcn_end2end_train_inference_Shuo_UADETRAC.py',os.path.join(dst_dir,'experiments/rfcn','rfcn_end2end_train_inference_Shuo_UADETRAC.py'))

#copy rfcn_transfer_learning_end2end_train_test_Shuo_AICity.py in Deformable-ConvNets/experiments/rfcn/
copyfile('rfcn_transfer_learning_end2end_train_test_Shuo_AICity.py',os.path.join(dst_dir,'experiments/rfcn','rfcn_transfer_learning_end2end_train_test_Shuo_AICity.py'))


#copy resnet_v1_101_rfcn_dcn.py in Deformable-ConvNets/rfcn/symbols/
copyfile('resnet_v1_101_rfcn_dcn.py',os.path.join(dst_dir,'rfcn/symbols','resnet_v1_101_rfcn_dcn.py'))




#copy resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal.yaml in Deformable-ConvNets/experiments/rfcn/cfgs/
copyfile('resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal.yaml',os.path.join(dst_dir,'experiments/rfcn/cfgs','resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal.yaml'))

#copy AICity.py in Deformable-ConvNets/lib/dataset/
copyfile('AICity.py',os.path.join(dst_dir,'lib/dataset','AICity.py'))

#copy UA_DETRAC.py in Deformable-ConvNets/lib/dataset/
copyfile('UA_DETRAC.py',os.path.join(dst_dir,'lib/dataset','UA_DETRAC.py'))

#overwrite __init__.py in Deformable-ConvNets/lib/dataset/ (replace the original one)
copyfile('__init__.py',os.path.join(dst_dir,'lib/dataset','__init__.py'))

#copy rfcn_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal-0003.params in Deformable-ConvNets/output/rfcn_dcn_Shuo_AICity/resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal/1080_all/ (create this folder)