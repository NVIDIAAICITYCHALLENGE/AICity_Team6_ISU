#!/bin/bash

#mkdir ../Deformable-ConvNets
mkdir ../Deformable-ConvNets/data
mkdir ../Deformable-ConvNets/data/data_Shuo
mkdir ../Deformable-ConvNets/data/data_Shuo/VOC1080
mkdir ../Deformable-ConvNets/data/data_Shuo/VOC1080/JPEGImages
mkdir ../Deformable-ConvNets/data/data_Shuo/VOC1080/ImageSets
mkdir ../Deformable-ConvNets/data/data_Shuo/VOC1080/ImageSets/Main
cp samples_aic1080/* ../Deformable-ConvNets/data/data_Shuo/VOC1080/JPEGImages/
mv ../Deformable-ConvNets/data/data_Shuo/VOC1080/JPEGImages/test.txt ../Deformable-ConvNets/data/data_Shuo/VOC1080/ImageSets/Main/test.txt

mkdir ../Deformable-ConvNets/output
mkdir ../Deformable-ConvNets/output/rfcn_dcn_Shuo_AICity
mkdir ../Deformable-ConvNets/output/rfcn_dcn_Shuo_AICity/resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal
mkdir ../Deformable-ConvNets/output/rfcn_dcn_Shuo_AICity/resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal/1080_all
mv rfcn_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal-0004.params ../Deformable-ConvNets/output/rfcn_dcn_Shuo_AICity/resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal/1080_all/rfcn_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal-0004.params

mv demo.sh ../Deformable-ConvNets/demo.sh