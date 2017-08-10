dfg


inference_rcnn.py in Deformable-ConvNets/rfcn/function/

Inference_Shuo_AICity.py in Deformable-ConvNets/rfcn/

rfcn_Inference_Shuo_AICity.py in Deformable-ConvNets/experiments/rfcn/

resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal.yaml in Deformable-ConvNets/experiments/rfcn//cfgs/

AICity.py in Deformable-ConvNets/lib/dataset/

__init__.py in Deformable-ConvNets/lib/dataset/ (replace the original one)

rfcn_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal-0003.params in Deformable-ConvNets/output/rfcn_dcn_Shuo_AICity/resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal/1080_all/ (create this folder)

[your test images.jpeg] in Deformable-ConvNets/data/data_Shuo/VOC1080/JPEGImages/

test.txt in /Deformable-ConvNets/data/data_Shuo/VOC1080/ImageSets/ and replace the image names in it with your actual test image names​
model_code.zip​

Then cd to Deformable-ConvNets and run:

python experiments/rfcn/rfcn_Inference_Shuo_AICity.py --cfg experiments/rfcn//cfgs/resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal.yaml

You should see test images with detected boundary boxes.
​​
