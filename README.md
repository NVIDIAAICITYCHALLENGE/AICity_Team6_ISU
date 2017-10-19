# Implementation of multi-class object detection using RDFCN for NVIDIA AICity Challenge 2017 and AVSS2017 UA_DETRAC Challenge

This repository contains the source codes of RDFCN implementation for the detection tasks of NVIDIA AICity Challenge and AVSS2017 UA_DETRAC Challenge. This source code is built upon the [original Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets) by rearchers from Microsoft Research Asia. The main contribution of this repo to the original repo includes:

* Add two more data readers: one for NVIDIA AICity Challenge 2017; one for AVSS2017 UA_DETRAC Challenge.

* Implement a transfer learning

* Add data inference function to output detection .txt files, one per image.

The codes have been tested on: 

* Ubuntu 16.04, NVIDIA TITIAN X GPU, Python 2.7 (both training and inference). 

* NVIDIA Jetson TX2, Python 2.7 (inference only).

[//]: # (Image References)

[stevens_cr_winchester_sample1]: ./images/AICITY/stevens_cr_winchester_sample1.jpeg "stevens_cr_winchester_sample1"
[stevens_cr_winchester_sample2]: ./images/AICITY/stevens_cr_winchester_sample2.jpeg "stevens_cr_winchester_sample2"
[walsh_santomas_sample1]: ./images/AICITY/walsh_santomas_sample1.jpeg "walsh_santomas_sample1"
[walsh_santomas_sample2]: ./images/AICITY/walsh_santomas_sample2.jpeg "walsh_santomas_sample2"
[480]: ./images/AICITY/480.png "480"
[540]: ./images/AICITY/540.png "540"
[1080]: ./images/AICITY/1080.png "1080"
[mAP_AICity]: ./images/AICITY/mAP_AICity.png "mAP_AICity"

[sample1]: ./images/UADETRAC/sample1.jpg "sample1"
[sample2]: ./images/UADETRAC/sample2.jpg "sample2"
[sample3]: ./images/UADETRAC/sample3.jpg "sample3"
[sample4]: ./images/UADETRAC/sample4.jpg "sample4"
[sample5]: ./images/UADETRAC/sample5.jpg "sample5"
[sample6]: ./images/UADETRAC/sample6.jpg "sample6"
[sample7]: ./images/UADETRAC/sample7.jpg "sample7"
[sample8]: ./images/UADETRAC/sample8.jpg "sample8"
[sample9]: ./images/UADETRAC/sample9.jpg "sample9"
[sample10]: ./images/UADETRAC/sample10.jpg "sample10"
[sample11]: ./images/UADETRAC/sample11.jpg "sample11"
[sample12]: ./images/UADETRAC/sample12.jpg "sample12"
[sample13]: ./images/UADETRAC/sample13.jpg "sample13"
[Precision_Recall_curve]: ./images/UADETRAC/Precision_Recall_curve.png "Precision_Recall_curve"

## Introduction

### NVIDIA AICity Challenge 2017

The track one of AI City Challenge by IEEE Smart World and NVIDIA required each paricipating team to detect 14-class objects from urban intersection surveillance cameras using deep learning methods.

Detailed information of NVIDIA AICity Challenge 2017 can be found [here](http://smart-city-conference.com/AICityChallenge/).

### AVSS2017 UA_DETRAC Challenge

The detection task of AVSS2017 Challenge required each paricipating team to detect vehicles from UA-DETRAC, a real-world multi-object detection and multi-object tracking benchmark.

Detailed information of AVSS2017 Challenge can be found [here](https://iwt4s.wordpress.com/challenge/).

UA-DETRAC dataset consists of 10 hours of videos captured with a Cannon EOS 550D camera at 24 different locations at Beijing and Tianjin in China. 

Detailed information of UA-DETRAC can be found [here](http://detrac-db.rit.albany.edu/).

### RDFCN

RDFCN stands for Region-based Deformable Fully Convolutional Networks. RDFCN consitas of two major components:

* R-FCN (Region-based Fully Convolutional Networks). R-FCN is initially described in a [NIPS 2016 paper](https://arxiv.org/abs/1605.06409). The Key idea of R-FCN is increasing model inference speed by using more shared convolution.

* Deformable ConvNets (Deformable Convolutional Networks). Deformable ConvNets is initially described in an [arxiv tech report](https://arxiv.org/abs/1703.06211). The key idea of Deformable ConvNets is increasing classification accuracy by using adaptive-shaped convolutional filter.

## Installation

1. Carefully follow the instructions of the offical implementation for Deformable Convolutional Networks based on MXNet [here](https://github.com/msracver/Deformable-ConvNets). 

At the end of this step: 

* The Deformable ConvNets repo should have been downloaded;

* MXNet should have been downloaded and properly compiled;

* The R-FCN demo should be able to run by `python ./rfcn/demo.py`.

Note: 

For installing MXnet on NVIDIA Jetson TX2, if the above [installation instruction](https://github.com/msracver/Deformable-ConvNets) doesn't work, you can try following these steps (The MXnet on Jetson TX2 installation has been verified by [Koray](https://www.linkedin.com/in/koray-ozcan-b003903b/)):

a. MxNet is only compatible with opencv-python >= 3.2.0.
Please first upgrade the OpenCV libraries for Jetson platform as described in [here](http://www.jetsonhacks.com/2017/04/05/build-opencv-nvidia-jetson-tx2/)

b. Then install MXnet libraries for Jetson TX2 as described [here](http://mxnet.io/get_started/install.html) (Devices - NVIDIA Jetson TX2).  

2. clone this repo into the root/parent directory of Deformable-ConvNets:

`cd path/to/Deformable-ConvNets/..`

`git clone https://github.com/wkelongws/RDFCN_UADETRAC_AICITY`

3. implement the contribution codes in this repo to the Deformable-ConvNets folder:

`cd RDFCN_UADETRAC_AICITY` 

`python setup.py`

## Test the submitted model on aic1080

1. Finish the Installation

2. Download the model weights from [here](https://1drv.ms/u/s!AmGvrNE6VIsKjUevX-7tNAhS_5AF). Put the downloaded file "rfcn_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal-0004.params" in the root folder of this repo "RDFCN_UADETRAC_AICITY"

3. Install the provided test images from aic1080 and the downloaded weights (create proper paths and move files accordingly)

`cd RDFCN_UADETRAC_AICITY`

`./test_data_install.sh`

4. Modify the inference code (optional):

* Comment line 242 in Deformable-ConvNets/rfcn/functions/inference_rcnn.py if you don't want to see labeled images on run time. By commenting line 242, the program will only output .txt detection files to "Deformable-ConvNets/data/output".

* Change the threshold at line 234 in Deformable-ConvNets/rfcn/functions/inference_rcnn.py. The current theshold is set at 0.2.  

5. Test the model by running:

`cd Deformable-ConvNets`

`sh demo.sh`

or alternatively:

`cd Deformable-ConvNets`

`python experiments/rfcn/rfcn_Inference_Shuo_AICity.py --cfg experiments/rfcn/cfgs/resnet_v1_101_voc0712_rfcn_dcn_Shuo_AICityVOC1080_FreezeCOCO_rpnOnly_all_withsignal.yaml`

## General Usage

1. Download data from UADETRAC and AICity into `data/` folder

2. Create configuration file

3. `cd path/to/Deformable-ConvNets`

4. Model training and Inference

* To train model from scratch (for both AICITY and UADETRAC), run:

`python experiments/rfcn/rfcn_end2end_train_Shuo_UADETRAC.py --cfg path/to/your/configuration/file`

Note: For AICITY, based on our experience "training from scratch" doesn't yield good model. Please refer to "Transfer Learning" section for other solution.

* To detect vechiles from the test dataset, run:

`python experiments/rfcn/rfcn_Inference_Shuo_UADETRAC.py --cfg path/to/your/configuration/file`

`python experiments/rfcn/rfcn_Inference_Shuo_AICity.py --cfg path/to/your/configuration/file`

Two sample configuration files, one for UADETRAC and one for AICity, have been added to `experiments/rfcn/cfgs/`

The weights of submitted model (and the corresponding configuration file) on aic1080 can be downloaded [here](https://1drv.ms/u/s!AmGvrNE6VIsKjUevX-7tNAhS_5AF).

The weights of submitted model (and the corresponding configuration file) on aic540 can be downloaded [here](https://1drv.ms/u/s!AmGvrNE6VIsKjUmeEitETomFbCXQ).

The weights of submitted model (and the corresponding configuration file) on aic480 can be downloaded [here](https://1drv.ms/u/s!AmGvrNE6VIsKjUiF1J2qP4TeNsRc).

The weights of submitted model (and the corresponding configuration file) on UADETRAC can be downloaded [here](https://1drv.ms/u/s!AmGvrNE6VIsKjVK3j5kV6BfDghAU).

## Transfer Learning

One major contribution of this repo is adding a transfer learning module to the original implementation.

To train model on AICity using transfer learning, run:

python experiments/rfcn/rfcn_transfer_learning_train_Shuo_AICity.py --cfg path/to/your/configuration/file

There are multiple ways of transfer learning in this RFCN structure:

1. Initialize using pretrained weights then fine-tune all the weight by training on AICITY data.

2. Initialize using pretrained weights, freeze the weights in backbone resnet-101 and RPN, then fine-tune the location-sensitive score maps feature extractor and the output layer by training on AICITY data.

3. Initialize using pretrained weights, freeze all the weights except the outlayer, then only fine-tune the outlayer by training on AICITY data.

Those three strategies have been implemented in "transfer_learning_end2end_freezeNo.py", "transfer_learning_end2end_freezeRPN.py", "transfer_learning_end2end_freezeBoth.py", respectively. To switch between different strategies, replace the codes in "transfer_learning_end2end.py" (placed in rfcn/ after running setup.py) with the one you want. You can choose the initialization weights by changing the codes starting at line 103 in "transfer_learning_end2end.py"

Some insights of how we eventually selected transfer learning (the second strategy) can be found in the presentation "experiments_before_workshop.pptx" in `presentations/`

## Experimental Results

### NVIDIA AICity Challenge 2017

Sample detections: (Annotations on the left with red boxes, detections on the right with green boxes, thresholded by condidence 0.1)

![alt text][stevens_cr_winchester_sample1]

![alt text][stevens_cr_winchester_sample2]

![alt text][walsh_santomas_sample1]

![alt text][walsh_santomas_sample2]

Precision-Recall cuve on validation dataset (aic1080):

![alt text][mAP_AICity]

AICity submission results (ranked 2nd place in all three dataset by 3:08PM Aug 9th, 2017)

The [detailed results](http://smart-city-conference.com/AICityChallenge/results.html) have been pusblished on NVIDIA AICity challenge website. 

* aic480

![alt text][480]

* aic540

![alt text][540]

* aic1080

![alt text][1080]

A preprocessed video demonstrating the detection on aic1080 can be found [here](https://youtu.be/F-lWyJ5Trk4). 0.2 is used as the detection confidence threshold in the demonstration video.

The workshop presentation can be found in `presentations/`

### AVSS2017 UA_DETRAC Challenge
​​
Sample detections: (detection with green boxes, thresholded by condidence 0.5)

![alt text][sample5]

![alt text][sample6]

![alt text][sample7]

![alt text][sample9]

![alt text][sample10]

![alt text][sample11]

![alt text][sample12]

![alt text][sample13]

Precision-Recall cuve on training dataset:

![alt text][Precision_Recall_curve]

## Disclaimer

This repo is still partially under contruction. The path of certain files may need correction. For any questions you can contact [Shuo Wang](https://github.com/wkelongws).

