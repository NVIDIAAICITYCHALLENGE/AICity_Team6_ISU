# Implementation of multi-class object detection using RDFCN for NVIDIA AICity Challenge 2017 and AVSS2017 UA_DETRAC Challenge

This repository contains the source codes of RDFCN implementation for the detection tasks of NVIDIA AICity Challenge and AVSS2017 UA_DETRAC Challenge. This source code is built upon the [original Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets) by rearchers from Microsoft Research Asia. The main contribution of this repo to the original repo includes:

* Add two more data readers: one for NVIDIA AICity Challenge 2017; one for AVSS2017 UA_DETRAC Challenge.

* Implement a transfer learning

* Add data inference function to output detection .txt files, one per image.

## Introduction

### NVIDIA AICity Challenge 2017

The track one of AI City Challenge by IEEE Smart World and NVIDIA required each paricipating team to detect 14-class objects from urban intersection surveillance cameras using deep learning methods.

Detailed information of NVIDIA AICity Challenge 2017 can be found [here](http://smart-city-conference.com/AICityChallenge/).

### AVSS2017 UA_DETRAC Challenge

The detection task of AVSS2017 Challenge required each paricipating team to detect vehicles from UA-DETRAC, a real-world multi-object detection and multi-object tracking benchmark.

Detailed information of AVSS2017 Challenge can be found [here](https://iwt4s.wordpress.com/challenge/).

UA-DETRAC dataset consists of 10 hours of videos captured with a Cannon EOS 550D camera at 24 different locations at Beijing and Tianjin in China. 

Detailed information of UA-DETRAC can be found [here](http://detrac-db.rit.albany.edu/)

### RDFCN

RDFCN stands for Region-based Deformable Fully Convolutional Networks. RDFCN consitas of two major components:

* R-FCN (Region-based Fully Convolutional Networks). R-FCN is initially described in a [NIPS 2016 paper](https://arxiv.org/abs/1605.06409). The Key idea of R-FCN is increasing model inference speed by using more shared convolution.

* Deformable ConvNets (Deformable Convolutional Networks). Deformable ConvNets is initially described in an [arxiv tech report](https://arxiv.org/abs/1703.06211). The key idea of Deformable ConvNets is increasing classification accuracy by using adaptive-shaped convolutional filter.

## Installation

## Usage

## Experimental Results

### NVIDIA AICity Challenge 2017

### AVSS2017 UA_DETRAC Challenge
​​
