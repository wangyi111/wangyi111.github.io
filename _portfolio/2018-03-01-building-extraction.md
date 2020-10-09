---
title: "Building Extraction"
excerpt: "Building extraction from remote sensing images with deep neural networks."
header:
  teaser: /assets/images/portfolio/building_extraction/top_mosaic_09cm_area1-th.png

gallery:
  - url: /assets/images/portfolio/building_extraction/ortho.jpg
    image_path: assets/images/portfolio/building_extraction/ortho.jpg
    alt: "example_image"
  - url: /assets/images/portfolio/building_extraction/result.png
    image_path: assets/images/portfolio/building_extraction/result.png
    alt: "example_result"
---

In this project, we focus on machine-learned building extraction from optical remote sensing images (RGB). We implemented 4 different methods: simple DNN, fully convolutional network, U-Net and modified U-Net, the last of which giving best segmentation result.

## 01: Dataset

The training and testing data include 33 aerial images of Vaihingen(Germany) area (resolution on a level of 2000*2000) from [ISPRS dataset](http://www2.isprs.org/commissions/comm3/wg4/tests.html).

To make the data fit into the GPU, we firstly did an overlapped cropping to get 256*256 size sub-image. Later on we made this procedure automatic by adding a processing module in the training and validation process which randomly crop the image before importing the network, in which way we don't need to create thousands of sub-images on the disk any more.

## 02: Networks

### 2-1: Simple DNN

The very initial way of doing pixel-wise semantic segmentation instead of image classification was to make the final output 1-D neurons the same amount as the total pixels of input image and then transfer back to 2d-image. We used this way in the very beginning, testing multilayer perceptron and deep ResNet, the latter giving far better result as expected, yet a sawtooth artifact showed up in the final building mask.

### 2-2: Fully convolutional networks

FCN has become a dominant way of doing semantic segmentation tasks, we used which for second trial. FCN consists only of convolutional layers, enabling 2-D mask output by using transposed convolutions and adding previous feature layers to reconstruct global information. We tested a simple FCN with 1/8, 1/16 and 1/32 feature maps, resulting in better accuracy and faster running time compared to simple DNN.

### 2-3: U-Net

The difference between U-Net and FCN is mainly the way they reconstruct global information. While FCN adds feature maps for up-sampling, U-Net concatenates feature maps. This simple idea turns out to be a big improvement as the concatenation gives more useful information. We implemented simple U-Net and got a better result than FCN.

### 2-4: Modified U-Net

In this last network, we modified U-Net to let it go deeper, has more channels, and introduce a resnet structure, as residual module has turned out to be a balm in CNNs. The result didn't let us down either, giving both the best testing and validation accuracy.

{% include gallery caption="Left: Orthophoto  &  Right: Extracted Building mask" %}