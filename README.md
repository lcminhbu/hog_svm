# INTRODUCTION

#### Data source: The image dataset used in project is from: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset
#### Project: This is an image classification project using SVM and HOG (Histogram of Oriented Gradients)
#### Goals:
- Implement an image classification system using HOG and SVM.
- The system is expected to have a high accuracy (compared to sklearn and skimage), and consume less time than sklearn and skimage.
- The system must have the accuracy higher than CNN which small dataset (we use <=6000 images in our dataset)

# Project structure
- **root/*_detai.ipynb**: Files containing detailed information about the implementation of algorithms from scratch (both sequential and parallel implementations).
- **root/demo_and_comparisons/*.py**: Files used for saving algorithm classes and some functions that is frequently used in our comparions.
- **root/demo_and_comparisons/*.ipynb**: Comparisons of running time and accuracy between HOG + SVM implementations, HOG + SVM and CNN.
- **root/demo_and_comparisons/demo_images**: Images that you can try in our **root/demo_and_comparisons/demo.ipynb** file

# Algorithms
## HOG:

We use HOG to extract image features as it helps SVM work better.

The input of SVM will be HOG feature vectors instead of raw images.

We implement the simplest variant of HOG, so our output will be different from OpenCV or skimage.

Our HOG implementation mainly follow: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/

and 

https://viblo.asia/p/tim-hieu-ve-phuong-phap-mo-ta-dac-trung-hog-histogram-of-oriented-gradients-V3m5WAwxZO7

## SVM:

This is an algorithm performing the classification task. The input is vectors of image features, and the output is whether -1 (dog) or 1 (cat)

There are 2 SVM implementation in our project. 

First we implemented SVM without any kernel (as known as linear kernel) and optimized by gradient descent. But the accuracy seems to be low as the images relationship is not linear.

Then we implement SVM using SMO (Sequential Minimal Optimization) to optimize. We create 2 kernel option for this implementation is linear and RBF. This implementation have the accuracy same as sklearn module.

SMO is an algorithm used for training support vector machines. It helps optimize the SVM model by solving a series of small quadratic programming problems

# NOTE:
We use Vietnamese as our main language in comments and presentations in most of our files.