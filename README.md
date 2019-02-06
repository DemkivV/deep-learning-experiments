# Deep Learning Portfolio
Collection of various deep learning experiments with details and demos of results. (Work in progress.)

Of course all demos consist of evaluation images that were never shown in training.

1. [Deep Learning Experiments](#deep-learning-experiments)

    1.1 [Identity Autoencoder](#identity-autoencoder)

    1.2 [Denoiser](#denoiser)

    1.3 [Super-Resolution](#super-resolution)

    1.4 [Pixel-Based Classification](#pixel-based-classification)

2. [Datasets](#datasets)

    2.1 [MNIST](#mnist)

    2.2 [Fashion MNIST](#fashion-mnist)

    2.3 [CIFAR-10 and CIFAR-100](#cifar-10-and-cifar-100)

    2.4 [Cityscapes](#cityscapes)

    

# Deep Learning Experiments
## Identity Autoencoder
Convolutional autoencoder. It could be useful to evaluate the precision of feature representations and find the right spot between undercomplete and overcomplete representations to avoid overtraining and underperformance for the given data domain.

### Details
* *Training details*:
  * *Training objective*: Reproduction of input after encoding it into a feature map.
  * *Architecture*: Shallow UNet (depth 2) with additional vertical residual connections.
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: Grayscale/color images.
  * *Ground truth*: Same as input.

### Results

#### MNIST Dataset
![](deep_learning_experiments/mnist/identity_25.gif)




#### Fashion MNIST Dataset
![](deep_learning_experiments/fashion_mnist/identity_25.gif)




#### CIFAR-10 Dataset
![](deep_learning_experiments/cifar10/identity_25.gif)




#### CIFAR-100 Dataset
![](deep_learning_experiments/cifar100/identity_25.gif)




## Denoiser
Convolutional autoencoder. It removes noise from the input image. This can be useful e.g. for photos taken in the dark.

### Details
* *Training details*:
  * *Training objective*: Reproduction of input after encoding it into a feature map.
  * *Architecture*: Shallow UNet (depth 2) with additional vertical residual connections.
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: Grayscale/color images.
    * *Augmentation*: With strong noise (preprocessed once before the training).
  * *Ground truth*: Unaugmented input.

### Results

#### MNIST Dataset
![](deep_learning_experiments/mnist/denoiser_25.gif)




#### Fashion MNIST Dataset
![](deep_learning_experiments/fashion_mnist/denoiser_25.gif)




#### CIFAR-10 Dataset
![](deep_learning_experiments/cifar10/denoiser_25.gif)




#### CIFAR-100 Dataset
![](deep_learning_experiments/cifar100/denoiser_25.gif)




## Super-Resolution
Convolutional autoencoder. It quadruples the resolution of the input image. This can be useful e.g. for supersampling, but also for more efficient rendering at a lower original resolution.

### Details
* *Training details*:
  * *Training objective*: Upscale the input image to the quadruple resolution (double width and height).
  * *Architecture*: Very shallow UNet (depth just 1) with additional vertical residual connections and additional upconvolution block.
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: Grayscale/color image downscaled by factor 2.
    * *Augmentation*: Noise (preprocessed once before the training).
  * *Ground truth*: Original resolution of input image.

### Results

#### MNIST Dataset
![](deep_learning_experiments/mnist/superresolution_25.gif)




#### Fashion MNIST Dataset
![](deep_learning_experiments/fashion_mnist/superresolution_25.gif)




#### CIFAR-10 Dataset
![](deep_learning_experiments/cifar10/superresolution_25.gif)




#### CIFAR-100 Dataset
![](deep_learning_experiments/cifar100/superresolution_25.gif)





## Pixel-Based Classification

*Coming soon*:

Recap of my Master's thesis results, where I trained a [Fully Convolutional DenseNet](https://arxiv.org/abs/1611.09326) for pixel-based classification with the Cityscapes dataset.

# Datasets

This chapter is meant to provide a short introduction to different interesting datasets, their advantages and when it's appropriate to work with them. Since this chapter has some tutorial flavor, it will soon be revised and probably separated into an own tutorial and/or learning path, which I'll be extending while learning myself.

Regarding the licenses for my predicted images, they are of course licensed accordingly with the same license as the input data. Regarding my own images, they are unlicensed and free for any use, similar to [the Unlicense](http://unlicense.org/) philosophy. Though, I appreciate staring or mentioning this repository. Thanks. :smile:

<!-- add valid license for media similar to the Unlicense -->



## MNIST

MNIST is the hello world of deep learning. The "*Hello World*" when experimenting with your first neural networks.  It's providing a total of . Since it's only image with the size (28, 28, 1), you can have the fastest neural network training possible. Given a decent GPU, you can finish training experiments within minutes. It's nice dataset to gather some experience with basic architecture elements and different architectures. Though, since the problem is very simple, some strategies that work.

Take a look at [the official website](http://yann.lecun.com/exdb/mnist/) for more info. According to [this source](http://www.pymvpa.org/datadb/mnist.html), it's licensed with *Creative Commons Attribution-ShareAlike 3.0*. It's easily accessible within Keras and commonly used for deep learning introductions.

As with all mentioned very low resolution datasets, with a *GTX 1070* it took roughly 5–6min for 25 epochs. Though, improvements for such simple tasks actually slowed down already after a few epochs.

<!-- TODO: Add a few example images -->

## Fashion MNIST

Since MNIST is too simple, fashion MNIST is the next level of complexity. It is similar to the MNIST dataset. The only difference is, that instead of digits, you have 10 different types of clothing. While for MNIST, a neural network could achieve extraordinary results with just classifying every pixel either black or white, this dataset offers new challenges: Meaningful grayscale values and patterns.

The dataset is provided by [Zalando's research group](https://github.com/zalandoresearch/fashion-mnist) and is licensed with the *MIT license*. It's also easily accessible within Keras and commonly used for deep learning introductions.

As with all mentioned very low resolution datasets, with a *GTX 1070* it took roughly 5–6min for 25 epochs. Though, improvements for such simple tasks actually slowed down already after a few epochs.

<!-- TODO: Add a few example images -->

## CIFAR-10 and CIFAR-100

The CIFAR10 and CIFAR100 datasets are the next step forward. In contrast to the previous two datasets, which were grayscale, this one is RGB and has a slightly higher resolution with a size of (32, 32, 3). Since the resolution is 2^n, it's also easier to work with convolutions, since you have to take less care with paddings, which is a little bit more tricky and restricting with the previous (28, 28, 1) shape. Those two datasets are very low resolution photographs. The CIFAR10 datasets consists of 10 classes while the CIFAR100 dataset consists of 100 classes accordingly. This is were classification is becoming more challenging.

Take a look at [the official website](http://yann.lecun.com/exdb/mnist/) for more info. The licensing seems unspecified, but the images could be subject to copyright. As well as the previously mentioned datasets, it's also easily accessible within Keras and commonly used for deep learning introductions.

As with all mentioned very low resolution datasets, with a *GTX 1070* it took roughly 5–6min for 25 epochs. Though, improvements for such simple tasks actually slowed down already after a few epochs.

## Cityscapes

Big jump forward: The Cityscapes is an oasis for more complex deep learning experiments. How that you warmed up with some easy and fast-to-train architectures, it's time to level up! This dataset provides a huge array of different data and is especially suited for gathering experience with autonomous driving problems. Here's a list of the available data:

* **Binocular photographs**
  * From the perspective of the car hood
  * Total of 25'000 binocular image pairs
  * Taken in...:
    * 50 different large German cities
    * different weather conditions and seasons
  * Available in either 8 Bit or 16 Bit (HDR).
* **Precomputed depth**
  * Disparity maps, computed with stereo matching from the binocular images
  * 25'000 images as well, one for every binocular image pair
* **Annotations** (labels)
  * 30 classes
  * 5'000 annotation images precisely annotated
  * 20'000 annotation images coarsely annotated
* [**Some scripts/tools**](https://github.com/mcordts/cityscapesScripts)
  * Among others: An [annotation tool](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/annotation/cityscapesLabelTool.py), with which you could create annotations e.g. for own data

The images have a high resolution and thus give you the possibility to experiment with very deep, state-of-the-art convolutional neural network architectures. You can also experiment with exploring advantages of multimodal input (e.g. additional depth data vs. just RGB), segmentation, class hierarchies and more. Highly recommended dataset.

<!-- (TODO: specify exact resolution) -->

The dataset is provided by Daimler AG, MPI Informatics and TU Darmstadt. You can request access on the [Cityscapes website](https://www.cityscapes-dataset.com/). They have a [custom license](https://www.cityscapes-dataset.com/license/), but – in short – it's free for non-commercial purposes.

<!-- TODO: Add a few example images -->

For my experiments with the *Fully Convolutional Densenets* and a *GTX 1070*, it took roughly 8–12h until results stagnated. Though, back then I worked with *Theano* and *Lasagne*, so not sure about the training time with today's *Keras* with *Tensorflow* backend. Probably in the same range though.