# Deep Learning Portfolio
Collection of various deep learning experiments with details and demos of results. (Work in progress.)

Of course all demos consist of evaluation images that were never shown in training.

1. [Autoencoders](#autoencoders)

    1.1 [Identity](#identity)

    1.2 [Denoising](#denoising)

    1.3 [Super-Resolution](#super-resolution)

# Autoencoders
## Identity
Convolutional identity autoencoder. It could be useful to evaluate the precision of feature representations and find the right spot between undercomplete and overcomplete representations to avoid overtraining and underperformance for the given data domain.

### Details
* *Training details*:
  * *Training objective*: Reproduction of input after encoding it into a feature map.
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: Grayscale/color images.
  * *Ground truth*: Same as input.

### Results
Animation of validation results for first 25 epochs of training.


#### MNIST Dataset
![](autoencoder/identity/mnist/conv_autoencoder_25.gif)




#### Fashion MNIST Dataset
![](autoencoder/identity/fashion_mnist/conv_autoencoder_25.gif)




#### CIFAR-10 Dataset
![](autoencoder/identity/cifar10/conv_autoencoder_25.gif)



## Denoising
Convolutional autoencoder. It removes noise from the input image. This can be useful e.g. for photos taken in the dark.

### Details
* *Training details*:
  * *Training objective*: Reproduction of input after encoding it into a feature map.
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: Grayscale/color images.
    * *Augmentation*: With noise (preprocessed once before the training).
  * *Ground truth*: Unaugmented input.

### Results
Animation of validation results for first 25 epochs of training.


#### MNIST Dataset
![](autoencoder/denoiser/mnist/conv_autoencoder_25.gif)




#### Fashion MNIST Dataset
![](autoencoder/denoiser/fashion_mnist/conv_autoencoder_25.gif)




#### CIFAR-10 Dataset
![](autoencoder/denoiser/cifar10/conv_autoencoder_25.gif)




## Super-Resolution
Convolutional autoencoder. It quadruples the resolution of the input image. This can be useful e.g. for supersampling, but also more efficient rendering at a lower original resolution.

### Details
* *Training details*:
  * *Training objective*: Upscale the input image to the quadruple resolution (double width and height).
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: Grayscale/color image downscaled by factor 2.
    * *Augmentation*: Noise (preprocessed once before the training).
  * *Ground truth*: Original resolution of input image.

### Results
Animation of validation results for first 25 epochs of training.


#### MNIST Dataset
![](autoencoder/superresolution/mnist/conv_autoencoder_25.gif)




#### Fashion MNIST Dataset
![](autoencoder/superresolution/fashion_mnist/conv_autoencoder_25.gif)




#### CIFAR-10 Dataset
![](autoencoder/superresolution/cifar10/conv_autoencoder_25.gif)


