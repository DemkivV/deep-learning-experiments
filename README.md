# Deeplearning Portfolio
Collection of various deep learning experiments with details and demos of results.

## Autoencoders
### Identity Autoencoder
Convolutional identity autoencoder.

#### Details
* *Training details*:
  * *Training objective*: Reproduction of input after encoding it into a feature map.
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: (28, 28, 1) grayscale images of digits (MNIST dataset).
  * *Ground truth*: Same as input.

#### Results
* *Demo content*:
  * Validation results (input and prediction).
  * Animation Over the course of 20 training epochs.
* *Layout*:
  * *Top row*: Input (= ground truth).
  * *Bottom row*: Prediction of the identity autoencoder.

![Animated ](autoencoder/identity/mnist/conv_autoencoder_20.gif)

### Denoiser Autoencoder
Convolutional denoiser autoencoder.

#### Details
* *Training details*:
  * *Training objective*: Reproduction of input after encoding it into a feature map.
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: (28, 28, 1) grayscale images of digits (MNIST dataset).
    * *Augmentation*: With noise (preprocessed once before the training).
  * *Ground truth*: Unaugmented input.

#### Results
* *Demo content*:
  * Validation results (input and prediction).
  * Animation Over the course of 20 training epochs.
* *Content*:
  * Validation results (input and prediction).
  * Over the course of 20 training epochs.
* *Layout*:
  * *Top row*: Input.
  * *Mid row*: Prediction of the denoiser autoencoder.
  * *Top row*: Ground truth.

![Animated ](autoencoder/denoiser/mnist/conv_autoencoder_20.gif)

### Super-Resolution Autoencoder
Convolutional super-resolution autoencoder which quadruples the resolution of the input image.

#### Details
* *Training details*:
  * *Training objective*: Upscale the input image to the quadruple resolution (double width and height).
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: (14, 14, 1) grayscale images of digits (MNIST dataset), downscaled from original size (28, 28, 1).
    * *Augmentation*: Noise (preprocessed once before the training).
  * *Ground truth*: (28, 28, 1) original resolution of input image.

#### Results
* *Demo content*:
  * Validation results (input and prediction).
  * Animation Over the course of 20 training epochs.
* *Content*:
  * Validation results (input and prediction).
  * Over the course of 20 training epochs.
* *Layout*:
  * *Top row*: Input.
  * *Mid row*: Prediction of the denoiser autoencoder.
  * *Top row*: Ground truth.

![Animated ](autoencoder/super-resolution/mnist/conv_autoencoder_20.gif)
