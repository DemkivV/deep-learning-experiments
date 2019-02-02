# deeplearning_portfolio
Demos regarding various deep learning experiments

## Autoencoders (AE)
### Identity Autoencoder (IAE)
Convolutional identity autoencoder for MNIST dataset.

#### Details
* *Training details*:
  * *Training objective*: Reproduction of input after encoding it into a feature map.
  * *Loss function*: Mean squared error.
  * *Optimizer*: Adam.
* *I/O*:
  * *Input*: (28, 28, 1) grayscale images of digits (MNIST dataset).
  * *Ground truth*: Same as input.

#### Results

* *Type of demo*: GIF animation.
* *Content*:
    * Validation results (input and prediction).
    * Over the course of 20 training epochs (afterwards only minor improvements).
* *Layout*:
    * *Top row*: Input (= ground truth).
    * *Bottom row*: Prediction of the IAE.

![Animated ](autoencoder/identity/mnist/conv_autoencoder_20.gif)
