# Assignment-3-DCGAN

# Deep Convolutional Generative Adversarial Network (DCGAN) for CelebA

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic-looking celebrity faces using the CelebA dataset.

## Table of Contents

* [Introduction](#introduction)
* [Dataset Preprocessing](#dataset-preprocessing)
* [Training the Model](#training-the-model)
* [Expected Outputs](#expected-outputs)
* [Code Documentation](#code-documentation)
* [Dependencies](#dependencies)

## Introduction

This project utilizes a DCGAN architecture to learn the distribution of the CelebA dataset and generate new images of celebrity faces. DCGANs consist of two neural networks:

* **Generator:** Creates new, synthetic images.
* **Discriminator:** Evaluates the authenticity of images (real or fake).

These networks are trained in an adversarial process, where the generator tries to "fool" the discriminator, and the discriminator tries to correctly classify images. This process leads to the generator producing increasingly realistic images.

## Dataset Preprocessing

The CelebA dataset is preprocessed using the following steps:

1.  **Loading the Dataset:** The `ImageFolder` dataset from `torchvision.datasets` is used to load the images from the specified root directory.
2.  **Transformations:** The following transformations are applied to each image:
    * `transforms.Resize(image_size)`: Resizes the image to the specified `image_size` (64x64 in this case).
    * `transforms.CenterCrop(image_size)`: Crops the center of the image to the specified `image_size`.
    * `transforms.ToTensor()`: Converts the image to a PyTorch tensor.
    * `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`: Normalizes the pixel values to the range [-1, 1]. This is important for the `Tanh` activation function used in the generator's output.
3.  **Dataloader:** A `DataLoader` is created to efficiently load the preprocessed images in batches during training.

## Training the Model

To train the DCGAN, follow these steps:

1.  **Define the Generator (G) and Discriminator (D) Networks:**
    * The code defines the `Generator` and `Discriminator` classes, which implement the DCGAN architecture.
    * The generator takes a latent vector (noise) as input and generates an image.
    * The discriminator takes an image as input and outputs a probability of it being real.
2.  **Initialize Weights:**
    * The `weights_init` function is used to initialize the weights of the convolutional and batch normalization layers in both the generator and discriminator.
3.  **Choose a Device:**
    * The code checks for the availability of a GPU and uses it if available; otherwise, it uses the CPU.
4.  **Define Loss Function and Optimizers:**
    * The Binary Cross-Entropy Loss (`BCELoss`) is used as the loss function.
    * The Adam optimizer is used to update the weights of both the generator and discriminator.
5.  **Training Loop:**
    * The training loop iterates over the specified number of epochs.
    * In each iteration:
        * The discriminator is trained to distinguish between real and fake images.
        * The generator is trained to generate images that can "fool" the discriminator.
        * Losses for both generator and discriminator are tracked.
        * The generator's output on a fixed set of noise vectors is saved periodically to visualize the training progress.
6.  **Visualize Training Progress:**
    * The code generates plots to visualize the generator and discriminator losses during training.
    * It also creates an animation showing the generated images at different stages of training.

## Expected Outputs

After training the DCGAN:

* **Loss Plots:** Plots showing the generator and discriminator losses over iterations, indicating the training dynamics.
* **Generated Images:** A sequence of generated images showing the improvement in image quality as the training progresses. The final generated images should resemble realistic celebrity faces.

## Code Documentation

All Python scripts include docstrings and inline comments explaining functions and logic. The code follows best practices for clean, modular, and reusable code.

## Dependencies

* Python 3.6+
* PyTorch 1.0+
* torchvision
* numpy
* matplotlib
* IPython (for visualization in notebooks)
