# Deep Convolutional GANs (DC_GANs) with Wasserstein-GP

## Introduction:
In this project, I aimed to train a DC GANs model with Wasserstein-GP method using DC_GANs architecutre with Tensorboard to generate the images of Cifar10 dataset images from random noise.

## Tensorboard:
TensorBoard, along with saving training or prediction images, allows you to save them in TensorBoard and examine the changes graphically during the training phase by recording scalar values such as loss and accuracy. It's a very useful and practical tool.

## Dataset:
- I used the Mnist dataset for this project, which consists of 10 labels (handwritten digits) with total 60k images on train and 10k images on test.

## Models:

### DC GANS:
  - DC GANs have two fundamental models called generator and discriminator just like basic GANs.
  -  The discriminator takes in generated and real images as input, passing them through several (conv, instanceNorm, LeakyRelu) operations to reduce their dimensions, ultimately producing an output of size (1,1,1) to determine whether the generated images are real or fake.
  -   The generator, on the other hand, takes in random noise of size (100,1,1) and increases its dimensions through ConvTranspose layers, generating an RGB image of size (3,64,64) at the end.

### ProblemS of GANs:
  - The generator and discriminator models actually train themselves using separate loss and optimizer values, striving to produce a common output without being heavily dependent on each other, making it difficult to reach Nash equilibrium. In a discrete setting, there's no guarantee of achieving success together. 
  - If the discriminator works too well, it easily detects that the images generated by the generator are fake, leading to very low loss, which in turn hinders gradient updates (Vanishing gradient). Similarly, if the generator works too well, it easily deceives the discriminator, resulting in unrealistic images even if the loss is low.
  
### Wasserstein-GP :
  - In the Wasserstein method, the discriminator takes on a role where it measures the KL divergence derived from Wasserstein instead of its old role.
  -  A decrease in KL divergence implies that the generator is producing data closer to real data. Therefore, the discriminator strives to increase it while the generator tries to decrease it, leading to better cooperation between them.
  -   In WGANs, gradients are clipped to [-0.01, 0.01], but it's noted that this isn't a very good solution.
  -    In the update that followed, instead of clipping, WGANs-GP penalizes gradients if the penalty deviates from 1. This has led to more effective results.

##### FOR MORE DETAILS:
 - DC_Gans: https://arxiv.org/abs/1511.06434
 - WGANs: https://arxiv.org/abs/1701.07875
 - WGANS-GP : https://arxiv.org/abs/1704.00028

## Train:

- The discriminator is trained as described in the paper, with gradient penalty calculated five times per epoch.
- The generator is trained once based on the results of discriminator training. 
- The loss function used is KL-Divergence instead of BCELoss. 
- The optimizer chosen is Adam with a learning rate of 5e-5.
- I trained just 5 epochs this model


## Results:
- After 5 epochs,There are generated images and graph of values on tensorboard.

## Usage: 
- You can train the model by setting "TRAIN" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Tensorboard files will created into "Tensorboard" folder during training time.
- Then you can generate the images from random noise by setting the "LOAD_CHECKPOINTS" and "TEST" values to "True" in the config file.
