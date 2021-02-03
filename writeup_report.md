# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the following layers:

-> initial lambda layer to normalize the images in parallel.
-> a cropping layer to crop the top 70 and bottom 20 pixels of each image in parallel.
-> a convolution layer having relu activation with 24 filters of size 5x5 filter_size followed by a max pool layer.
-> a convolution layer having relu activation with 36 filters of size 5x5 filter_size followed by a max pool layer.
-> a convolution layer having relu activation with 48 filters of size 5x5 filter_size followed by a max pool layer.
-> a convolution layer having relu activation with 64 filters of size 3x3 filter_size followed by a max pool layer.
-> a convolution layer having relu activation with 64 filters of size 3x3 filter_size followed by a max pool layer.
-> a flatten layer.
-> a dense layer with 100 neurons.
-> a dense layer with 50 neurons.
-> a dense layer with 10 neurons.
-> final output dense layer with 1 neuron.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after each of the convolution layers in order to reduce overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I have used adam optimizer, so I did not tune the learning rate, and instead used the most common learning rate as 0.001

#### 4. Appropriate training data

I actually used the data provided in the /opt/carnd_p3/data folder and did not collect it by myself. However, i did flip the images so as to reduce the bias caused by left turns.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with the basic convolution neural network similar to the one taught in the lectures using images as the train images and steering angle as the train labels. The car started driving but it was performing poorly on the turns.

The first step i tried was to augment the data by flipping the images so as to train the model on more number of images. Then I added max pool layer after each of the convolution layer so as to provide regularization and reduce overfitting. I also tried using batch normalization but the model did not perform very well.

Finally, I realized that it was mentioned that the script drive.py reads images in RGB format whereas cv2 reads in BGR format. So, I did that change, and after the model performed well and the car was able to drive autonomously on the track.

#### 2. Final Model Architecture

The final model architecture is as explained above in the lines 57 - 70.

#### 3. Creation of the Training Set & Training Process

I used the training data provided in the /opt/carnd_p3/data folder. However, I did augmentation of the images. For example, here is an image that has then been flipped:

original image:

[//examples/image.jpg]

flipped image:

[//examples/flipped_image.jpg]

After this, I preprocessed the data by normalizing each image using a lambda layer. For splitting the data, I did a random split using train_test_split function with test_size = 20% and then finally shuffled it before yielding from generator function.

After data collection is done and i tried various models for training the car to drive autonomously on the track. As explained above, the car was able to drive autonomously on the track1 as expected.