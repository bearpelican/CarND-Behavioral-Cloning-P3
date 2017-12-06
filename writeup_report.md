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

[image1]: ./examples/cnn-architecture.png "Architecture Visualization"
[image2]: ./examples/dirt_road.jpg "Dirt Road"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* training.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run.mp4 contains video recording for autonomous mode submission

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The training.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the NVIDIA architecture that was suggested to us. (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

5 Convolutional layers with 5x5 filters, 3x3 filters, and depths ranging from 24-64. 4 Dense layers with sizes 100, 50, 10, 1.

The model includes RELU layers to introduce nonlinearity (cell 8), and the data is normalized in the model using a Keras lambda layer (cells 8). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (cell 8). 

The model also has several batch normalization layers to speed up training and potentially reduce overfitting (cell 8)

The model was trained and validated on different data sets (80/20 split) to ensure that the model was not overfitting (cell 11). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with a default learning rate of .001 (cell 9). There was no manual learning rate decrease.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I solely used the sample data given to us. I found that the model performed worse when trained on manually recoded data. (cell 6)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from the NVIDIA architecture, test with the sample data, and iterate from there.

After building with initial architecture and testing with the sample data (center lane only), I realized I was overfitting by quite a bit (big difference between training and validation loss).

To alleviate this, I used data augmentation. This would also help with my model accuracy.
* Flip images horizontally and reverse the steering
* Include right and left camera images. Add a slight angle to the steering to offset camera location.

Data augmentation alone was not enough to prevent overfitting. I added batch normalization and dropout to the model. You can see the training and validation losses converge (cell 12).

Finally, I ran the output model against the simulator. The model seemed to fail every single time when it approached the dirt road

![dirt_road][image2]

My hypothesis was that the model was biased on going straight and did not turn fast enough. To address this, I removed 70% of the images that had 0-degree steering. This means I trained on mostly images that had the car turning (cell 6 - discard_prob). 

I also added a stronger steering offset for the right camera images. Since my autonomous car had trouble turning left, I changed the right camera steering offset from -0.23 degrees to -0.27 (cell 4). I believe this causes the car to make sharper left turns when it is near the right lane.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (training.ipynb cell 8) consisted of a convolution neural network with the following layers and layer sizes:

1. Normalization
2. Conv2d (5x5 filter, depth=24)
3. Conv2d (5x5 filter, depth=36)
4. Conv2d (5x5 filter, depth=48)
5. BatchNorm
6. Conv2d (3x3 filter, depth=64)
7. Conv2d (3x3 filter, depth=64)
8. Flatten
9. Dense (size 100)
10. BatchNorm
11. Dropout (50%)
12. Dense (size 50)
13. BatchNorm
14. Dropout (50%)
15. Dense (size 10)
16. BatchNorm
17. Dense (size 1)

Here is a visualization of the architecture taken from NVIDIA

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The training set solely consisted of data from the sample set. No self-recorded data was used for training.

*** Note
I did manually record a several laps. Some in reverse and some at tricky turns. However, when I added the manual recordings to the sample data set, I found that the model somehow performed worse. 

My hypothesis is that the recorded data had much smoother steering than the sample data. The sample data was generated by short steering bursts, while my recorded data had long, gradual steering angles. This means the recorded data caused the model to understeer. Due to the differences in driving styles, I imagine I would need to train on either data sets, but not on both combined.
***

I used images from the center, left and right cameras. I also flipped the images and steering measurements to double the data set.

The training labels were preprocessed by adding a 0.23 steering offset to the left camera images and a -0.27 steering offset to the right camera images. The right camera offset is higher because the car had trouble with making sharp left turns.

I also removed 70% of the images that had 0-degree steering. This means the model was trained mostly on images that required turning (cell 5, 6 - discard_prob).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by convergence of the training and validation losses. I used an adam optimizer and did not adjust the default learning rate.
