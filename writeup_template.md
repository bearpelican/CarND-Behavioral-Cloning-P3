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

The model includes RELU layers to introduce nonlinearity (cells 7), and the data is normalized in the model using a Keras lambda layer (cells 7). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (cell 7). 

The model also has several batch normalization layers to speed up training and potentially reduce overfitting (cell 7)

The model was trained and validated on different data sets (80/20 split) to ensure that the model was not overfitting (cell 9). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with a learning rate of .001 (cell 7). The learning rate was decreased manually after the first run (cell 11 and 12).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of the sample data given to us, center lane driving, and recordings of tricky turns.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from the NVIDIA architecture, test with the sample data, and iterate from there.

After building with initial architecture and testing with the sample data (center lane only), I realized I was overfitting by quite a bit (big difference between training and validation loss).

To alleviate this, I used data augmentation. This would also help with my model accuracy.
* Flip images horizontally and reverse the steering
* Include right and left camera images. Add a slight angle to the steering to offset camera location.

Data augmentation alone was not enough to prevent overfitting. I added batch normalization and dropout to the model. You can see the training and validation losses converge (cell 10).

Finally, I ran the output model against the simulator. The model seemed to fail every single time when it approached the dirt road

![dirt_road][image1]

To improve driving behavior in this case, I manually recorded a few sequences to add to the data set. This included specific sequences on the track where the autonomous car had messed up - i.e. the dirt road.

Still, the model was biased on going straight and did not turn fast enough. To address this, I removed 70% of the images that had 0-degree steering. This means I trained on mostly images that required turning (cell 8 - discard_prob). 

I also added a stronger steering offset for the right camera images. Since my autonomous car had trouble turning left, I increased the right camera steering labels to favor turning left.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

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

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded several sequences at the location where the initial autonomous model failed - left turn before the dirt road. These images show the location where the car had trouble:

![alt text][image3]
![alt text][image4]
![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded the car going on the track in reverse direction to snag a few more data points.

To augment the data sat, I also flipped images and angles thinking that this would to reduce the model bias steering to the left. Since the track direction is counter-clockwise. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 4000 number of data points. I then preprocessed this data by adding a 0.23 steering offset to the left camera images and a -0.27 steering offset to the right camera images. The right camera offset is higher because the car had trouble with making sharp left turns.

We also removed 70% of the images that had 0-degree steering. This means the model was trained mostly on images that required turning (cell 8 - discard_prob). 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by convergence of the training and validation losses. I used an adam optimizer, but still adjusted the learning rate manually - just in case.
