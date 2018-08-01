# **Behavioral Cloning** 


---



[//]: # (Image References)


[image-recovery-1]:  ./writeup-images/left-to-middle-part1.jpg "Recovery Image"
[image-recovery-2]: ./writeup-images/left-to-middle-part2.jpg "Recovery Image"
[image-center-lane]: ./writeup-images/center-img.jpg "Normal Image"
[image-nvidia]: ./writeup-images/nvidia.png "nvidia"
[image-flipped]: ./writeup-images/nvidia.png "Flipped Image"


#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* output_video.mp4 and behavior-cloning-video.mov


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Used Nvidia CNN, [code here](https://github.com/BingbingLai/udacity-behavior-cloning/blob/master/model.py#L60-L73): 

- a convolution neural network with 5x5 filter sizes and 3x3 filter sizes
- [Line 61-67](https://github.com/BingbingLai/udacity-behavior-cloning/blob/master/model.py#L63-L67): RELU layers to introduce nonlinearity, and 
- the data is normalized in the model using a Keras lambda layer ([code line 61](https://github.com/BingbingLai/udacity-behavior-cloning/blob/master/model.py#L60-L73)). 
- that's cropped ([code line 62](https://github.com/BingbingLai/udacity-behavior-cloning/blob/master/model.py#L62)), and
- lastly, dropout layers in order to reduce overfitting 





#### 2. Attempts to reduce overfitting in the model

- model.py [line 73](https://github.com/BingbingLai/udacity-behavior-cloning/blob/master/model.py#L73): The model contains dropout layers in order to reduce overfitting 

- model.py [line 31-38](https://github.com/BingbingLai/udacity-behavior-cloning/blob/master/model.py#L31-L48): The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with the following parameters:

- [line 75](https://github.com/BingbingLai/udacity-behavior-cloning/blob/master/model.py#L75-L83)
- learning rate: lr=0.001
- loss: mean square error: 'mse'
- metrics: ['accuracy']




#### 4. Appropriate training data

Choosed the following data to keep vehicle driving in the middle:

- stop using the training data from udacity
- trained the following data:
	- 3 laps of center lane driving, and
	- 2 laps of reverse driving center lane, and
	- recovery from the left and right to the middle of the road, and
	- data aroud driving around corners


For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


- used lenet because udacity mentioned it
  - the car always drove into the river
  - trained the car to drive around corners and also from the opposite direction
  - and it never really worked, no matter how I trained the data
  
- then I tried [Nvidia Network Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), because it's used in the real car, so thought it's probably worth trying:
  - repeated the same process with lenet
  - did not found any solution that could make the car drive the entire track

- probably spend 50+ hours training the data. None really worked; found this [article](https://medium.com/@fromtheast/you-dont-need-lots-of-data-udacity-behavioral-cloning-6d2d87316c52) wrote by the udacity student, insights are:
  - data is the key
  - add ranom brightless to the images
  - the turning could use the following stretagy:
	  - when turning left, the LEFT-hand-side-images could steer more to turn left, conversely
	  - when turning right, the RIGHT-hand-side-images could apply steer more to the right
	  
	  
- with the insights above and [more data](https://github.com/BingbingLai/carnd-project-3/tree/generator), [more trained data](https://github.com/BingbingLai/udacity-behavior-cloning/tree/master/all_training_data), the car was able to drive thru the entire track 1
	- the udacity's training data was less helpful, and was not being used to train the model. assumed the udacity's data is must needed was wrong.
	

	  
#### 2. Final Model Architecture
the model's code is [here](https://github.com/BingbingLai/udacity-behavior-cloning/blob/master/model.py#L60-L83)

basically the image below:
![alt text][image-nvidia]

#### 3. Creation of the Training Set & Training Process

- Recorded three laps on track one using center lane driving:

![alt text][image-center-lane]

- Recorded the vehicle recovering from the left side to the road center:

![alt text][image-recovery-1]
![alt text][image-recovery-2]

- Repeated the process above for recovering from the right side


- Flipped the all the images:

- Had 20k+ of data points
- Randomly shuffled the data set and put 20% of the data into a validation set. 

