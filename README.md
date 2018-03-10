# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files:
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

---

My Study
---
First of all, I tried to teach the vehicle drive besides the lake. A video was recorded by simulator and trained on **LeNet**. It turned out that it keep turning around or driving in and out the lane line no matter how I drove it back to center. Then I decided to train on the **network #0** designed in 'Traffic Sign Project' using same **dataset #0**. Unfortunately, the vehicle still do not know how to drive. The performance was enhanced after retrained the new **dataset #1**, because I think dataset #0 might not cover all the situations. The vehicle already knew how to drive forward, turn around and cross the bridge, but failed on the road without lane line after crossing the bridge. I think I need more data to teach the vehicle how to drive.
There are some steps how the data was preprocessed: normalization,  augmentation, and cropping. At the beginning, vehicle drives better with normalized data than without one, so the data will be fed after normalization with mean and standard deviation were 0.0 and  0.5. The images were flipped horizontally to create more data. And as lesson said, the images were also cropped to decrease irrelevant stuffs in the image to become noises.

#### Network #0
- Data Normalization
- Convolution:    5x5, 64, act.: ReLU
- Convolution:    3x3, 64, act.: ReLU
- MaxPooling:     2x2
- Dropout:        0.5
- Convolution:    3x3, 32, act.: ReLU
- Dropout:        0.5
- Flatten
- FC Convolution: 512, act.: ReLU
- FC Convolution: 256, act.: ReLU
- Dropout:        0.5
- FC Convolution: 1

New **dataset #2** was then recorded to test that if I should change my network architecture. In order to test my network's robustness, **NVidia's network #1** for autonomous vehicle was implemented to compare the performance. Under the identical condition and input (RGB color image), both of them looks similar and went off the road at same place which I think the dataset is still not good enough @@. The dataset was revised and test again. It turned out network #1 is better than network #0 which can get the vehicle back to the center from ignoring the lane line and out of the way. Meanwhile, I found network #0 took more time to train than network #1 due to too much parameters. It cost almost 11GB to store all the data on GPU with batch size is 16, estimated by [the code from ZFTurbo & Fabrício Pereira](https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model), which I do not have. So I decided to adjust my **network #2** based on network #0 and #1. The strides was expanded from (1, 1) to (2, 2) in convolution to reduce the size, the images were converted into YUV color space to seek for better performance.

#### Network #1

Ref.: [NVidia's End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

- Data Normalization
- Convolution:    5x5, 24, strides: 2x2 , act.: ReLU
- Convolution:    5x5, 36, strides: 2x2 , act.: ReLU
- Convolution:    5x5, 48, strides: 2x2 , act.: ReLU
- Convolution:    3x3, 64, strides: 2x2 , act.: ReLU
- Convolution:    3x3, 64, strides: 2x2 , act.: ReLU
- Flatten
- FC Convolution: 100, act.: ReLU
- FC Convolution: 50, act.: ReLU
- FC Convolution: 10, act.: ReLU
- FC Convolution: 1

Whoola!!! After this revision, the vehicle can run the entire loop, but it crossed over the lane line sometimes after a turn. I decided to add more dataset to enhance the robustness of network.

#### Network #2
- Data Normalization
- Convolution:    5x5, 64, strides: 2x2 , act.: ReLU
- Convolution:    3x3, 64, strides: 2x2 , act.: ReLU
- MaxPooling:     2x2
- Dropout:        0.5
- Convolution:    3x3, 32, act.: ReLU
- Dropout:        0.5
- Flatten
- FC Convolution: 512, act.: ReLU
- FC Convolution: 256, act.: ReLU
- Dropout:        0.5
- FC Convolution: 1

---

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
