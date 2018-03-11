# Behaviorial Cloning Project

[](https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8)
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

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

Whoola!!! After this revision, the vehicle can run the entire loop, but it crossed over the lane line sometimes after a turn. I preferred to add some corner cases into the dataset to enhance the robustness of network. After adding some data inside and retrained, the vehicle failed to pass the whole loop which made me confused which might lead to my network is not robust at all. Finally, I found I forgot to convert image into YUV color space before feeding it into the model, which made me curious why NVidia's was failed as well @@. The track has been conquered!

#### Network #2
- Data Normalization
- Convolution:    5x5, 64, strides: 2x2 , act.: ReLU
- Convolution:    5x5, 64, strides: 2x2 , act.: ReLU
- MaxPooling:     2x2
- Dropout:        0.5
- Convolution:    3x3, 32, act.: ReLU
- MaxPooling:     2x2
- Dropout:        0.5
- Convolution:    3x3, 16, act.: ReLU
- Flatten
- FC Convolution: 200, act.: ReLU
- FC Convolution: 20, act.: ReLU
- FC Convolution: 1

After the lake track, here's the jungle track. I think this will be more difficult than previous one due to more complicated surroundings. I spent lots of time in a problem 'MemoryError' when loading data into np.array like 'y_train = np.array(measurements)', which I fixed it by creating empty list first('y_train = []'). I found that my vehicle was stopped in the middle of the track, so I trigger it to move on during recoding.

Done! The videos will show how the vehicle drives with the links at the bottom~

### summary
In model part. I tried the network implemnted in Traffic Sign Project which look not very well without modification. So I adjusted my network based on NVidia's one to make it understand more which I think the previous network is not wide enough at the beginning. The final model is network #2 shown above.
In dataset part, I first recorded one loop as dataset which made vehicle sometimes stuck in some place (crossed the lane line, bumped to the bridge, went into the farm,...etc), so the more dataset was recored. The final dataset is consisted of: 
### Lake track
- 3 loop forward
- 2 loop backward
- Some corner cases (such as have a big turn closed to bridge or the lane)

### Jungle track
- 2 loop forward
- 1 loop backward
- Some corner cases (take a big turn closed the edge, moved into shadowed area)

## Results

| Lake Track | Jungle Track |
|-|-|
|[![lake track](https://img.youtube.com/vi/DtqpVTdao9I/0.jpg)](https://www.youtube.com/watch?v=DtqpVTdao9I) | [![lake track](https://img.youtube.com/vi/G_Apv_TdHmc/0.jpg)](https://www.youtube.com/watch?v=G_Apv_TdHmc) |
