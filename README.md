# Motion

## Abstract

Often today it is very common to measure somebody’s health and wellbeing solely on the basis of speed and strength. Although these are important, often for sport/performance, there is another dimension of flexibility and mobility which are strong contributors to longevity/injury-prevention. One major hurdle for people is that flexibility and mobility take far longer build and train (especially if your biology is that of a male!), and it is easy for a person to fall into the trap of only training what they are good at which is often very inefficient. 

Motion is a program which aims to produce a tailored program for users to effectively improve flexibility and mobility effectively. Motion implements the CMU OpenPose 1.7 Body_25 model for an in-depth analysis of an overhead squat. This involves using OpenCV for getting a video file input, extracting frames, stitching frames, saving frames as .jpg files, manipulating output data from the model to find Euclidean distances, angles of joints, vectorising points in 2-Dimensional space, and plotting the poses onto images in order to analyse a user’s overhead squat alongside dabbling in pose similarity and pose classifications using KNN and CNN approaches using Tensorflow and Keras libraries.

This has become especially relevant with the happening of Covid-19 pandemic in 2020 which affected everyone. Throughout this period, prevented from socialising, discouraged from going outside, businesses are shut, gyms are closed, and takeaways are cheap. It becomes very easy for people to forget to look after themselves. Especially where people are expected to carry on with all usual responsibilities such as a professional or University work. At Universities, the continually increasing emphasis on mental health from students has become more significant, applying more pressure and responsibilities than ever before onto these institutes. On the scale of a community, this can be a seemingly impossible and increasingly complicated problem to manage. To a computer scientist, it is understood that the solution to any problem is also the sum of the solutions to all its counterparts. Even if somebody could not solve the entire problem, targeting some of the problem may bring us much closer to a complete solution. Motion wishes to try and help some of this problem on the scale of the individual. In England it is estimated that 1 in 4 people will suffer from mental health problems each year (Mind.org.uk, 2020) 17% of these are in the forms of anxiety and depression. People who took part in a recent Harvard study has showed that by introducing a form of minor activity daily such an hour of walking, stretching, or any kind of low intensity movement can reduce risk of major depression by 26% and on top of that significantly help individuals from relapsing (Harvard, 2019). This does however, require individuals to take some responsibility for themselves. 

In conclusion, the CMU model has made significant progress over the last 4 years (since 2017 release) and is very precise from a front on perspective. The model is only reliable when the user is square to the camera, as human pose estimation continues to improve, hopefully one day it will have the reliability to be implemented into an application like Motion.

## Overview of human Pose Estimation 

Carnegie Mellon University are among the forefront of AI research and in this case Human Pose Estimation. OpenPose was released in October 2017 and has since had frequent and major improvements. Best of all, it is freely available to all non-commercial use, alongside great documentation. It was not until later that it was decided to use the CMU body_25 model, but for background knowledge after reading from the official OpenPose cited documentation (Zhe Cao, 2018) and sources online (Tanugraha, 2019) , (Raj, 2019). Here is a brief overview.

There are two main approaches to this problem, first is the simpler top-down approach where the model is trained to detect the human first (which is relatively simple) and then deriving each key point from the humans detected. Simply, detect each person then derive body parts from each body for every person. 

The second approach is the bottom-up approach which OpenPose uses. This is where each body part belonging to everyone is detected first and then grouped and associated together to form the person. Simply, locate everybody part in frame then using grouping methods such as part affinity fields and part confidence maps stitch each human together. Using this method, performance seems to suffer far less as more people are detected in frame.

Confidence maps are a 2D representation of the belief that a particular key point occurs in each given pixel. If there is one person in the image, a single peak should exist in each confidence map if the corresponding part is visible.

Part Affinity fields for part association needs a confidence value measuring the association of each pair of body parts belonging to which person within an image. This preserves the location and orientation information across the region supporting the limb (body part/key point). Represented by a 2-Dimensional vector field for each limb, pointing from one part of the limb to the other. Each limb has a part affinity field directing to associated body parts.
![](imageDirectory/HPEOverview1.JPG)
![](imageDirectory/HPEOverview2.JPG)

From an RGB image input, the Convolutional Neural Network splits into two layers and is stacked, default stack depth is set to 6, refining its predictions at each stage. The first layer outputs Part Confidence Maps, and the second outputting Part Affinity Fields as shown in the pipeline of figure 1. Each iteration of the stack returns more confidence in results. Then finally, outputs are processed by greedy inference to output 2D pose points for each person in frame. Both the body_25 and COCO model mentioned later are based off this architecture (shown in figure 2).

## Implementation/Realisation

### Implementing a model

The first problem was the decision in which model to use as a basis. This was a very important factor as the performance of the model would greatly impact the project. After researching the current best performing models open for commercial use and personal projects. Carnegie Mellon University (CMU) in Pennsylvania released OpenPose which achieves much better results than any other freely available model. 

The first model implemented was a CPU based COCO model. As seen in figure 10, on an intel core i5 (2.90 GHZ) CPU it was running at 1.8 frames. This model tracks 17 points on the human body. When moving the frame rate would drop as low as 0.5 fps and would often be unable to confidently locate key points. This was an issue for a project which required a high degree of accuracy such as Motion.

![](imageDirectory/coco.PNG)

Figure 11 shows how the COCO model maps a human pose. It tracks 17 key points across the body, one major issue is that it does not track below the ankle. This information is extremely important for certain movements. Because of the slow speed and lack of accuracy it was decided it were best to switch to an alternative model. For implementation, an anaconda environment was created with an older version of TensorFlow (1.1.5.3) and python 3.7.9. As the original model is in C++ a wrapper called SWIG was used so it could be implemented with python.
![](imageDirectory/coco2.PNG)

Not long after this (17th of November 2020) CMU released the newest version of OpenPose 1.7. (CMU, 2017) Using the body_25 model, which is 40% faster and 5% more accurate and includes foot key points. 

When implementing body_25 model for speed it was GPU boosted, however should be no different to a CPU implementation, only faster.

his model was significantly better, as it ran between 14-15 frames per second even with large amounts of movement. On top of this, the body_25 model tracks 24 key points (25 if you include the background) including toes and heels, which is very valuable for analysing many movements where the individual is on their feet.

As shown in figure 13 this mapping Is far more intricate which will allow for more in-depth analysis. For example, by having points 1 (Neck), 8 (Mid-hip), 9 (Right-hip), and 12 (Left-Hip). It is much easier to detect posture or tilting in the upper body. It also looks to be a much truer representation of  a human pose. This model also maps below the ankle ,points 21 and 21 track the heels and points (22,23) , (19,20) tracks the big and little toes on each foot. This degree of pose estimation allows analysis in the relationship between heel and toes. This can be used as an indicator for balance or for a squat, it could show if heels are leaving the ground (which is quite common in inflexible people) and can be done by looking at the distances between the ankles and toes. In the real world, these distances will not change however, if the distance between the corresponding toes and ankles increase from the perspective of the camera then the cause of this would be due to the heel coming off the floor. Because of the overwhelming advantages and good documentation for the second implementation, that was the model chosen in the further development of Motion.

## Demo
will do some more detailed documentation explaining how ot works and the maths behind it another day.

![](imageDirectory/dataflow.png)

![](imageDirectory/exampleInput.png)

![](imageDirectory/skeleton.png)

![](imageDirectory/allAngles.png)

![](imageDirectory/splits.png)

![](imageDirectory/squatAngles.png)

![](imageDirectory/outputExample.png)




## Environment Set Up

Setting up a working environment proved to be very difficult as every user online had a different set up. This took a large amount of time trying different combinations which others claimed to work. Firstly Tensorflow-gpu and Keras were set up and installed within the environment. In the end downgrading Cuda from 11.1 to 10.2 and Cudnn from 8.0.5 to 7.6.5 helped solve some compatibility issues (although various other driver updates were installed). Cmake was used as a python wrapper for the model. The last hurdle was building the environment, downgrading to Visual Studio 2017 but only by using the Enterprise edition somehow worked. In the end, the environment was as follows:

System: Windows 10 

OpenPose: 1.7

Cuda: 10.2

Cudnn: 7.6.5 (for Cuda 10.2)

Visual Studio 2017 Enterprise

Python: 3.7.8


3rd party dependencies:

•	Caffe – fast deep learning framework

•	Pyblind11 – for Cmake python wrapper


Library Dependencies:

•	Math – maths operators.

•	From pathlib – Path function for directory traversal.

•	Os – for getting pathways and directory management.

•	Cv2 – computer vision tools.

•	Numpy – manipulating arrays. 

•	Tensorflow – Machine learning library for neural networks.

•	Keras – for batch normalisation in CNN layers.

•	Sklearn – managing training data and getting accuracy.

•	Matplotlib.pyplot - plotting results.

Clone the github repo: https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

Open cmake:

![](imageDirectory/cmake1.png)


![](imageDirectory/cmake2.png)

Set the correct location for source code and to the binary files. Select the boxes shown especially BUILD_PYTHON and the body_25 model as shown in image. Then click open project and be sure to generate it for windows 64x with visual studio 17 and build the solution. 
