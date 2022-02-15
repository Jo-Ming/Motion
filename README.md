# Motion

## Abstract

Often today it is very common to measure somebody’s health and wellbeing solely on the basis of speed and strength. Although these are important, often for sport/performance, there is another dimension of flexibility and mobility which are strong contributors to longevity/injury-prevention. One major hurdle for people is that flexibility and mobility take far longer build and train (especially if your biology is that of a male!), and it is easy for a person to fall into the trap of only training what they are good at which is often very inefficient. 

Motion is a program which aims to produce a tailored program for users to effectively improve flexibility and mobility effectively. Motion implements the CMU OpenPose 1.7 Body_25 model for an in-depth analysis of an overhead squat. This involves using OpenCV for getting a video file input, extracting frames, stitching frames, saving frames as .jpg files, manipulating output data from the model to find Euclidean distances, angles of joints, vectorising points in 2-Dimensional space, and plotting the poses onto images in order to analyse a user’s overhead squat alongside dabbling in pose similarity and pose classifications using KNN and CNN approaches using Tensorflow and Keras libraries.

This has become especially relevant with the happening of Covid-19 pandemic in 2020 which affected everyone. Throughout this period, prevented from socialising, discouraged from going outside, businesses are shut, gyms are closed, and takeaways are cheap. It becomes very easy for people to forget to look after themselves. Especially where people are expected to carry on with all usual responsibilities such as a professional or University work. At Universities, the continually increasing emphasis on mental health from students has become more significant, applying more pressure and responsibilities than ever before onto these institutes. On the scale of a community, this can be a seemingly impossible and increasingly complicated problem to manage. To a computer scientist, it is understood that the solution to any problem is also the sum of the solutions to all its counterparts. Even if somebody could not solve the entire problem, targeting some of the problem may bring us much closer to a complete solution. Motion wishes to try and help some of this problem on the scale of the individual. In England it is estimated that 1 in 4 people will suffer from mental health problems each year (Mind.org.uk, 2020) 17% of these are in the forms of anxiety and depression. People who took part in a recent Harvard study has showed that by introducing a form of minor activity daily such an hour of walking, stretching, or any kind of low intensity movement can reduce risk of major depression by 26% and on top of that significantly help individuals from relapsing (Harvard, 2019). This does however, require individuals to take some responsibility for themselves. 

In conclusion, the CMU model has made significant progress over the last 4 years (since 2017 release) and is very precise from a front on perspective. The model is only reliable when the user is square to the camera, as human pose estimation continues to improve, hopefully one day it will have the reliability to be implemented into an application like Motion.

### Overview of human Pose Estimation 

Carnegie Mellon University are among the forefront of AI research and in this case Human Pose Estimation. OpenPose was released in October 2017 and has since had frequent and major improvements. Best of all, it is freely available to all non-commercial use, alongside great documentation. It was not until later that it was decided to use the CMU body_25 model, but for background knowledge after reading from the official OpenPose cited documentation (Zhe Cao, 2018) and sources online (Tanugraha, 2019) , (Raj, 2019). Here is a brief overview.

There are two main approaches to this problem, first is the simpler top-down approach where the model is trained to detect the human first (which is relatively simple) and then deriving each key point from the humans detected. Simply, detect each person then derive body parts from each body for every person. 

The second approach is the bottom-up approach which OpenPose uses. This is where each body part belonging to everyone is detected first and then grouped and associated together to form the person. Simply, locate everybody part in frame then using grouping methods such as part affinity fields and part confidence maps stitch each human together. Using this method, performance seems to suffer far less as more people are detected in frame.

Confidence maps are a 2D representation of the belief that a particular key point occurs in each given pixel. If there is one person in the image, a single peak should exist in each confidence map if the corresponding part is visible.

Part Affinity fields for part association needs a confidence value measuring the association of each pair of body parts belonging to which person within an image. This preserves the location and orientation information across the region supporting the limb (body part/key point). Represented by a 2-Dimensional vector field for each limb, pointing from one part of the limb to the other. Each limb has a part affinity field directing to associated body parts.


#### Demo
will do some more detailed documentation explaining how ot works and the maths behind it another day.

![](imageDirectory/dataflow.png)

![](imageDirectory/exampleInput.png)

![](imageDirectory/skeleton.png)

![](imageDirectory/allAngles.png)

![](imageDirectory/splits.png)

![](imageDirectory/squatAngles.png)

![](imageDirectory/outputExample.png)




##### Environment Set Up

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
