import tensorflow as tf
import functions as fun
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D
import numpy as np
from sklearn.metrics import accuracy_score
from keras.layers import BatchNormalization

def main():
    classes = ['Cobra Pose', 'Downward Dog', 'Squat', 'Stand']
    # classes 0,1,2 = cobra, ddoggy, squat
    dataflow, classList = getDataflow('TrainingImages')

    for data in dataflow: #ignore confidence
        data = data[0:1]
    
    dataflow = np.array(dataflow)
    classList = np.array(classList)

    #This awesome function will randomly divide the data into training and test data
    trainingData, testingData, trainingClasses, testingClasses = train_test_split(dataflow, classList, test_size = 0.2, random_state = 1)

    trainingData = np.array(trainingData).reshape(len(trainingData), 25, 2)
    testingData = np.array(testingData).reshape(len(testingData), 25, 2)

    runConvNN(trainingData, testingData, trainingClasses, testingClasses)
    knnPrediction = knnClassifier(testingData[1], dataflow, classList, 3)
    print(knnPrediction)

def getDataflow(trainingDirectory):
    dataflow = []#to store our dataflow of poses
    classList = [] # to store corresponding class/label
    #get the names of each jpg
    images = [img for img in os.listdir(trainingDirectory + '/CobraPose') if img.endswith(".jpg")] #get the name of files
    #get poses for each cobra image
    for cobra in images:
        pose = fun.getFocalPose(trainingDirectory + '/CobraPose/' + cobra) #find our desired subject
        SFNpose = fun.SFNpose(pose) #scale factor, centre, and normalise the pose this way height and width of person isnt important
        dataflow.append(SFNpose) #poses should always have a consistent shape from the output of mpi body_25 model
        classList.append(0)

    #get data for D-Doggy poses
    images = [img for img in os.listdir(trainingDirectory + '/DownwardDog') if img.endswith(".jpg")] #get the name of files
    #get poses for each dog
    for DDoggy in images:
        pose = fun.getFocalPose(trainingDirectory + '/DownwardDog/' + DDoggy)
        SFNpose = fun.SFNpose(pose) #pre processing the pose. SFNpose = Scale factored Normalised pose
        dataflow.append(SFNpose)
        classList.append(1)
    
    images = [img for img in os.listdir(trainingDirectory + '/Squat') if img.endswith(".jpg")] #get the name of files
    for squat in images:
        pose = fun.getFocalPose(trainingDirectory + '/Squat/' + squat) 
        SFNpose = fun.SFNpose(pose) #pre processing the pose. SFNpose = Scale factored Normalised pose
        dataflow.append(SFNpose)
        classList.append(2)
    
    images = [img for img in os.listdir(trainingDirectory + '/Stand') if img.endswith('.jpg')]
    for Stand in images:
        pose = fun.getFocalPose(trainingDirectory + '/Stand/' + Stand)
        SFNpose = fun.SFNpose(pose) #pre processing the pose. SFNpose = Scale factored Normalised pose
        dataflow.append(SFNpose)
        classList.append(3)
    return dataflow, classList

def runConvNN(trainingData, testingData, trainingClasses, testingClasses):
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    
    model.add(Conv1D(filters = 16, kernel_size= 3, input_shape = (25,2), activation = 'relu')) #add a convolutional layer to our model

    model.add(BatchNormalization())
    
    model.add(tf.keras.layers.Dense(68, activation=tf.nn.relu)) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(68, activation=tf.nn.relu)) #this is the first dense layer using the recified linear activation function
    
    model.add(tf.keras.layers.Dense(10, activation = 'sigmoid')) # output layer using 10 nodes (because there are 10 outputs 0-9)
    
    model.add(tf.keras.layers.Flatten()) #this is to make the input layer in 1 dimension.
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(trainingData, trainingClasses, batch_size = 1, epochs = 15) #Training our model
    
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    print("convmodel saved.")
    model.save('motionCNN.model') #save the model
    
def knnClassifier(inputPose, targetPoseDataflow, classList, k):
    distances = [] #list to append distances to

    #lets use a dictionary to make getting neighbour classes easier at the end
    distClassDict = {} #will hold the distance from our input to target pose and class of target
    pointer = 0
    #for every example pose check the similarity using euclidean distance
    for targetPose in targetPoseDataflow:
        #add distances between each keypoint 
        dist = fun.getPoseDistance(inputPose, targetPose) #find distance
        distances.append(dist) #add to our list of distances
        #add to our
        distClassDict[dist]= classList[pointer]
        pointer += 1
    
    distances = sortDist(distances)

    #now we can get the nearest k classes of neighbours from our sorted list
    kNeighbourClasses = []
    for i in range(k):
        kNeighbourClasses.append(distClassDict[distances[i]])
    
    #now we can classify by using the majority vote of this class list
    print(distances)
    print(distClassDict)
    print(kNeighbourClasses)

    classprediction = max(set(kNeighbourClasses), key=kNeighbourClasses.count) #take the modal value
    return classprediction

#modified thwe sortFrames() function from functions.py to sort the distance values
#by not creating 3 list each iteration we can make the quicksort faster for smaller lists
def sortDist(distList, low=0, high= None):
    if high == None: #we are inside the first call
        high = len(distList) - 1
        pIndex = partition(distList, low, high) #partition around pivot
        sortDist(distList,low,pIndex-1) #sort lower half
        sortDist(distList, pIndex, high) #sort upper half
    elif low<high:
        pIndex = partition(distList, low, high) #partition around pivot
        sortDist(distList,low,pIndex-1) #sort lower half
        sortDist(distList, pIndex, high) #sort upper half
    return distList

#lomuto partitiion
def partition(a, low, high):
    i = low -1
    pivot = a[high]
    for j in range(low,high):
        if a[j] <= pivot:
            i+=1
            a[i], a[j] = a[j], a[i]
    a[i+1], a[high] = a[high], a[i+1]
    return i+1

main()