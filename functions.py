"""
functions.py will contain helper functions used in the other Motion scripts
"""

import numpy as np
import imgPoseEstimation
import cv2
import math
import os
from pathlib import Path
import glob
from random import randint
from scipy import spatial

keyPointList = ['Nose', 'Neck', 'Right-Shoulder', 'Right-Elbow', 'Right-Wrist', 
                    'Left-Shoulder', 'Left-Elbow', 'Left-Wrist', 'Mid-Hips', 'Right-Hip',
                    'Right-Knee', 'Right-Ankle', 'Left-Hip', 'Left-Knee', 'Left-Ankle',
                    'Right-Eye', 'Left-Eye','Right-Ear', 'Left-Ear', 'Left-BigToe', 
                    'Left-SmallToe', 'Left-Heel', 'Right-BigToe', 'Right-SmallToe', 'Right-Heel',
                    'Background'
                    ]

"""Function will be called to test our functions"""
def test():
    imgFilePath = 'imageDirectory/duo.jpg'
    vidFilePath = 'videoDirectory/hindu-pressup.mp4'
    outputDirectoy = 'outputDirectory'
    stitchDirectory = 'outputDirectory/hindu-pressup_Output/Extracted_Frames'

    #extractVideoFrames(vidFilePath, outputDirectoy)
    #stitchFramesIntoVid(stitchDirectory, fps=30)
    #ting = APIPoseOnImage(imgFilePath)
    #print(getAngle(0,0,0))

    """
    #testing displayImage
    displayImg(imgFilePath)

    #testing pose from image
    show = 1 #1 to show output, 0 for just array
    poseList = poseFromImage(imgFilePath, show)
    print(poseList)

    pose = getPerson(poseList) 
    formattedPose = formatPose(pose)
    printFormattedKeyPointPositions(formattedPose)

    target = getTargetPose(poseList, imgDimensions)

    formattedTarget = formatPose(target)
    printFormattedKeyPointPositions(formattedTarget)

    #testing display video
    displayVideo(displayVideo(vidFilePath))

    pointA = 1,1
    pointB = 5,4
    print(getEuclideanDistance(pointA, pointB))
    """
#this function is for the analyser script and process an image and return desired data
def getFocalPose(imagePath):
    """
    process is as follows:
    1. get image dimensions
    2. call openpose python API to return list of poses
    3. find pose closest to centre of the image
    4. format the pose
    """
    imageDimensions = getImgDimensions(imagePath) 
    poseList = poseFromImage(imagePath)
    desiredPose = getTargetPose(poseList, imageDimensions)
    pose = formatPose(desiredPose)
    return pose
#this function draws points on each keypoint desired within the plotmap
def plotKeypointsOnImage(pose, imagePath, threshold, plotMap):
    image = cv2.imread(imagePath)
    points = [] #list to store detected points
    pointer = 0

    #loop through each point in our pose
    for point in pose:
        #conver to int to get coordinates
        x = int(float(point[0]))
        y = int(float(point[1]))
        confidence = point[2]

        if (float(confidence) > threshold and plotMap[pointer] == 1): 
            cv2.circle(image,(x,y), 5, (139,0,139), thickness = -1, lineType=cv2.FILLED)
            cv2.putText(image, "{}".format(pointer), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,215), 1, lineType=cv2.LINE_AA)
            points.append(point) #if point is over theshold we store
        else:
            points.append(None)
        pointer += 1
    cv2.imshow("Mapped-Keypoints", image)
    cv2.waitKey(0)
    return image
    cv2.destroyAllWindows()

   
def RunPoseEstOnFrames(filePath):
    #create a video capture object can pass 0 for a webcam input.
    cap = cv2.VideoCapture(filePath)

    #check if video is found
    if(cap.isOpened()==False):
        print("error: Couldn't open video file")
    
    #read until video ends
    while(cap.isOpened()):
        #capture each video object 
        ret, frame = cap.read()
    
        if ret == True:
            #display the resulting frame
            cv2.imshow('Frame', frame)

            #Press 'Q' key to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        #otherwise no need to execute loop
        else: 
            break
    
    #release the video object once done
    cap.release()

    #close all frames
    cv2.destroyAllWindows()

#This function is for returning the complete pose data array from an image
def poseFromImage(filePath):
    poses = imgPoseEstimation.getKeypointArray(filePath)
    return poses

#this function is for opening an image
def displayImg(filePath):
    image = cv2.imread(filePath)
    cv2.imshow('image', image)
    cv2.waitKey(0)

def displayWebcam():
    #create a webcam capture object can pass
    cap = cv2.VideoCapture(0)

    #check if video is found
    if(cap.isOpened()==False):
        print("error: Couldn't open video file")
    
    #read until video ends
    while(cap.isOpened()):
        #capture each video object 
        ret, video = cap.read()
    
        if ret == True:

            #display the resulting frame
            cv2.imshow('Video', video)

            #Press 'Q' key to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        #otherwise no need to execute loop
        else: 
            break
    
    #release the video object once done
    cap.release()

    #close all frames
    cv2.destroyAllWindows()

#this function captures video frames and displays the video
def displayVideo(filePath):
    #create a video capture object can pass 0 for a webcam input.
    cap = cv2.VideoCapture(filePath)

    #check if video is found
    if(cap.isOpened()==False):
        print("error: Couldn't open video file")
    
    #read until video ends
    while(cap.isOpened()):
        #capture each video object 
        ret, video = cap.read()
    
        if ret == True:

            #display the resulting frame
            cv2.imshow('Video', video)

            #any other processing can go inside of here

            #Press 'Q' key to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        #otherwise no need to execute loop
        else: 
            break
    
    #release the video object once done
    cap.release()

    #close all frames
    cv2.destroyAllWindows()

#returns the euclideanDistance between two points. 
def getEuclideanDistance(pointA, pointB):
    #euclidean distance is desfined by: squareRoot((x1-x2)^2 + (y1-y1)^2)
    return math.sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2 )

#here we can use euclidean distance to between each point then we can triangulate these points and apply the cosine rule to calculate the angle
#pointB should be centre point in which we find the angle opposite the side bounded by pointA and pointC 
def getAngle(pointA, pointB, pointC):
    
    sideA = getEuclideanDistance(pointA, pointB)
    sideB = getEuclideanDistance(pointB, pointC)
    sideC = getEuclideanDistance(pointA, pointC)

    angle = math.degrees(math.acos((sideA**2 + sideB**2 - sideC**2)/(2*sideA*sideB)))

    return int(angle)

#This function displays KeyPoints and their corresponding x-coord, y-coord, confidence
def printKeyPointPositions(poseArray):
    pointer = 0
    poseArray = getPerson(poseArray)
    for keypoint in keyPointList:
        print(keypoint + ": ", poseArray[pointer])
        pointer += 1

def printFormattedKeyPointPositions(formattedPose):
    pointer = 0
    for pointer in range(len(formattedPose)):
        outputString = "BodyPart: " + keyPointList[pointer] + "  X" , formattedPose[pointer][0] + "  Y", formattedPose[pointer][1] + "  Confidence: " + formattedPose[pointer][2]
        pointer += 1
        print(outputString)

#the python openpose API returns a list of lists for multiple person detection so this function will return one targeted persons pose
def getPerson(poseEstimationOutput, personIndex):
    poseArray = poseEstimationOutput[personIndex]
    return poseArray

#returns formatted y-coord, x-coord, confidence from [x y conf] --> [x, y, conf]
def formatKeyPoint(keyPoint):
    """
    the keypoint is passed as a single element so we will use the splitter (by whitespace) to return 3 elements 
    this format will be useful for data handling.
    """
    #convert to string and then use a slice to get rid of the '[' and ']' character
    keyPoint = str(keyPoint)[1:-1]
    formattedKeyPoint = keyPoint.split()
    
    return formattedKeyPoint

#will return relavent data to given location
def targetKeyPoint(poseList, location):
    return poseList[location]

#converts pose into a format which is easier for data manipulation
def formatPose(pose):
    formattedPose = []
    for keyPoint in pose:
        formattedKeyPoint = formatKeyPoint(keyPoint)
        formattedPose.append(formattedKeyPoint)
    return formattedPose

#we are going to target the pose positioned most centre of the camera view
def getTargetPose(poseList, imgDimensions):
    heightMid = imgDimensions[0]/2
    widthMid = imgDimensions[1]/2
    centrePoint = [heightMid,widthMid]

    target = 0
    distances = [] #this list will store the distances of each pose in the pose list from the centre

    pointer = 0
    #the distances will be calculated by the sum of each distance of each keypoint from the centre pixel
    """
    for pose in poseList:
        distance = 0
        for keyPoint in keyPointList:
            joint = formatKeyPoint(pose[pointer])
            coord = [float(joint[0]), float(joint[1])]
            distance = distance + getEuclideanDistance(coord, centrePoint)
        distances.append(distance)
        pointer += 1
    """
    focalPoints = [0,1,8] #just checking nose neck and mid hip
    for pose in poseList:
        distance = 0
        for keyPoint in keyPointList:
            joint = formatKeyPoint(pose[pointer])
            coord = [float(joint[0]), float(joint[1])]
            distance = distance + getEuclideanDistance(coord, centrePoint)
        distances.append(distance)
        pointer += 1

    pointer = 1
    smallestDistance = distances[0]
    smallestDistPos = 0
    #check the list and return the position with the shortest distance
    for pointer in range(len(distances)):
        if(distances[pointer] < smallestDistance):
            smallestDistance = distances[pointer]
            smallestDistPos = pointer
        pointer += 1

    targetPose = getPerson(poseList, smallestDistPos)
    return targetPose

#returns the dimensions of given image (height, width, number of channels)
def getImgDimensions(imgPath):
    image = cv2.imread(imgPath)
    imgDimensions = image.shape
    return imgDimensions

#returns the number of frames within given video
def getVideoLength(videoPath):
    cap = cv2.VideoCapture(videoPath) 
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

#This function will save every frame from a .mp4 file inside its own directory within the outputDirectory 
def extractVideoFrames(videoPath, outputDirectory):
    print('***Beginning Extraction Phase***')

    fileName = getFilename(videoPath)

    #check length of our video
    numberOfFrames = getVideoLength(videoPath)
    if numberOfFrames == 0:
        print("Length of video is 0 frames. Exiting extracting phase")
        return 0
    
    cap = cv2.VideoCapture(videoPath) #create our video capture object 
    frameCounter = 1

    #test the first frame
    ret, frame = cap.read() #ret is a boolean signaling if we sucessfully extracted frame

    if ret == 0:
        print("Failed to extract first frame. Exiting extracting phase")
        return 0
    
    saveDirectory = outputDirectory + "/" + fileName + "_Output"
    
    #creates new video directory for Motion ouputs if it doesn't already exist
    Path(saveDirectory).mkdir(parents=True, exist_ok=True)

    #creates directory for original extracted frames if it doesn't already exist
    saveDirectory = saveDirectory +  "/Extracted_Frames"
    Path(saveDirectory).mkdir(parents=True, exist_ok=True)


    testFilePath = os.path.join(
                saveDirectory, 
                fileName + "{}_{}.jpg".format("_frame",frameCounter))
    
    cv2.imwrite(testFilePath, frame) #writing image to desired location

    frameCounter += 1

    if os.path.isfile(testFilePath): #if test is succeeds... its go time
        print("Saving Test Frame Successful, Continuing Extraction Phase")

        while ret:
            ret, frame = cap.read() #check next frame
            if ret == False:
                break
            saveFramePath = os.path.join(saveDirectory,fileName + "{}_{}.jpg".format("_frame",frameCounter))
            cv2.imwrite(saveFramePath, frame)
            #print("Frame " , frameCounter , " saved.")
            #print(ret)
            frameCounter += 1
    else:
        print("Error: saving test frame unsuccessful.")
        return 0
    
    cap.release()
    print("***Extraction Complete***")

def stitchFramesIntoVid2(frameDirectory, saveName):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #output = cv2.VideoWriter('outputDirectory/'+saveName+'_Output/'+saveName +'.avi', fourcc, 10,(width,height))

    for filename in glob.glob(frameDirectory):
        frame = cv2.imread(filename)
        output.write(frame)
    
    output.release()

#returns name of the file without extension
def getFilename(filePath):
    #Here we split the full path into directory path + filename eg. vid/right/here/video.mp4 --> vid/right/here , video.mp4
    _, fileName = os.path.split(filePath)

    #remove extension eg. video.mp4 ==> video , .mp4
    fileNameWithoutExtension = os.path.splitext(fileName)[0]

    return fileNameWithoutExtension

#for some reason the ordering of the frames is not based on the order they are stored when looping through and using glob
#so this function will be used to return the desired order. I will base this off of a quicksort/merge sort as it will become more efficient 
# (relative to other sorts) as the number of frames increase. To me more memory efficient I will keep it in-place, meaning we wont produce duplicate lists
def sortFrames(frameList, low=0, high= None):
    if high == None: #we are inside the first call
        high = len(frameList) -1
        pIndex = partition(frameList, low, high) #partition around pivot
        sortFrames(frameList,low,pIndex-1) #sort lower half
        sortFrames(frameList, pIndex, high) #sort upper half
    elif low<high:
        pIndex = partition(frameList, low, high) #partition around pivot
        sortFrames(frameList,low,pIndex-1) #sort lower half
        sortFrames(frameList, pIndex, high) #sort upper half

def createArray(size = 10, max = 50):
    return [randint(0,max) for _ in range(size)]

#lomuto partitiion
def partition(a, low, high):
    i = low -1
    pivot = getFrameNumber(a[high])
    for j in range(low,high):
        if getFrameNumber(a[j]) <= pivot:
            i+=1
            a[i], a[j] = a[j], a[i]
    a[i+1], a[high] = a[high], a[i+1]
    return i+1

def getFrameNumber(frameName):
    if frameName ==0:
        return 0
    frameNumber='' #will convert to int at the end
    start = len(frameName) - 5
    for i in range(start, 0, -1): # -3 is because we dont need to bother with the .jpg at the end
        if frameName[i] == '_':
            return int(frameNumber)
        else:
            frameNumber = frameName[i] + frameNumber

def getImageFilenames(frameDirectory):
    images = [img for img in os.listdir(frameDirectory) if img.endswith(".jpg")] 
    return images

def stitchFramesIntoVid(frameDirectory, fps):

    #create a list of filenames for images with all .jpg filenames for our video frames.
    images = [img for img in os.listdir(frameDirectory) if img.endswith(".jpg")] 
    sortFrames(images) #order them sequentially 
    frame = cv2.imread(os.path.join(frameDirectory, images[0])) #get the first frame
    saveName = getFilename(images[0])
    size = len(saveName)
    saveName = saveName[:size-8] #cut out the "frame_1"
    height, width, layers = frame.shape

    #define codec to create videoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #XVID / DIVX
    output = cv2.VideoWriter('outputDirectory/'+saveName +'_Output/' + saveName +'poseEstimation.avi', fourcc, fps,(width,height))

    #loop through our frames
    for image in images:
        output.write(cv2.imread(os.path.join(frameDirectory, image)))
        #cv2.imshow('image',image)
        #cv2.waitKey(1)

    cv2.destroyAllWindows()
    output.release()

#scale, normalise, flatten the data contained within the pose

def SFNpose(pose):
    SFNpose = []
    #for storing the max and mix X and Y values assume they belong to first values
    xMax = float(pose[0][0])
    xMin = float(pose[0][0])

    yMax = float(pose[0][1])
    yMin = float(pose[0][1])

    xSum = float(0)
    ySum = float(0)

    #find the extreme X,Y values
    for point in pose:
        currX = float(point[0])
        currY = float(point[1])

        xSum = float(xSum) + currX
        ySum = float(ySum) + currY

        if (currX > xMax):
            xMax = currX
        elif(currX < xMin):
            xMin = currX

        if (currY > yMax):
            yMax = currY
        elif(currY < yMin):
            yMin = currY
    
    #now we have our extremes we can work out height, width, and scale factor
    width = float(xMax - xMin)
    height = float(yMax - yMin)

    #scalefactor = max(width,height)
    if height > width:
        scaleFactor = height
    else:
        scaleFactor = width

    #find mean of x and y values
    xMean = xSum / float(len(pose))
    yMean = ySum / float(len(pose))

    #create new scaled pose
    #returning a centred pose at 0,0 scaled and normalised between -0.5 -> 0.5
    for point in pose:
        x = float(point[0])
        y = float(point[1])

        x = (x-xMean)/scaleFactor
        y = (y-yMean)/scaleFactor  

        SFNpose.append((x,y))

    return SFNpose

def getCoordinatesSFN(sfnPose, location):
    x = sfnPose[location][0]
    y = sfnPose[location][1]
    return (float(x), float(y))

#vectorises each keypoint connection and returns a list of vectors
def vectorise(sfnPose):
    keypointConnections = [[1, 0], [1, 2], [1, 5], 
                            [2, 3], [3, 4], [5, 6], 
                            [6, 7], [0, 15], [15, 17], 
                            [0, 16],[16, 18], [1, 8],
                            [8, 9], [9, 10], [10, 11], 
                            [11, 22], [22, 23], [11, 24],
                            [8, 12], [12, 13], [13, 14], 
                            [14, 19], [19, 20], [14, 21]]
    vectorList = []
    for connection in keypointConnections:
        point1 = getCoordinatesSFN(sfnPose, connection[0])
        point2 = getCoordinatesSFN(sfnPose, connection[1])

        vector = ((point2[0] - point1[0]), (point2[1]- point1[1]))
        vectorList.append(vector)
    return vectorList

def getGradient(pointA, pointB):
    rise = pointB[1] - pointA[1]
    run = pointB[0] - pointA[0]
    return rise/run

def getFloatCoords(pose, position):
    x = pose[position][0]
    y = pose[position][1]
    return (float(x), float(y))

#finds the cumulative distance between a poses key points using euclidean distance to another pose
def getPoseDistance(pose1, pose2):
    cumDist = 0 #cumulative distance
    #loop through the poses
    for i in range(len(pose1)):
        #get the coordinates of each point and find the distances between them
        currPoint = getFloatCoords(pose1, i)
        refPoint = getFloatCoords(pose2, i)

        dist = getEuclideanDistance(currPoint, refPoint)

        #add to our distance
        cumDist = cumDist + dist
    
    return cumDist

#takes two equal length sequences and returns a single number representing the dot product
def dotProduct(listA, listB):
    sum = 0 
    for i in range(len(listA)):
        sum += listA[i] * listB[i] #sum of the product between each corresponding points in the lists
    return sum #dot product 

def vectorNorm(vector):
    norm = 0
    for i in range(len(vector)):
        norm +=[i]**2 #square each element in vector for a non negative value
    return norm**0.5 #return square root for norm

#1-(dotproduct/vectorNorm)
def cosineSimilarity(vectorList1, vectorList2):
    simList = []
    #workout the cosine similarity for each corresponding vector 
    for i in range(len(vectorList1)):
        x = vectorList1[i]
        y = vectorList2[i]
        cosineSim = 1 - spatial.distance.cosine(x,y)
        simList.append(cosineSim)
    totalSim = 0
    #return the average similarity for all vectors
    for sim in simList:
        totalSim +=  sim
    
    averageSim = totalSim / len(simList)
    return averageSim

#pythagorean theorem
def vectorMagnitude(vector):
    x = vector[0]
    y = vector[1]
    return ((x**2) + (y**2))**0.5

#returns list of unique values
def removeDuplicates(list):
    return [i for j, i in enumerate(list) if i not in list[:j]]

test()