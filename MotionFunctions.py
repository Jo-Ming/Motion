import numpy as np
import imgPoseEstimation
import cv2
import math
import os
from pathlib import Path
import glob
from random import randint

keyPointList = ['Nose', 'Neck', 'Right-Shoulder', 'Right-Elbow', 'Right-Wrist', 
                    'Left-Shoulder', 'Left-Elbow', 'Left-Wrist', 'Mid-Hips', 'Right-Hip'
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
    print(getAngle(0,0,0))

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

#This function is for returning the keypoint array from an image
def poseFromImage(filePath):
    pose = imgPoseEstimation.getKeypointArray(filePath, show)
    return pose

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

#returns the euclideanDistance between two points. By triangulating 2 coordinates we can use pythagorus to find the distance (hypoteneuse) between them
def getEuclideanDistance(pointA, pointB):
    #euclidean distance is desfined by: squareRoot((x1-x2)^2 + (y1-y1)^2)
    return math.sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2 )

#here we can use euclidean distance to between each point then we can triangulate these points and apply the cosine rule to calculate the angle
#pointB should be centre point in which we find the angle opposite the side bounded by pointA and pointC 
def getAngle(pointA, pointB, pointC):
    
    #sideA = getEuclideanDistance(pointA, pointB)
    #sideB = getEuclideanDistance(pointB, pointC)
    #sideC = getEuclideanDistance(pointA, pointC)

    sideA = 10
    sideB = 10
    sideC = 10

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
    for pointer in range(len(keyPointList)):
        outputString = "BodyPart: " + keyPointList[pointer] + "  X: " , formattedPose[pointer][0] + "  Y: ", formattedPose[pointer][1] + "  Confidence: " + formattedPose[pointer][2]
        pointer += 1
        print(outputString)

#the python openpose API returns a list of lists for multiple person detection so this function will return one targeted persons pose
def getPerson(poseEstimationOutput, personIndex):
    poseArray = poseEstimationOutput[personIndex]
    return poseArray

#returns formatted y-coord, x-coord, confidence from [y x conf] --> [y, x, conf]
def formatKeyPoint(keyPoint):
    """
    the keypoint is passed as a single element so we will use the splitter (by whitespace) to return 3 elements 
    this format will be useful for data handling.
    for some reason the API returns the Y axis First 
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
        pIndex = partition(frameList, low, high) #partition arounf pivot
        sortFrames(frameList,low,pIndex-1) #sort lower half
        sortFrames(frameList, pIndex, high) #sort upper half
    elif low<high:
        pIndex = partition(frameList, low, high) #partition arounf pivot
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
    output = cv2.VideoWriter('outputDirectory/'+saveName +'_Output/' + saveName +'poseEstmation.avi', fourcc, fps,(width,height))

    #loop through our frames
    for image in images:
        output.write(cv2.imread(os.path.join(frameDirectory, image)))
        #cv2.imshow('image',image)
        #cv2.waitKey(1)

    cv2.destroyAllWindows()
    output.release()

test()