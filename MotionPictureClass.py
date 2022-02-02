"""
This object called a still motion will be used to work on frames within a video
A motion will be made up of still motions.
"""
import os
import cv2
import functions as fun #contains helper functions
import math
from pathlib import Path
import os

class MotionPicture:
    
    #to construct itself
    def __init__(self, dimensions, imagePath, threshold):

        #This list is according to the output of the python api for the openpose mpi body_25 model
        self.keypoints = ['Nose', 'Neck', 'Right-Shoulder', 'Right-Elbow', 'Right-Wrist', 
                    'Left-Shoulder', 'Left-Elbow', 'Left-Wrist', 'Mid-Hips', 'Right-Hip',
                    'Right-Knee', 'Right-Ankle', 'Left-Hip', 'Left-Knee', 'Left-Ankle',
                    'Right-Eye', 'Left-Eye','Right-Ear', 'Left-Ear', 'Left-BigToe',
                    'Left-SmallToe', 'Left-Heel', 'Right-BigToe', 'Right-SmallToe', 'Right-Heel',
                    'Background'
                    ]
        #contains pointers for which ley points are connected to eachother, this is useful for drawing poses
        self.keypointConnections = [[1, 0], [1, 2], [1, 5], 
                                    [2, 3], [3, 4], [5, 6], 
                                    [6, 7], [0, 15], [15, 17], 
                                    [0, 16],[16, 18], [1, 8],
                                    [8, 9], [9, 10], [10, 11], 
                                    [11, 22], [22, 23], [11, 24],
                                    [8, 12], [12, 13], [13, 14], 
                                    [14, 19], [19, 20], [14, 21]]
        #is derived from keypointConnections but holds list with pointers for working out angles. A joint is which keyparts relate to eachother when moving
        self.joints = [ [0,1,2],[0,1,5],[1,2,3],
                        [1,5,6],[2,3,4],[5,6,7],
                        [1,8,12],[1,8,9],[8,12,13],
                        [8,9,10],[9,10,11],[12,13,14],
                        [13,14,20],[10,11,23],[11,8,14]]

        #A dictionary containing joints with pointers to their loction/focal 
        self.jointNames =  ['HeadTilt-Right','Headtilt-left','Right-Shoulder',
                                 'Left-Shoulder','Right-Elbow','Left-Elbow',
                                 'LowerBack-Left','LowerBack-Right','Left-Hip',
                                 'Right-Hip','Right-Knee', 'Left-Knee' ,
                                 'Left-Ankle','Right-Ankle']

        self.imagePath = imagePath #path for image 
        self.name = self.setFileName() #name of image
        self.image = self.setImage() #The actual image itself
        self.canvas = self.setCanvas() #blank image
        self.skeleton = self.setImage() #will be a canvas to draw on
        self.dimensions = dimensions #dimensions of image: (height, width, channels) #we made it so that the superclass passes dimension
        self.pose = self.setPose() # sets attribute containing the poseArray
        self.valid = True #assume we can use this frame for analysis
        self.threshold = threshold 

        if self.pose != 0: #only call other functions if pose is detected
            self.confPoints = self.confidenceCheck() #filter estimates by confidence
            self.angleList = self.findAngles()
            self.SFNpose = self.getSFNpose() #scalefactor and normalise
        else:
            self.valid = False #we can't use this frame

    """
    #to destroy itself
    def __del__(self):
        print(self.name + ' destroyed...')
    """
    
    def getSkeleton(self):
        return self.skeleton
    
    #sets the name attribute
    def setFileName(self):
        #Here we split the full path into directory path + filename eg. vid/right/here/video.mp4 --> vid/right/here , video.mp4
        path = self.imagePath
        _, fileName = os.path.split(path)
        return fileName
    #sets the image attribute
    def setImage(self):
        image = cv2.imread(self.imagePath)
        return image

    def setCanvas(self):
        image = cv2.imread("imageDirectory/canvas.jpg")
        return image

    #gets name
    def getName(self):
        return self.name
    
    #displays the image
    def showImage(self, title):
        image = cv2.imread(self.imagePath)
        cv2.imshow(title, image)
        cv2.waitKey(0)
    
    #returns x and y coords as integers
    def getCoordinates(self, position):
        x = self.pose[position][0]
        y = self.pose[position][1]
        return (int(float(x)), int(float(y)))

    #returns SFN x and y coords as floats
    def getSFNCoordinates(self, position):
        x = self.SFNpose[position][0]
        y = self.SFNpose[position][1]
        return (float(x), float(y))
    
    def getAngle(self, position):
        try:
            return self.angleList[position][1]
        except:
            print("Angle unable to be calculated")
            return 0
    
    #this function get the desired pose from its self.image
    def setPose(self):
        """
        process is as follows:
        1. call openpose python API to return list of poses
        2. find pose closest to centre of the image
        3. format the pose
        """
        poseList = fun.poseFromImage(self.imagePath)
        #find centre of frame
        heightMid = self.dimensions[0]/2
        widthMid = self.dimensions[1]/2

        centrePoint = [heightMid,widthMid] 
        desiredPose = self.findTarget(poseList, centrePoint)

        try:
            pose = fun.formatPose(desiredPose)
        except:
            self.valid = False #motionPicture cant be used for analysis
            return 0
        
        return pose 
    
    #returns the pose
    def getPose(self):
        return self.pose
    
    #print the pose points in console in an easily readable format
    def printKeypoints(self):
        pointer = 0
        for pointer in range(len(self.pose)):
            outputString = "BodyPart: " + self.keypoints[pointer] + "  X" , self.pose[pointer][0] + "  Y", self.pose[pointer][1] + "  Confidence: " + self.pose[pointer][2]
            pointer += 1
            print(outputString)
    
    #print all joint angles into console
    def printJointAngles(self):
        pointer = 0
        for joint in self.jointNames:
            try:
                angle = self.angleList[pointer][1]
                outputString = 'joint: ' + joint + '  Angle: ' , angle
            except:
                outputString = 'joint: ' + joint + '  Angle: Not Found (confidence below Threshold)'
            pointer += 1
            print(outputString)

    #This routine creates an attribute which filters out positions with confidence values below our threshold
    def confidenceCheck(self):
        pose = self.pose
        confPose = []
        for point in pose:
            x = point[0]
            y = point[1]
            confidence = float(point[2])
            if confidence < self.threshold: #if the confidence is less than our threshold then sack it
                confPose.append(None)
            else:
                confPose.append((x,y))
        return confPose

    #this function will draw the skelton bt connecting the keypoints using straight lines
    def drawSkeleton(self, showAngles):
        image = self.image #get our image to draw onto
        #image = self.canvas #get our image to draw onto
        pointer = 0

        #go through the connections
        for pair in self.keypointConnections:
            #if both points are in the list then we are confident enough to draw it onto our image
            if self.confPoints[pair[0]] != None and self.confPoints[pair[1]] != None:
                #set our pointers for keypoints
                pointA = int(pair[0])
                pointB = int(pair[1])

                #get coordinates for the first keypoint
                x1 = int(float(self.confPoints[pointA][0]))
                y1 = int(float(self.confPoints[pointA][1]))

                #get coordinates for our second keypoint 
                x2 = int(float(self.confPoints[pointB][0]))
                y2 = int(float(self.confPoints[pointB][1]))

                if self.confPoints[pointA] and self.confPoints[pointB]:       
                    #draw a line connecting point A to point B
                    cv2.line(image, (x1,y1), (x2,y2), (128,0,128), 3, lineType=cv2.FILLED)
                    #draw a circle at keypointA
                    cv2.circle(image, (x1,y1), 4, (170,108,57), thickness = -1, lineType=cv2.FILLED)
                if showAngles == True:
                    image = self.drawAngles(image)
        self.skeleton = image
    
    #draw the calculated angles onto our image
    def drawAngles(self, image):
        for item in self.angleList:
            if item != None: #if pose is detected
                position = item[0]
                angle = item[1]

                x = int(float(self.confPoints[position][0]))
                y = int(float(self.confPoints[position][1]))

                #write angle at corresponding point
                cv2.putText(image, "{}".format(angle), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 0, lineType=cv2.LINE_AA)
        return image

    #only write desired angles onto image
    def drawSelectedAngles(self, positions):
        image = self.image
        for point in positions:
            item = self.angleList[point]

            position = item[0]
            angle = item[1]

            x = int(float(self.confPoints[position][0]))
            y = int(float(self.confPoints[position][1]))

            #write angle at corresponding point
            cv2.putText(image, "{}".format(angle), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,216,106), 2, lineType=cv2.LINE_AA)
        return image

    #calculates the angles at every joint
    def findAngles(self):
        angleList=[] #empty list to store angles and return will be stored in format (pointer, angle)

        for joint in self.joints: 
            """
            the process will be a s follows
            joint = pointA, pointB, pointC
            1.get the coordinates of each point
            2. triangulate by finding the distance between each point
            3. use laws of cosine to find angle

            pointB should be centre point in which we find the angle opposite the side bounded by pointA and pointC 
            """

            pointA, pointB, pointC = joint
            
            #get positions of each corner 
            coordA = self.confPoints[pointA]
            coordB = self.confPoints[pointB]
            coordC = self.confPoints[pointC]

            #If the model is confident enough to locate each vertici ie. not None post filtration
            if (coordA != None and coordB != None and coordC != None):

                #get x, y coordinates for each vertex as integers for cv drawing functions
                Ax = int(float(coordA[0]))
                Ay = int(float(coordA[1]))

                Bx = int(float(coordB[0]))
                By = int(float(coordB[1]))

                Cx = int(float(coordC[0]))
                Cy = int(float(coordC[1]))

                #workout the magnitude of shortest path separating each point to triangulate
                sideA = fun.getEuclideanDistance((Ax,Ay), (Bx,By))
                sideB = fun.getEuclideanDistance((Bx,By), (Cx,Cy))
                sideC = fun.getEuclideanDistance((Ax,Ay), (Cx,Cy))

                #rearrange cosine rule to get angle as subject
                angle = int(math.degrees(math.acos((sideA**2 + sideB**2 - sideC**2)/(2*sideA*sideB))))
                angleList.append((pointB, angle))
            else:
                #not confident enough to look for angle
                angleList.append(None)

        return angleList
    
    #vectorises each keypoint connection and returns a list of vectors
    def vectorise(self):
        vectorList = []
        for connection in self.keypointConnections:
            point1 = self.getCoordinates(connection[0])
            point2 = self.getCoordinates(connection[1])

            vector = ((point2[0] - point1[0]), (point2[1]- point1[1]))
            vectorList.append(vector)
        return vectorList

    #saves the processed image into our processed frames directory
    def saveSkeleton(self, outputPath, shortName):
        if self.pose == 0:
            return 0 #no pose in frame
        editedImage = self.skeleton 
        saveDirectory = outputPath + "/" + shortName + "_Output"

        #creates new video directory for Motion ouputs if it doesn't already exist
        Path(saveDirectory).mkdir(parents=True, exist_ok=True)

        #creates directory for original extracted frames if it doesn't already exist
        saveDirectory = saveDirectory +  "/processed_Frames/"
        Path(saveDirectory).mkdir(parents=True, exist_ok=True)

        filePath = os.path.join(saveDirectory + self.name)

        cv2.imwrite(filePath, editedImage) #writing image to desired location

    #looks for pose most central to cameraview
    def findTarget(self, poseList, centrePoint):
        referencePoints = [0,1,8] #instead of checking everypoint we will use reference points [Nose, Neck, Mid-Hips]

        distances = [] #this list will store the distances of each pose in the pose list from the centre

        pointer = 0
        #the distances will be calculated by the sum of each distance of each keypoint from the centre pixel

        try:
            for pose in poseList:
                distance = 0
                for keyPoint in referencePoints:
                    joint = fun.formatKeyPoint(pose[pointer])
                    coord = [float(joint[0]), float(joint[1])]
                    distance = distance + fun.getEuclideanDistance(coord, centrePoint)
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

            targetPose = fun.getPerson(poseList, smallestDistPos)
            return targetPose

        except:
            return None #nobody detected
    
    #scale, normalise, flatten the data contained within the pose
    def getSFNpose(self):
        SFNpose = []
        pose = self.pose
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
