from MotionPictureClass import MotionPicture
import cv2
import os
from pathlib import Path
import functions as fun

class Motion:

    def __init__(self, name, videoPath, threshold):

        self.name = name
        self.videoPath = videoPath
        self.threshold = threshold
        self.outputDirectory = 'outputDirectory'
        self.shortName = self.getNameWithoutExt()
        self.length = self.getLength()
        #self.motionPictureList = self.codifyMotion() #will be made up of the list of MotionPictures for each frame  
    
    def __del__(self):
        print(self.name + ' destroyed...')
    
    def getLength(self):
        cap = cv2.VideoCapture(self.videoPath) #create our video capture object 

        numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
        length = numberOfFrames #setting length of the video in frames
        return length
    
    def getNameWithoutExt(self):
        filePath = self.videoPath
        #Here we split the full path into directory path + filename eg. vid/right/here/video.mp4 --> vid/right/here , video.mp4
        _, fileName = os.path.split(filePath)

        #remove extension eg. video.mp4 ==> video , .mp4
        fileNameWithoutExtension = os.path.splitext(fileName)[0]

        return fileNameWithoutExtension                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

    def playVideo(self):

        filePath = self.videoPath
        #create a video capture object can pass 0 for a webcam input.
        cap = cv2.VideoCapture(filePath)

        #check if video is found
        if(cap.isOpened()==False):
            print("error: Couldn't open video file for: " + self.name)
        
        #read until video ends
        while(cap.isOpened()):
            #capture frame  
            ret, frame = cap.read()
        
            if ret == True:

                #display the resulting frame
                cv2.imshow('Video', frame)

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


    #will create MotionPictures for each frame also extracting and saving each frame
    def codifyMotion(self, showAngles):

        print('*****Preparing To Analyze Motion*****')
        #get attributes
        videoPath = self.videoPath
        shortName = self.shortName
        threshold = self.threshold

        cap = cv2.VideoCapture(videoPath) #create our video capture object 
        frameCounter = 1

        numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
        numberOfFrames = self.length #getting length of the video in frames

        motionPictureList = [] #to store motionPicture objects
    
        print('****Beginning Extraction Phase****')

        if numberOfFrames == 0:
            print("Length of video is 0 frames. Exiting extracting phase")
            return 0
    
        #test the first frame
        ret, frame = cap.read() #ret is a boolean signaling if we sucessfully extracted frame
        self.frameDimensions = frame.shape #we can check once here and pass it through to MotionPictures instead of having them call cv2.imread() an extra time for each frame

        if ret == 0:
            print("Failed to extract first frame. Exiting extracting phase")
            return 0
        
        saveDirectory = self.outputDirectory + "/" + self.shortName + "_Output"
        
        #creates new video directory for Motion ouputs if it doesn't already exist
        Path(saveDirectory).mkdir(parents=True, exist_ok=True)

        #creates directory for original extracted frames if it doesn't already exist
        saveExtractedDirectory = saveDirectory +  "/Extracted_Frames" #final ouput for each frame will be stroed in here
        Path(saveExtractedDirectory).mkdir(parents=True, exist_ok=True)

        #creates directory for original extracted frames if it doesn't already exist
        saveProcessedDirectory = saveDirectory +  "/Processed_Frames" #final ouput for each frame will be stroed in here
        Path(saveProcessedDirectory).mkdir(parents=True, exist_ok=True)
        self.outputFrameDirectory = saveProcessedDirectory 

        #save the original frame first
        testExtractedFilePath = os.path.join(
                    saveExtractedDirectory + '/' +
                    shortName + "{}_{}.jpg".format("_frame",frameCounter))
        
        cv2.imwrite(testExtractedFilePath, frame) #writing original image to desired location

        testProcessedFilePath = os.path.join(
                    saveProcessedDirectory + '/' +
                    shortName + "{}_{}.jpg".format("_frame",frameCounter))

        testMotionPicture = MotionPicture(frame.shape, testExtractedFilePath, threshold) #use the just now extracted frame to create MotionPicture Object
        testMotionPicture.drawSkeleton(showAngles)
        testMotionPicture.saveSkeleton(self.outputDirectory, self.shortName) #save the processed image
        motionPictureList.append(testMotionPicture)

        frameCounter += 1

        if os.path.isfile(testProcessedFilePath): #if test is succeeds... its go time
            print("*****Test Frame Successful, Continuing Motion Analysis*****")
            #del testMotionPicture;

            while ret:
                ret, frame = cap.read() #check next frame
                if ret == False: 
                    break # this would mean we were unable to extract the next frame

                #save the original frame
                saveExtractedFramePath = os.path.join(saveExtractedDirectory + '/' + shortName + "{}_{}.jpg".format("_frame",frameCounter))
                cv2.imwrite(saveExtractedFramePath, frame)
                
                #process and save image
                saveProcessedFilePath = os.path.join(
                saveProcessedDirectory + '/' +
                shortName + "{}_{}.jpg".format("_pose_frame",frameCounter))

                #create object
                frameObject = 'MotionPicture' + str(frameCounter)

                frameObject = MotionPicture(self.frameDimensions, saveExtractedFramePath, threshold)
                frameObject.drawSkeleton(showAngles)
                frameObject.saveSkeleton(self.outputDirectory, self.shortName)
                #add it to list
                motionPictureList.append(frameObject)

                frameCounter += 1 #increment our counter
        else:
            print("Error: test frame unsuccessful.")
            return 0
        
        cap.release()
        print("***Extraction Complete***")
        self.motionPictureList =  motionPictureList

    #this function will stitch together each frame image in a direcorty into a video
    def stitchDirectory(self, fps):

        saveDirectory = self.outputDirectory + "/" + self.shortName + "_Output"
        saveProcessedDirectory = saveDirectory +  "/Processed_Frames"

        #create a list of filenames for images with all .jpg filenames for our video frames.
        images = [img for img in os.listdir(saveProcessedDirectory) if img.endswith(".jpg")] 
        fun.sortFrames(images) #order them sequentially 
        frame = cv2.imread(os.path.join(saveProcessedDirectory, images[0])) #get the first frame
        saveName = self.name
        size = len(saveName)
        height, width, layers = frame.shape

        #define codec to create videoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID') #XVID / DIVX
        output = cv2.VideoWriter('outputDirectory/'+ self.shortName +'_Output/' + self.shortName +'poseEstmation.avi', fourcc, fps,(width,height))

        #loop through our frames
        for image in images:
            output.write(cv2.imread(os.path.join(saveProcessedDirectory, image)))

        cv2.destroyAllWindows()
        output.release()
    
    #this method returns the motionPicture where the desired keypoint is at its lowest throughout this motion
    def getAtLowest(self, Keypoint):
        
        position = 0
        lowest = int(self.frameDimensions[1]) #take the max Y coord to start
        lowPos = 0
        for motionPicture in self.motionPictureList:
            currHeight = int(float(motionPicture.pose[Keypoint][1])) #[keypoint] is the position [1] is the y coord
            if currHeight > lowest: #numpy array the y coord moves top downwards so the lowest point is in fact the 'highest' y value 
                lowest = currHeight
                lowPos = position
            position += 1   
        return self.motionPictureList[lowPos]

    #this method returns the motionPicture where the desired keypoint is at its highest throughout this motion
    def getAtHighest(self, Keypoint):
        
        position = 0
        highest = 0 #take the min Y coord to start
        highestPos = 0

        for motionPicture in self.motionPictureList:
            currHeight = int(float(motionPicture.pose[Keypoint][1])) #[keypoint] is the position [1] is the y coord
            if currHeight < highest: #numpy array the y coord moves top downwards so the 'highgest' point is technically the lowest y value 
                higest = currHeight
                highestPos = position
            position += 1   
        return self.motionPictureList[highestPos]
    
    #this method will return the position of targeted point in each frame representing the path of the keypoint throughout a motion
    def keypointPathway(self, keypoint):
        path = [] #holds the position at each frames

        for motionPicture in self.motionPictureList:
            currentPos = motionPicture.getCoordinates(keypoint)
            path.append(currentPos)
        return path
    #this method will return the angle of targeted joint in each frame representing the angle of the joint throughout a motion
    def angleTrack(self, anglePointer):
        angleList = [] #holds the position at each frames

        for motionPicture in self.motionPictureList:
            currentAngle = motionPicture.getAngle(anglePointer)
            path.append(currentAngle)
        return angleList
    
        
