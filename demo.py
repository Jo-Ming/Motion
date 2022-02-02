"""
This script will be dedicated to showing the functionality of Motion when presenting
"""
from MotionPictureClass import MotionPicture
import functions as fun
from MotionClass import Motion
import cv2
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import exerciseBank

def main():
    """For this demo we show the process of creating our output .avi file"""
    #demo1()
    """Demo 2 is an overhead squat analysis"""
    #demo2()
    #demo3()
    """Demo 4 is a pose comparison"""
    #demo4()
    squatAnalysis()
  

#will get the users height and Motion.
def getInput():
    
    #This loop will ensure the user inputs an integer to be used for distance estimation
    validHeight = False
    while(validHeight == False):
        try:
            height = int(input('Enter height (cm): ')) #to help calculate rough distances
            validHeight = True
        except:
            print("Please input an integer.")

    #getting the file location for the motion
    validVid = False
    while(validVid == False):
        location = 'videoDirectory/' + input('Enter File location: ')
        try:
            motion = Motion('Over head Squat', location, threshold = 0.7)
            desired = ''

            if motion.length > 0: #has length
                motion.playVideo()
                desired = input("Is this the right video? y or n ")
            else:
                print('Input not valid. Please check location.')

            if(desired == 'y'):
                motion.codifyMotion(False)
                motion.stitchDirectory(30) # 30fps is the normal speed of (phone) camera used
                validVid = True
        except:
            print("No file found. Please try again.")
    return height*0.93, motion #only tracks up to nose so full height would be even less accurate


def squatAnalysis():
    # positions of target areas = ["Shoulders","Thoracic Spine", "Lumbar Spine", "Hips", "Quads", "Glutes", "Calves"]
    targetAreas= [] #What areas the stretches should focus on

    #first get the height and motion from the user
    height, squat = getInput()
    topSquat = squat.getAtHighest(8) #create a MP object for top of squat
    topSquat.showImage('Top of squat')

    topSquatDistances = getSquatDistances(topSquat) # order: wristDist, ElbowDist, ShoulderDist, Hip Dist, noseHip dist, Knee Dist, ankleDist, Right Toe to heel Dist, Left Toe to Heel dist
    topSquatSFNDistances = getSFNSquatDistances(topSquat)
    #print(topSquatDistances)
    topSquatRatios = getSquatRatios(topSquatDistances)
    #print("Ratios: ", topSquatRatios)
    topAngles = getSquatAngles(topSquat) #order: headTilt Right, Right Shoulder, Left Shoulder, right elbow, left elbow, lower back right, left hip, right hip, left ankle, right ankle
    #print('Angles: ' , topAngles)
    
    botSquat = squat.getAtLowest(8) #create a motion Picture object for the bottom of squat
    botSquat.showImage('Bottom of squat') 

    botSquatDistances = getSquatDistances(botSquat) #order: wristDist, ElbowDist, ShoulderDist, Hip Dist, noseHip dist,  Knee Dist, ankleDist,RHipRankleDist, LHipLAnkleDist, Right Toe to heel Dist, Left Toe to Heel dist
    botSquatSFNDistances = getSFNSquatDistances(botSquat)
    #print(botSquatDistances)
    botSquatRatios = getSquatRatios(botSquatDistances) #elbow shoulder ratio, knee waist ratio, feet to shoulder ratio
    #print("Bot squat ratios: ", botSquatRatios)
    botAngles = getSquatAngles(botSquat) #headTilt Right, Right Shoulder, Left Shoulder, right elbow, left elbow, lower back right, left hip, right hip, left ankle, right ankle
    print('Bot squat Angles: ' , botAngles)

    #we can begin analysis 
    botSquat.drawSkeleton(False)
    topSquat.drawSkeleton(False)

    #here is the pose in a more reader friendly layout 
    print("Pose at bottom of squat")
    botSquat.printKeypoints()

    #work out distance relationship to pixels
    nosePosY = topSquat.getCoordinates(0)[1]
    bigToeY = topSquat.getCoordinates(22)[1] #use right big toe 
    #distance in y coords between nose and bigToe is rough height
    pixelHeight = abs(bigToeY - nosePosY) #difference in Y values when standing
    cmPERpxl = (height*0.93)/pixelHeight #can use this to estimate distance multiplied by 0.93 because my nose to the top of my head is about 7% of my body

    ############### Static Analysis: we analyse the still motion pictures #####################

    ############################### analyse the upper body ####################################

    #display shoulder and elbow angles
    image = botSquat.drawSelectedAngles([2,3,4,5])
    cv2.imshow('Squat With Angles',image)
    cv2.waitKey(0)

    elbowDistanceEstimate = int(botSquatDistances[1]*cmPERpxl)
    print(botSquatDistances[1], cmPERpxl)
    botSquatLElbow = botSquat.getCoordinates(3)
    botSquatRElbow = botSquat.getCoordinates(6)

    #draw line showing shoulderDistance
    cv2.line(image, botSquatLElbow, botSquatRElbow, (55,175,212), 3, lineType=cv2.FILLED) #BGR
    cv2.putText(image, "elbowDist:{}cm".format(elbowDistanceEstimate), (botSquatLElbow[0]+10, botSquatLElbow[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 0, lineType=cv2.LINE_AA)
    cv2.imshow('elbow Distance', image)
    cv2.waitKey(0)

    #check shoulders
    LShldrAngleBot = botAngles[2]
    RShldrAngleBot = botAngles[1]


     #if there is a large difference between shoulder mobility this could mean injury/increased potential for injury
     #check for differences in shoulder angles
    if(abs(LShldrAngleBot - RShldrAngleBot) > 10): #would signify imbalance
        targetAreas.append(0) #add shoulders to target Areas
        (print('detected shoulder imbalanced'))
        if(LShldrAngleBot>RShldrAngleBot):
            print('Left shoulder restricted')
        else:
            print('Right shoulder restricted')

    #check elbow angles 
    #we want the elbows to be straight. Bend in elbows tends to signal tight shoulders, chest, and T-spine
    RElbowAngleBot = botAngles[3]
    LElbowAngleBot = botAngles[4]
    #injurys can also stem from where they connect from so cant rule out shoulders if arm bends
    if(RElbowAngleBot < 170 or LElbowAngleBot < 170):
        targetAreas.append(0)# add shoulders
        print('imbalance between arms, potential risk of injury detected')
        if(LElbowAngleBot<RElbowAngleBot):
            print('left arm/shoulder restricted')
        else:
            print('Right arm/shoulder restricted')

    botSquatShldrElbwRatio = botSquatRatios[0]
    if botSquatShldrElbwRatio > 2: #check how far elbows are out in relation to shoulders
        targetAreas.append(0)
        targetAreas.append(1)

    #get wrist positions and shoulder positions whilst standing 
    standingLWrist = topSquat.getCoordinates(7)
    standingRWrist = topSquat.getCoordinates(4)
    standingRShldr = topSquat.getCoordinates(2)
    standingLShldr = topSquat.getCoordinates(5)

    #get sqautted wrists
    squatRWrist = botSquat.getCoordinates(4)
    squatLWrist = botSquat.getCoordinates(7)

    #get squat shoulders
    squatRShldr = botSquat.getCoordinates(2)
    squatLShldr = botSquat.getCoordinates(5)

    standLeftWSDist = fun.getEuclideanDistance(standingLWrist, standingLShldr)
    standRightWSDist = fun.getEuclideanDistance(standingRWrist,standingRShldr)
    standingShldrElbowDist = (standLeftWSDist + standRightWSDist) / 2 #take the average dist

    squatLeftWSDist = fun.getEuclideanDistance(squatLWrist, squatLShldr)
    squatRightWSDist = fun.getEuclideanDistance(squatRWrist,squatRShldr)
    squatShldrElbowDist = (squatLeftWSDist + squatRightWSDist) / 2 #average dist

    HeadToHipChange = topSquatSFNDistances[4]-botSquatSFNDistances[4] #proportial change to whole pose
    #if the distance between the wrists and the shoulder decrease and head is coming closer to the hips then the hands are coming forwards which suggests tight shoulders + therassic spine + lower back
    #and the distance between head and hips has a significant decrease
    if((squatShldrElbowDist - standingShldrElbowDist) < -10 and headToHipChange > 0.15):
        targetAreas.append(0)
        targetAreas.append(1)
        #usually the body compensate using one shoulder more if there is an injury/restriction on the other
        if(squatLeftWSDist > squatRightWSDist):
            print('Detected restriction in left shoulder/therassic spine')
        elif(squatLeftWSDist < squatRightWSDist):
            print('Detected restriction in right shoulder/therassic spine')


     ####################### Analyse the lower body ######################################

    image = botSquat.drawSelectedAngles([8,9,10,11,12,13])
    cv2.imshow('Squat With With Lower Body Angles',image)
    cv2.waitKey(0)

    #if feet are too wide this signals tight adductors/groin areas 
    if botSquatRatios[2] > 1.5:
        targetAreas.append(3) #although we don't have an aductor targetted routine at the moment the frog stretch is still an ideal stretch ideal for this scenario

    #find distance between toes and ankles when standing

    #get coords and distances between left and right toes and ankles in standing position
    standingRBToe = topSquat.getCoordinates(22)
    standingLBToe = topSquat.getCoordinates(19)

    standingLAnkle = topSquat.getCoordinates(14)
    standingRAnkle = topSquat.getCoordinates(11)

    RAnkleToeDist1 = fun.getEuclideanDistance(standingRBToe, standingRAnkle)
    LAnkleToeDist1 = fun.getEuclideanDistance(standingLBToe, standingLAnkle)

    #now do the same for squat position
    squatRBToe = botSquat.getCoordinates(22)
    squatLBToe = botSquat.getCoordinates(19)

    squatLAnkle = botSquat.getCoordinates(14)
    squatRAnkle = botSquat.getCoordinates(11)

    RAnkleToeDist2 = fun.getEuclideanDistance(squatRBToe, squatRAnkle)
    LAnkleToeDist2 = fun.getEuclideanDistance(squatLBToe, squatLAnkle)

    #if the distance between the ankle and toes or toes and heels increase (from the perspective of the camera then the heels are coming off of the floor )
    changeOnRightFoot = RAnkleToeDist2 - RAnkleToeDist1
    changeOnLeftfoot = LAnkleToeDist2 - LAnkleToeDist1
    if(changeOnLeftfoot > 1 or changeOnRightFoot >1):
        targetAreas.append(6) #calves
        if(changeOnLeftfoot > 1 and changeOnRightFoot >1):
            print('detected: both heels have come off the floor')
        elif(changeOnLeftfoot > 1):
            print('Left heel left the floor')
        elif(changeOnRightFoot > 1):
            print('Right heel left the floor')
    
    #if knee to waist ratio is below 1 and your feet are shoulderwidth apart then knees are pointing inwards which is bad form usially due to tight hips and putting a lot more strain onto ligaments
    if (botSquatRatios[1] <= 1 and botSquatRatios[2] >= 1):
        targetAreas.append(3)
        print("Inward pointing knees detected, Risking injury. Work out pointing your knees forward or even slightly outward.")
    
    #check if feet are pointing inwards 15, 21 are left and right foot vectors
    botSquatVectors = botSquat.vectorise()
    RightFootVect = botSquatVectors[15]
    LeftFootVect = botSquatVectors[21]

    #check gradients
    RVectGrad = RightFootVect[1]/RightFootVect[0]
    LVectgrad = LeftFootVect[1]/LeftFootVect[0]
    #right foot wants foot vector pointing to left of image and left foot vector pointing left
    #openCV builds image top down so it is opposite to intuition 
    if (RVectGrad >= 0 or LVectGrad <= 0):
        targetAreas.append(3)
        print("Inward pointing feet detected")
        if(RVectGrad >= 0):
            print("Warning right foot pointing inward. This puts extra strain on ligamnets.")
        if(LVectGrad <= 0):
            print("Warning left foot pointing inward. This puts extra strain on ligamnets.")

    hipHeight = botSquat.getCoordinates(8)
    leftKnee = botSquat.getCoordinates(13)
    rightKnee = botSquat.getCoordinates(10)

    kneeHeight = (leftKnee[1] + rightKnee[1])/2

    #squat is shallow target could be calves, lower back, glutes, and hips
    #ideal squat should go below parallel
    if hipHeight[1] < kneeHeight:
        targetAreas.append(3)
        targetAreas.append(4)

    ######################## Dynamic Analysis: tracking across motion #########################

    noseTrack = squat.keypointPathway(0)

    leftShldrTrack = squat.keypointPathway(5)
    rightShldrtrack = squat.keypointPathway(2)

    leftHipTrack = squat.keypointPathway(12)
    rightHipTrack = squat.keypointPathway(9)

    #if a persons squat for is stable then a line between the shoulders and hips should reamin parallel throughout the motion
    #two lines are parallel if the have the same gradient 
    # we can calculate gradient using, rise over run: (y2-y1) - (x2-x1) 
    pointer = 0
    wobbleFrameCount = 0 #this index will somewhat reflect the stability of the motion
    for frame in leftShldrTrack:
        shldrlineGradient = fun.getGradient(leftShldrTrack[pointer], rightShldrtrack[pointer])
        hipLineGradient = fun.getGradient(leftHipTrack[pointer], rightHipTrack[pointer])

        if abs(shldrlineGradient - hipLineGradient) > 0.1: #allow small margin of forgiveness
            wobbleFrameCount += 1
        pointer += 1

    #we want the persosn head to stay in line for an ideal squat
    xSum = 0
    for frame in noseTrack:
        xSum += frame[0]
    xBar = xSum / len(noseTrack)

    varience = 0
    for frame in noseTrack:
        varience += (frame[0] - xBar)**2/len(noseTrack)

    #now we can find the standard deviation
    sd = math.sqrt(varience)

    # finally we can score the movement
    #currently the score can only be based of ratioList, target areas, wobbleframecount

    print(wobbleFrameCount)
    print(targetAreas)
    targetAreas = fun.removeDuplicates(targetAreas)
    exerciseBank.showRecommendedStretches(targetAreas)


#returns all angles relavent to squat analysis
def getSquatAngles(squat):
    anglePositions = [0,2,3,4,5,7,8,9,10,11,12,13]

    angleList = [] #headTilt Right, Right Shoulder, Left Shoulder, right elbow, left elbow, lower back right, left hip, right hip, left ankle, right ankle

    for position in anglePositions:
        currentAngle = squat.getAngle(position)
        angleList.append(currentAngle)
    return angleList

#returns distances needed for squat analysis
def getSquatRatios(distanceList):
    #distancesList order:noseHipDist, wristDist, ElbowDist, ShoulderDist, Hip Dist, Knee Dist, ankleDist, Right Toe to heel Dist, Left Toe to Heel dist
    ratioList = [] #elbow shoulder ratio, knee waist ratio, feet to shoulder ratio

    #elbow to shoulder ratio
    elbwShldrRatio = distanceList[2]/distanceList[3]
    ratioList.append(elbwShldrRatio)

    #knee waist ratio
    kneeWaistRatio = distanceList[5]/distanceList[4]
    ratioList.append(kneeWaistRatio)

    #ankle to shoulder ratio
    ankleShldrRatio = distanceList[6]/distanceList[3]
    ratioList.append(ankleShldrRatio)

    return ratioList

# returns relative distances in order: wristDist, ElbowDist, ShoulderDist, Hip Dist, noseHipDist, Knee Dist, ankleDist,RHipRankleDist, LHipLAnkleDist, Right Toe to heel Dist, Left Toe to Heel dist, RAnkleToeDist, LAnkleToeDist
def getSFNSquatDistances(squat):

    # use Scaled Factored,Normalised pose
    #print(squat.SFNpose)
    distList = []

    #locate nose and mid hip + distance
    nose = squat.getSFNCoordinates(0)
    midHip = squat.getSFNCoordinates(8)
    noseHipDist = fun.getEuclideanDistance(nose, midHip)
    distList.append(noseHipDist)
    
    #locate wrists + Distance
    RWrist = squat.getSFNCoordinates(4)
    LWrist = squat.getSFNCoordinates(7)
    wristDist = fun.getEuclideanDistance(RWrist, LWrist)
    distList.append(wristDist)
    
    #locate elbows + Distance
    RElbow = squat.getSFNCoordinates(3)
    LElbow = squat.getSFNCoordinates(6) 
    elbowDist = fun.getEuclideanDistance(RElbow, LElbow)
    distList.append(elbowDist)

    #locate shoulders + Distance
    RShldr = squat.getSFNCoordinates(2)
    LShldr = squat.getSFNCoordinates(5) 
    shldrDist = fun.getEuclideanDistance(RShldr, LShldr)
    distList.append(shldrDist)

    #wrist to shoulder distance
    RWristShldrDist = fun.getEuclideanDistance(RWrist, RShldr)
    LWristShldrDist = fun.getEuclideanDistance(LWrist, LShldr)

    #locate Hips + dist
    RHip = squat.getSFNCoordinates(9)
    LHip = squat.getSFNCoordinates(12) 
    hipDist = fun.getEuclideanDistance(RHip, LHip)
    distList.append(hipDist)

    #distance nose to hips 
    nose = squat.getSFNCoordinates(0)
    midHip = squat.getSFNCoordinates(8) 
    noseHipLength = fun.getEuclideanDistance(nose, midHip)

    #locate knees + dist
    RKnee = squat.getSFNCoordinates(10)
    LKnee = squat.getSFNCoordinates(13) 
    kneeDist = fun.getEuclideanDistance(RKnee, LKnee)
    distList.append(kneeDist)

    #locate ankles + distance
    RAnkle = squat.getSFNCoordinates(14)
    LAnkle = squat.getSFNCoordinates(11) 
    ankleDist = fun.getEuclideanDistance(RAnkle, LAnkle)
    distList.append(ankleDist)

    #hip to ankle distance 
    RHipRAnkleDist = fun.getEuclideanDistance(RHip, RAnkle)
    LHipLAnkleDist = fun.getEuclideanDistance(LHip, LAnkle)
    distList.append(RHipRAnkleDist)
    distList.append(LHipLAnkleDist)

    #ToeHeel distances
    RBToe = squat.getSFNCoordinates(22)
    RHeel = squat.getSFNCoordinates(24)
    RToeHeelDist = fun.getEuclideanDistance(RBToe, RHeel)
    distList.append(RToeHeelDist) 

    LBToe = squat.getSFNCoordinates(19)
    LHeel = squat.getSFNCoordinates(21)
    LToeHeelDist = fun.getEuclideanDistance(LBToe, LHeel)
    distList.append(LToeHeelDist)

    #toe ankle distances

    RAnkleToeDist = fun.getEuclideanDistance(RBToe, RAnkle)
    LAnkleToeDist = fun.getEuclideanDistance(LBToe, LAnkle)
    distList.append(RAnkleToeDist)
    distList.append(LAnkleToeDist)

    #toe Ankle Distances
    LBToe = squat.getSFNCoordinates(19)
    LHeel = squat.getSFNCoordinates(21)
    LToeHeelDist = fun.getEuclideanDistance(LBToe, LHeel)
    distList.append(LToeHeelDist)

    print(distList)

    return distList

# returns relative distances in order: wristDist, ElbowDist, ShoulderDist, Hip Dist, noseHipDist Knee Dist, ankleDist,RHipRankleDist, LHipLAnkleDist, Right Toe to heel Dist, Left Toe to Heel dist
def getSquatDistances(squat):

    # use Scaled Factored,Normalised pose
    #print(squat.SFNpose)
    distList = []

    #locate nose and mid hip + distance
    nose = squat.getCoordinates(0)
    midHip = squat.getCoordinates(8)
    noseHipDist = fun.getEuclideanDistance(nose, midHip)
    distList.append(noseHipDist)
    
    #locate wrists + Distance
    RWrist = squat.getCoordinates(4)
    LWrist = squat.getCoordinates(7)
    wristDist = fun.getEuclideanDistance(RWrist, LWrist)
    distList.append(wristDist)
    
    #locate elbows + Distance
    RElbow = squat.getCoordinates(3)
    LElbow = squat.getCoordinates(6) 
    elbowDist = fun.getEuclideanDistance(RElbow, LElbow)
    distList.append(elbowDist)

    #locate shoulders + Distance
    RShldr = squat.getCoordinates(2)
    LShldr = squat.getCoordinates(5) 
    shldrDist = fun.getEuclideanDistance(RShldr, LShldr)
    distList.append(shldrDist)

    #wrist to shoulder distance
    RWristShldrDist = fun.getEuclideanDistance(RWrist, RShldr)
    LWristShldrDist = fun.getEuclideanDistance(LWrist, LShldr)

    #locate Hips + dist
    RHip = squat.getCoordinates(9)
    LHip = squat.getCoordinates(12) 
    hipDist = fun.getEuclideanDistance(RHip, LHip)
    distList.append(hipDist)

    #distance nose to hips 
    nose = squat.getCoordinates(0)
    midHip = squat.getCoordinates(8) 
    noseHipLength = fun.getEuclideanDistance(nose, midHip)

    #locate knees + dist
    RKnee = squat.getCoordinates(10)
    LKnee = squat.getCoordinates(13) 
    kneeDist = fun.getEuclideanDistance(RKnee, LKnee)
    distList.append(kneeDist)

    #locate ankles + distance
    RAnkle = squat.getCoordinates(14)
    LAnkle = squat.getCoordinates(11) 
    ankleDist = fun.getEuclideanDistance(RAnkle, LAnkle)
    distList.append(ankleDist)

    #hip to ankle distance 
    RHipRAnkleDist = fun.getEuclideanDistance(RHip, RAnkle)
    LHipLAnkleDist = fun.getEuclideanDistance(LHip, LAnkle)
    distList.append(RHipRAnkleDist)
    distList.append(LHipLAnkleDist)

    #ToeHeel distances
    RBToe = squat.getCoordinates(22)
    RHeel = squat.getCoordinates(24)
    RToeHeelDist = fun.getEuclideanDistance(RBToe, RHeel)
    distList.append(RToeHeelDist)

    #toe Ankle Distances

    LBToe = squat.getCoordinates(19)
    LHeel = squat.getCoordinates(21)
    LToeHeelDist = fun.getEuclideanDistance(LBToe, LHeel)
    distList.append(LToeHeelDist)
    
    print(distList)

    return distList

def demo1():
    """
    This demo is to show how to create an output with the pose estimation drawn onto the original video
    """
    #directory for example video
    demo1Path = 'videoDirectory/pancake.mp4'

    demo1Motion = Motion('demo1 show Pose Estimation',demo1Path, threshold = 0.7) #make an instance of motion object
    demo1Motion.playVideo() #show video
    demo1Motion.codifyMotion(showAngles = True) #analyse the frames
    demo1Motion.stitchDirectory(10) #put it back at 10fps slow mo 

def demo2():
    """
    demo 2 is an in depth front (overhead) squat analysis
    """
    targetAreas = [] #targets for joint mobility and stretches

    print("Perform an overhead squat. For this hold a exercise band, towel, or stick. grip at a comfortable width rotate it overhead and keep the shoulders in position throughout a full squat.")

    #video location
    valid = False
    while(valid == False):
        try:
            height = int(input('Enter height (cm): ')) #to help calculate rough distances
            valid = True
        except:
            print("Please input an integer.")
    height = int(height*0.93) #roughly 13cm from my nose to the top of my head which is around abouts 7 percent

    demoPath2 = 'videoDirectory/stephen.mp4'

    #create instance of a motion
    demo2Motion = Motion('Demo2 Front Squat Analysis', demoPath2, threshold = 0.7)
    demo2Motion.playVideo()
    demo2Motion.codifyMotion(showAngles = False)
    demo2Motion.stitchDirectory(20) # 30fps is the normal speed of (phone) camera used

    #this is the frame where our hips are lowest so we will analyse this
    squat = demo2Motion.getAtLowest(8) #mid hips = 8
    squat.drawSkeleton(False)
    squat.showImage('Squat Analysis') #show our motion pictures image

    #here is the pose in a more reader friendly layout 
    squat.printKeypoints()

    ############### Static Analysis: we analyse the still motion pictures #####################

    ############################### analyse the upper body ####################################

    #convert the pose elements from strings to floats
    
    #elbows are points 3 and 6
    squatRElbow = squat.getCoordinates(6)
    squatLElbow = squat.getCoordinates(3)
    #shoulder points 2 and 5
    squatRShldr = squat.getCoordinates(2)
    squatLShldr = squat.getCoordinates(5)

    #find distance between nose to toe for rough distance estimation
    standing = demo2Motion.motionPictureList[0] #first frame I am standing
    nosePosY = standing.getCoordinates(0)[1]
    bigToeY = standing.getCoordinates(22)[1] #use right big toe 

    #get angles of joints 
    LShldrAngle = squat.getAngle(3)
    RShldrAngle = squat.getAngle(2)
    LElbowAngle = squat.getAngle(5)
    RElbowAngle = squat.getAngle(4)

    #if there is a large difference between shoulder mobility this could mean injury/increased potential for injury
    if(abs(LShldrAngle - RShldrAngle) > 10):
        targetAreas.append(0)
        (print('detected shoulder imbalanced'))
        if(LShldrAngle>RShldrAngle):
            print('Left shoulder restricted')
        else:
            print('Right shoulder restricted')
    #injurys can also stem from where they connect from so cant rule out shoulders if arm bends
    if(abs(LElbowAngle - RElbowAngle) > 10):
        (print('imbalance between arms, potential injury detected'))
        if(LElbowAngle<RElbowAngle):
            print('left arm/shoulder restricted')
        else:
            print('Right arm/shoulder restricted')

    #display shoulder and elbow angles
    image = squat.drawSelectedAngles([2,3,4,5])
    cv2.imshow('Squat With Angles',image)
    cv2.waitKey(0)

    #estimate Distances
    #find the distance between elbows
    squatElbowDist = fun.getEuclideanDistance(squatRElbow,squatLElbow)
    #find distance between shoulders 
    squatShoulderDist = fun.getEuclideanDistance(squatRShldr,squatLShldr)
    squatShldrElbwRatio = squatElbowDist/squatShoulderDist #closer to 1 is a better score (1 being elbows shoulder width apart)

    if squatShldrElbwRatio > 2:
        targetAreas.append(0)
        targetAreas.append(1)

    #distance in y coords between nose and bigToe is rough height
    pixelHeight = abs(bigToeY - nosePosY) #difference in Y values when standing
    cmPERpxl = height/pixelHeight #can use this to estimate distance

    elbowDistanceEstimate = int(squatElbowDist*cmPERpxl)

    #draw line showing shoulderDistance
    cv2.line(image, squatRElbow, squatLElbow, (55,175,212), 3, lineType=cv2.FILLED) #BGR
    cv2.putText(image, "elbowDist:{}cm".format(elbowDistanceEstimate), (squatLElbow[0]+10, squatLElbow[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 0, lineType=cv2.LINE_AA)
    cv2.imshow('elbow Distance', image)
    cv2.waitKey(0)

    #get wrist positions and shoulder positions whilst standing 
    standingLWrist = standing.getCoordinates(7)
    standingRWrist = standing.getCoordinates(4)
    standingRShldr = standing.getCoordinates(2)
    standingLShldr = standing.getCoordinates(5)

    #get sqautted wrists
    squatRWrist = squat.getCoordinates(4)
    squatLWrist = squat.getCoordinates(7)

    standLeftWSDist = fun.getEuclideanDistance(standingLWrist, standingLShldr)
    standRightWSDist = fun.getEuclideanDistance(standingRWrist,standingRShldr)
    standingShldrElbowDist = (standLeftWSDist + standRightWSDist) / 2 #take the average dist

    squatLeftWSDist = fun.getEuclideanDistance(squatLWrist, squatLShldr)
    squatRightWSDist = fun.getEuclideanDistance(squatRWrist,squatRShldr)
    squatShldrElbowDist = (squatLeftWSDist + squatRightWSDist) / 2 #average dist
    
    #if the distance between the wrists and the shoulder decrease then the hands are coming forwards which suggests tight shoulders + therassic spine
    if((squatShldrElbowDist - standingShldrElbowDist) < -10 ):
        targetAreas.append(0)
        targetAreas.append(1)
        #usually the body compensate using one shoulder more if there is an injury/restriction on the other
        if(squatLeftWSDist > squatRightWSDist):
            print('Detected restriction in left shoulder/therassic spine')
        elif(squatLeftWSDist < squatRightWSDist):
            print('Detected restriction in right shoulder/therassic spine')

    ####################### Analyse the lower body ######################################

    image = squat.drawSelectedAngles([8,9,10,11,12,13])
    cv2.imshow('Squat With With Lower Body Angles',image)
    cv2.waitKey(0)

    #find distance between toes and ankles when standing

    #get coords and distances between left and right toes and ankles in standing position
    standingRBToe = standing.getCoordinates(22)
    standingLBToe = standing.getCoordinates(19)

    standingLAnkle = standing.getCoordinates(14)
    standingRAnkle = standing.getCoordinates(11)

    RAnkleToeDist1 = fun.getEuclideanDistance(standingRBToe, standingRAnkle)
    LAnkleToeDist1 = fun.getEuclideanDistance(standingLBToe, standingLAnkle)

    #now do the same for squat position
    squatRBToe = squat.getCoordinates(22)
    squatLBToe = squat.getCoordinates(19)

    squatLAnkle = squat.getCoordinates(14)
    squatRAnkle = squat.getCoordinates(11)

    RAnkleToeDist2 = fun.getEuclideanDistance(squatRBToe, squatRAnkle)
    LAnkleToeDist2 = fun.getEuclideanDistance(squatLBToe, squatLAnkle)

    #if the distance between the ankle and toes increase (from the perspective of the camera then the heels are coming off of the floor)
    changeOnRightFoot = RAnkleToeDist2 - RAnkleToeDist1
    changeOnLeftfoot = LAnkleToeDist2 - LAnkleToeDist1
    if(changeOnLeftfoot > 1 or changeOnRightFoot >1):
        targetAreas.append(6) #calves
        if(changeOnLeftfoot > 1 and changeOnRightFoot >1):
            print('detected: both heels have come off the floor')
        elif(changeOnLeftfoot > 1):
            print('Left heel left the floor')
        elif(changeOnRightFoot > 1):
            print('Right heel left the floor')

    hipHeight = squat.getCoordinates(8)
    leftKnee = squat.getCoordinates(13)
    rightKnee = squat.getCoordinates(10)

    kneeHeight = (leftKnee[1] + rightKnee[1])/2

    #squat is shallow target could be calves, lower back, glutes, and hips
    #ideal squat should go below parallel
    if hipHeight[1] < kneeHeight:
        targetAreas.append(3)
        targetAreas.append(4)
    
    ######################## Dynamic Analysis: tracking across motion #########################

    noseTrack = demo2Motion.keypointPathway(0)

    leftShldrTrack = demo2Motion.keypointPathway(5)
    rightShldrtrack = demo2Motion.keypointPathway(2)

    leftHipTrack = demo2Motion.keypointPathway(12)
    rightHipTrack = demo2Motion.keypointPathway(9)

    #if a persons squat for is stable then a line between the shoulders and hips should reamin parallel throughout the motion
    #two lines are parallel if the have the same gradient 
    # we can calculate gradient using, rise over run: (y2-y1) - (x2-x1) 
    pointer = 0
    wobbleFrameCount = 0 #this index will somewhat reflect the stability of the motion
    for frame in leftShldrTrack:
        shldrlineGradient = fun.getGradient(leftShldrTrack[pointer], rightShldrtrack[pointer])
        hipLineGradient = fun.getGradient(leftHipTrack[pointer], rightHipTrack[pointer])

        if abs(shldrlineGradient - hipLineGradient) > 0.1: #allow small margin of forgiveness
            wobbleFrameCount += 1
        pointer += 1

    #we want the persosn head to stay in line for an ideal squat
    xSum = 0
    for frame in noseTrack:
        xSum += frame[0]
    xBar = xSum / len(noseTrack)

    varience = 0
    for frame in noseTrack:
        varience += (frame[0] - xBar)**2/len(noseTrack)

    #now we can find the standard deviation
    sd = math.sqrt(varience)

    # finally we can score the movement
    #currently the score can only be based of ratioList, target areas, wobbleframecount
    score = 5 #assume perfect score

    #score -= score - ((5/len(targetAreas)*len(targetAreas)))
    if score == 0:
        return 0 #failed every checkpoint

    print("Shoulder width ratio: " , squatShldrElbwRatio)
    print(wobbleFrameCount)
    print(targetAreas)
    targetAreas = fun.removeDuplicates(targetAreas)
    exerciseBank.showRecommendedStretches(targetAreas)

#pose recognition/classification
def demo3():
    """
    demo 3 is a pose classification demo
    """
    #get pose images
    demoStand = 'imageDirectory/stand.jpg'
    dimensions = fun.getImgDimensions(demoStand)
    example = MotionPicture(dimensions, demoStand, 0.7)
    pose = example.pose
    pose = np.array(fun.SFNpose(pose))
    pose = np.array(pose).reshape(1, 25, 2)

    poseClassificationModel = tf.keras.models.load_model('motionCNN.model') # load the model

    posePrediction = poseClassificationModel.predict(pose)
    print(posePrediction)

#compare cosine similarity ad euclidean distance across two motions
def demo4():
    motion1 = Motion("Squat1", 'videoDirectory/frontSquatDemo.mp4', threshold = 0.7)
    motion1.codifyMotion(showAngles = True)
    poseList1 = []
    vectorList1 = []
    #get pose for each frame and vectorise 
    for motionPicture in motion1.motionPictureList:
        #poseList1.append(motionPicture.pose)
        #vectorList1.append(motionPicture.vectorise())
        sfnPose = motionPicture.getSFNpose()
        poseList1.append(sfnPose)
        vectorList1.append(fun.vectorise(sfnPose))

    poseList2 = []
    vectorList2 = []
    motion2 = Motion("Squat2", 'videoDirectory/overheadSquat2.mp4', threshold = 0.7)
    motion2.codifyMotion(showAngles = True)
    #get pose for each frame and vectorise 
    for motionPicture in motion2.motionPictureList:
        #poseList2.append(motionPicture.pose)
        #vectorList2.append(motionPicture.vectorise())
        sfnPose = motionPicture.getSFNpose()
        poseList2.append(sfnPose)
        vectorList2.append(fun.vectorise(sfnPose))
    
    #find the euclidean distance and cosine similarity between each frame
    distanceList = []
    cosineSimList = []
    iterations = 0

    #we loop through shorter motion to avoid iteration errors
    if len(poseList1)> len(poseList2):
        iterations = len(poseList2)
    else:
        iterations = len(poseList1)
    
    #get distances and cosine similarity for each frame
    for i in range(iterations):
        dist = fun.getPoseDistance(poseList1[i], poseList2[i])
        distanceList.append(dist)

        cosineSimList.append(fun.cosineSimilarity(vectorList1[i], vectorList2[i]))

    print(distanceList)
    print(cosineSimList)

    #plot these values onto graphs
    frames = range(0, iterations)
    fig, ax = plt.subplots()
    ax.plot(frames, distanceList)

    ax.set(xlabel='Frame Number', ylabel='Euclidean Distance', title = "Distance similarity graph")
    ax.grid()
    #fig.savefig("distanceGraph.png")
    plt.show()
    
    #cosine similarity graph
    fig, ax = plt.subplots()
    ax.plot(frames, cosineSimList)

    ax.set(xlabel='Frame Number', ylabel='Cosine Similarity', title = "Cosine similarity graph")
    ax.grid()
    #fig.savefig("distanceGraph.png")
    plt.show()
    return 0

main()