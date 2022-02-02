import cv2

def showRecommendedStretches(targetAreas):
    stretchDirectory = "StretchImages/" 

    #the flexibility and mobility in these regions are the biggest factors for this motion
    stretchRegions = ["Shoulders","Thoracic Spine", "Lumbar Spine", "Hips", "Quads", "Glutes", "Calves"]


    for area in targetAreas:
        print("Area we need to work on: " + stretchRegions[area])
        if stretchRegions[area] == "Shoulders":

            #present Shoulder Dislocations

            #description
            print("""Shoulder dislocation - use an exercise band or a broom/towel etc. This exercise is great for shoulder mobility and strengthening the shoulders
            and in many ways! To do this find a comfortable grip, extend both arms straight and rotate your arms over your head until the band is behind your back. Also try changing
            your grip and different tilting positions in the hips ie. bend forwards. Reps 3x8 30 seconds rest in between.""") 
            image = cv2.imread(stretchDirectory + "ShoulderDislocations.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Shoulder Dislocations", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Shoulder Dislocations", image_)
            cv2.waitKey(0)

            #present the hand reach stretch

            #description
            print("""Shoulder Stretch - Raise your right arm over your head palm touching your upper back Withyour left hand reach up from behind your lower back palm facing away from your back
            and troy to touch or even grip your fingers together, if you can do this congratulations you have passed the Apley Scratch test and have great shoulder mobility.
            If you can't do this worry not, use a band or a stick/bar and grip as shown. then you can gently pull with your higher hand to stretch the lower shoulder and also 
            pull with the lower hand to stretch the upper shoulder. Switch hands to strech both sides. Reps 3x8 30 seconds rest in between.""")
            image = cv2.imread(stretchDirectory + "backreachLower.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Hand Reach Back Scratch Stretch Lower", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Hand Reach Back Scratch Stretch Lower", image_)
            cv2.waitKey(0)
            image = cv2.imread(stretchDirectory + "backreachUpper.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Hand Reach Back Scratch Stretch Upper", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Hand Reach Back Scratch Stretch Upper", image_)
            cv2.waitKey(0)

            #child Pose

            #description
            print("""From a table position on your hands and knees. lean your hips back whilst reaching your hands as far as comfortable in front of you with your elbows straight. Reps 2x30 seconds with 30 seconds of rest.""")
            image = cv2.imread(stretchDirectory + "childPose.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Child Pose", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Child Pose", image_)
            cv2.waitKey(0)

            cv2.destroyAllWindows()#be sure to destroy all windows before proceeding

        elif stretchRegions[area] == "Thoracic Spine":

            #cat Pose
             
            #description
            print("""Cat pose - Go into a table position on hands and knees then bend your back upwards as far as you and inhale as deep as possible then as you exhale progress to the Cow Pose. Reps 3x8 30 seconds rest.""")
            #present Shoulder Dislocations
            image = cv2.imread(stretchDirectory + "catPose.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Cat Pose", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Cat Pose", image_)
            cv2.waitKey(0)

            #Cow Pose

            #description
            print("""Cow pose - from the table position on your hands and knees arch you back pushing your stomach forward as far as possible while exhailing fully. Then move back into cat pose 3x8 30 seconds rest.""")
            #present Shoulder Dislocations
            image = cv2.imread(stretchDirectory + "cowPose.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Cow Pose", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Cow Pose", image_)
            cv2.waitKey(0)

            #elevated Thoracic spine stretch
            #description
            print("""T-Spine Stretch - this stretch use an elevated surface and put your elbows on top, fists facing towards the sky and start with your shoulders in line with your hips and elbows. Then  push your chest towards the floor. reps 2x30seconds 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "thoracicSpineStretch.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Thoracic Spine Stretch", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Thoracic Spine Stretch", image_)
            cv2.waitKey(0)

            #back Roll
            #description
            print("""Back Roll - curl into a ball/fetal position and roll your back out on the floor trying to find any tight spots/knots. You can also hold the back of the roll and try to touch your knees to the floor for a deeper stretch. Reps 2x30seconds 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "rollBack.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Back Roll", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Back Roll", image_)
            cv2.waitKey(0)

            #bar Hang
            #description
            print("""DeadHanging - relax and just hang off of a bar or something to create a light tension along the spine. try to relax and just hold the hang. Doing this frequently has tremendous health benefits, and is especially great for fixing posture. Reps 2x30seconds 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "barHang.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Dead Hang", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Dead Hang", image_)
            cv2.waitKey(0)

            cv2.destroyAllWindows() #be sure to destroy all the windows before proceeding

        elif stretchRegions[area] == "Hips":
            #Butterfly Stretch
             
            #description
            print("""Butterfly sit - on the floow with the soles of your feet pressed against eachother. Keep a straight back and try you push both knees towards the floor.
            to deepen the stretch you can move your heels towards your hips. 3x8 30 seconds rest""")
            image = cv2.imread(stretchDirectory + "butterflyStretch.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Butterfly Stretch", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Butterfly Stretch", image_)
            cv2.waitKey(0)

            #deep Lunge

            #description
            print("""Deep lunge - Lunge one leg forward and extent the back leg so that your knee is either just off the floor or slightly touching. Reps 3x8 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "deepLunge.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Deep Lunge", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Deep Lunge", image_)
            cv2.waitKey(0)

            #90's 
            #description
            print("""90's - sit on the floor bend your knees and have your feet flat on the floor. Then rotate your hips from side to side slowly whilst keeping your
            knees at a 90 degree angle. Reps 3x8 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "90's.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("90's", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("90's", image_)
            cv2.waitKey(0)

            #frog Stretch
            #description
            print("""Frog Stretch - Start off in a table position then spread your knees as far is comfortable and work your hips back as far as you can towards your heels and gently hold. 2x30seconds 30 seconds rest.""")
        
            image = cv2.imread(stretchDirectory + "frogStretch.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Frog Stretch", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Frog Stretch", image_)
            cv2.waitKey(0)

            #Zurcher Squat
            #description
            print("""Zurcher Squat - Hold yourself at the bottom of a squat put your palms together and with your elbows, gently push the knees outwards and hold. To get deeper into this stretch you can hold a weight in your hands as shown. 2x30seconds 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "zurchersquat.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Zurcher Squat", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Zurcher Squat", image_)
            cv2.waitKey(0)

            #Pigeon Stretch
            #description
            print("""Pigeon Stretch - bend one knee forwards as though you were doing a deep lunge, then rotate your knee outwards whilst keeping it bent at 90 until it touches the floor.
            keep your rear leg in a neutral position (straight ish) then keep your hips square and gently lean forwards into the stretch. This is great for the hip flexors, groin, and lower back. Reps 2x30seconds 30 seconds rest. """)
            image = cv2.imread(stretchDirectory + "pigeonStretch.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Pigeon Stretch", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Pigeon Stretch", image_)
            cv2.waitKey(0)

            cv2.destroyAllWindows() #be sure to destroy all the windows before proceeding

        elif stretchRegions[area] == "Quads":
            #Deep Quad Stretch
            #description
            print("""Quad Stretch - Place your foot laces flat on an elevated surface or up against a wall. put your knee on the floot and try to get it in line with your foot.
            Then straighten your back and gently push your hips forwards into the stretch to reach deep into the hip flexors as well as the quad. This exercise often helps with tight lowerback pains. Reps 2x30seconds 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "deepQuadStretch.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Deep Quad Stretch", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Deep Quad Stretch", image_)
            cv2.waitKey(0)

        elif stretchRegions[area] == "Calves":
            #Wall calf stretch
            #description
            print("""Calf stretch (wall) - Using a wall flex your foot so that your toes are facing upwards and push the ball of the foot into the wall then slowly bend your knee towards the wall. Reps 2x30seconds 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "calfWallStretch.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Wall Calf Stretch", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Wall Calf Stretch", image_)
            cv2.waitKey(0)

            #Elevated calf stretch
            #description
            print("""calf stretch - Place your foot flat on an elevated surface (Stairs should work fine). the gently bend your knee over your foot and gradually uncrease the weight ontop of the foot. Reps 2x30seconds 30 seconds rest.""")
            image = cv2.imread(stretchDirectory + "elevatedCalfStretch.jpg")
            #resize window (without this windows are too large)
            cv2.namedWindow("Elevated Calf Stretch", cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            image_ = cv2.resize(image, (height, width))
            cv2.imshow("Elevated Calf Stretch", image_)
            cv2.waitKey(0)
            