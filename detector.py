import numpy as np
import dlib
import imutils
import cv2

video = cv2.VideoCapture(0)

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


#To detect face
detector = dlib.get_frontal_face_detector()

#To detect facial landmarks
#predictor = dlib.shape_predictor(args["shape_detector"])
predictor  = dlib.shape_predictor("F:\IP Projs\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

#Uncomment below code and pass the image read to mainDetector to detect in static image. Current code works for video from cam
#image = cv2.imread("image")


#Calculates euclidean distance between two pts (x1,y1) and (x2,y2)
def calcDist(pt1,pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

totalBlinks = 0
minFrameBlinks = 0
minFrameToCountAsBlink = 3
thresholdEAR = 0.2 #even smaller for chinese!
flag = 0

def mainDetector(image):

    #Total blinks till now
    global totalBlinks

    #Number of frames in which blink was detected
    global minFrameBlinks

    #Threshold maintained , blinks in minimum number of frames to count it as a blink. This is done as EAR changes very very frequently.
    global minFrameToCountAsBlink

    #Threshold EAR to consider a blink
    global thresholdEAR

    #Flag used as toggler to detect blink. If this is not maintained, then totalBlinks is incremented to infinity when eyes are closed (for some time)
    global flag

    #The image passed to this function is the normal unprocessed colour image read via opencv

    #Scale the image read
    image = imutils.resize(image,width=500)

    #converting to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #finding all the faces
    rectangles = detector(gray,1)

    print('RECT',type(rectangles))
    for (i,rect) in enumerate(rectangles): #For each face
        shape = predictor(gray,rect) #Find facial landmarks
        print('PRED',type(shape))
        shape = shape_to_np(shape) #Convert to numpy array
        #Shape is a 68x2 array, 68 landmarks, each having x and y coordinate

        (x,y,w,h) = rect_to_bb(rect) #Values of rectangle (area) where face was detected in image

        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) #Draw rectangle around the face

        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #Annotation

        ind=0

        #Below loop plots facial landmarks as small circles
        for (x,y) in shape:
            # Below code plots numbers for the dots of facial landmarks
            cv2.circle(image,(x,y),1,(0,255,0),-1)

            ind+=1

        #Below code is to calculate Eye Aspect Ratio
        rightEye = (calcDist(shape[43],shape[47]) + calcDist(shape[44],shape[46]))/(2*calcDist(shape[42],shape[45])) #EAR for right eye
        leftEye = (calcDist(shape[37],shape[41]) + calcDist(shape[38],shape[40]))/(2*calcDist(shape[39],shape[36])) #EAR for left eye
        ear =  (leftEye+rightEye)/2 #AVG
        cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if(ear<thresholdEAR and flag==0):
            #print('inc1')
            minFrameBlinks+=1
        elif ear>=thresholdEAR:
            #print('greater')
            flag=0
            minFrameBlinks=0
        #print(ear,thresholdEAR)
        #print(minFrameBlinks)
        if minFrameBlinks == minFrameToCountAsBlink:
            totalBlinks+=1
            #print('blink')
            minFrameBlinks=0
            flag=1

        cv2.putText(image, "Blinks: {}".format(totalBlinks), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #print(flag)
    cv2.imshow("output",image)
    #cv2.waitKey(0)


while True:
    check,frame = video.read()
    mainDetector(frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break