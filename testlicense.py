import cv2
import numpy as np
import os
import math
import random
img = cv2.imread("test.jpg")


kNearest = cv2.ml.KNearest_create()


class PossibleChar:
    def __init__(self, contour):
        self.contour = contour
        self.rect = cv2.boundingRect(self.contour)
        [X, Y, width, height] = self.rect
        self.X = X
        self.Y = Y
        self.width = width
        self.height = height
        self.area = self.width * self.height
        self.aspect_ratio = float(self.width)/float(self.height)
        self.CenterX = (self.X + self.X + self.width) / 2
        self.CenterY = (self.Y + self.Y + self.height) / 2

        self.fltDiagonalSize = math.sqrt((self.width ** 2) + (self.height ** 2))


def display(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess(img):
    ### HSV SHOWS BRIHGTENED REGIONS BETTER THAN GRAY
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHue, imgSat, imgValue = cv2.split(hsv)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tophat = cv2.morphologyEx(imgValue, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(imgValue, cv2.MORPH_BLACKHAT, kernel)
    top_plus_black = cv2.add(imgValue, tophat)
    top_plus_black_minus_black = cv2.subtract(top_plus_black, blackhat)

    blur = cv2.GaussianBlur(top_plus_black_minus_black, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    display(thresh, "Thresholded Image")
    return imgValue, thresh


imgGray, imgThresh = preprocess(img)
imgThreshCopy = imgThresh.copy()
img2, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
MIN_AREA = 80
MIN_WIDTH = 2
MIN_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0


def loadKNNDataAndTrainKNN():
    allContoursWithData = []
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return False

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return False

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest.setDefaultK(1)

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    return True    

def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""
    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.CenterX)
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)
    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.X, currentChar.Y)
        pt2 = ((currentChar.X + currentChar.width), (currentChar.Y + currentChar.height))
        cv2.rectangle(imgThreshColor, pt1, pt2, (0.0, 255.0, 0.0), 2)
        imgROI = imgThresh[currentChar.Y : currentChar.Y + currentChar.height, currentChar.X : currentChar.X + currentChar.width]

        imgROIResized = cv2.resize(imgROI, (20, 30))

        npaROIResized = imgROIResized.reshape((1, 600))

        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)

        strCurrentChar = str(chr(int(npaResults[0][0])))

        strChars = strChars + strCurrentChar

    return strChars

def findListOfListsOfMatchingChars(listOfPossibleChars):

    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:
        listOfMatchingChars = []

        for possibleMatchingChar in listOfPossibleChars:
            if possibleMatchingChar == possibleChar:
                continue
            fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

            fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

            fltChangeInArea = float(abs(possibleMatchingChar.area - possibleChar.area)) / float(possibleChar.area)

            fltChangeInWidth = float(abs(possibleMatchingChar.width - possibleChar.width)) / float(possibleChar.width)
            fltChangeInHeight = float(abs(possibleMatchingChar.height - possibleChar.height)) / float(possibleChar.height)

            if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * 5.0) and
                fltAngleBetweenChars < 12.0 and
                fltChangeInArea < 0.5 and
                fltChangeInWidth < 0.8 and
                fltChangeInHeight < 0.2):

                listOfMatchingChars.append(possibleMatchingChar)

        listOfMatchingChars.append(possibleChar)

        if len(listOfMatchingChars) < 3:
            continue
        
        listOfListsOfMatchingChars.append(listOfMatchingChars)

        listOfPossibleChars = list(set(listOfPossibleChars) - set(listOfMatchingChars))

    return listOfListsOfMatchingChars


def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.CenterX - secondChar.CenterX)
    intY = abs(firstChar.CenterY - secondChar.CenterY)
    return math.sqrt((intX ** 2) + (intY ** 2))


def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.CenterX - secondChar.CenterX))
    fltOpp = float(abs(firstChar.CenterY - secondChar.CenterY))
    if fltAdj != 0.0:
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708
    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)
    return fltAngleInDeg


def detect_chars(imgThresh):
    possiblechars = []
    imgContour = np.zeros((imgThresh.shape[0], imgThresh.shape[1], 3), np.uint8)
    for i in range(len(contours)):
        possibleChar = PossibleChar(contours[i])
        if (possibleChar.area>MIN_AREA and possibleChar.width>MIN_WIDTH and possibleChar.height>MIN_HEIGHT and 
            possibleChar.aspect_ratio>MIN_ASPECT_RATIO and possibleChar.aspect_ratio<MAX_ASPECT_RATIO):
            
            cv2.drawContours(imgContour, contours, i, (255.0, 255.0, 255.0))
            possiblechars.append(possibleChar)
            
    display(imgContour, "{}".format(i))
    return possiblechars


loadKNNDataAndTrainKNN()

possiblechars = detect_chars(imgThresh)

listOfMatchingChars = findListOfListsOfMatchingChars(possiblechars)

imgContour = np.zeros((imgThresh.shape[0], imgThresh.shape[1], 3), np.uint8)
contours = []
for listOfMatchingChar in listOfMatchingChars:
    for matchingChar in listOfMatchingChar:
        contours.append(matchingChar.contour)
        cv2.drawContours(imgContour, contours, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
display(imgContour,"test")
print recognizeCharsInPlate(imgThresh, listOfMatchingChars[0])


