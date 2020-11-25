import cv2
import numpy as np
from mss import mss

handHistogram = None
amountOfMeasuringRectangles = 9
xCoordinatesOfMeasuringRectangles_topLeft = None
yCoordinatesOfMeasuringRectangles_topLeft = None
xCoordinatesOfMeasuringRectangles_bottomRight = None
yCoordinatesOfMeasuringRectangles_bottomRight = None

farthestPointList = []
XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList = None, None
detectionRadiusOfFarthestPointsFromCommonFarthestPoint = 200

lastCenterPointPositions = []
XCenterPointOfCenterPointList, YCenterPointOfCenterPointList = None, None
detectionRadiusOfNewCenterPointsFromCommonCenterPoint = 75

shouldCameraBeShown = True
countDownWhetherCameraShouldBeShown = 40


def returnCameraIndexes():
    """ checks the first 10 Camerainputs and returns an array containing the available inputs."""
    arr = []
    for index, _ in enumerate(range(10, 0, -1)):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
    return arr


def rescaleFrame(frame, wpercent=130, hpercent=130):
    """Rescales the frame to a given percentage"""
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def getContoursFromMaskedImage(maskedHistogramImage):
    """Returns the contours of a given masked Image"""
    grayscaledMaskedHistogramImage = cv2.cvtColor(maskedHistogramImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayscaledMaskedHistogramImage, 0, 255, 0)
    cont, hierarchyDontCare = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def drawMeasuringRectangles(frame):
    """Draws 'amountOfMeasuringRectangles' Rectangles on the given frame and returns the modified image"""
    rows, cols, dontCare = frame.shape
    global amountOfMeasuringRectangles, xCoordinatesOfMeasuringRectangles_topLeft, yCoordinatesOfMeasuringRectangles_topLeft, xCoordinatesOfMeasuringRectangles_bottomRight, yCoordinatesOfMeasuringRectangles_bottomRight

    # position messure points of hand histogram
    xCoordinatesOfMeasuringRectangles_topLeft = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    yCoordinatesOfMeasuringRectangles_topLeft = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    # define shape of drawn small rectangles | here 10x10
    xCoordinatesOfMeasuringRectangles_bottomRight = xCoordinatesOfMeasuringRectangles_topLeft + 10
    yCoordinatesOfMeasuringRectangles_bottomRight = yCoordinatesOfMeasuringRectangles_topLeft + 10

    # draw calculated rectangles
    for i in range(amountOfMeasuringRectangles):
        cv2.rectangle(frame,
                      (yCoordinatesOfMeasuringRectangles_topLeft[i], xCoordinatesOfMeasuringRectangles_topLeft[i]),
                      (yCoordinatesOfMeasuringRectangles_bottomRight[i],
                       xCoordinatesOfMeasuringRectangles_bottomRight[i]),
                      (0, 255, 0), 1)

    return frame


def createHistogramFromMeasuringRectangles(frame):
    """Creates a Histogram from the given frame and measuring Rectangles through combining said rectangles to one Image"""
    global xCoordinatesOfMeasuringRectangles_topLeft, yCoordinatesOfMeasuringRectangles_topLeft

    # convert cv2 bgr colorspace to hsv colorspace for easier handling
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # create new blank Region Of Interest matrix/image
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    # fill ROI with the sample rectangles
    for i in range(amountOfMeasuringRectangles):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[xCoordinatesOfMeasuringRectangles_topLeft[i]:
                                                    xCoordinatesOfMeasuringRectangles_topLeft[i] + 10,
                                          yCoordinatesOfMeasuringRectangles_topLeft[i]:
                                          yCoordinatesOfMeasuringRectangles_topLeft[i] + 10]

    # create a Hand histogram and normalize it
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # remove noise and retun
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def maskFrameWithHistogram(frame, hist):
    """Returns the given frame masked by the given Histogram. The mask is created using Circles drawn over the matching Pixels"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mask area that matches with the histogram via back projection
    histogramMaskBackProjection = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    cv2.imshow("histogramMaskedFrame_histogramBackProjection", histogramMaskBackProjection)

    maskingCircle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(histogramMaskBackProjection, -1, maskingCircle, histogramMaskBackProjection)

    ret, thresh = cv2.threshold(histogramMaskBackProjection, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    cv2.imshow("histogramMaskedFrame_thresh", thresh)

    return cv2.bitwise_and(frame, thresh)


def getCenterCoordinatesOfContour(maxContour):
    """Returns the Centercoordinates of a given contour in the shape  X, Y"""
    moment = cv2.moments(maxContour)
    if moment['m00'] == 0:
        return None
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])
    return cx, cy


def getFarthestPointFromContour(defects, contour, centroid):
    """Returns the farthest point from a given centerpoint on a contour using defects"""
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            return tuple(contour[farthest_defect][0])
        else:
            return None


def drawCirclesOnTraversedPoints(frame, traversedPoints):
    """Draws Circles on the given frame using coordinates contained in traversedPoints. The circles are ever decreasingly in size.
    Also draws a Circle around the common centerpoint of the traversedPoints."""
    if traversedPoints is not None:
        for i in range(len(traversedPoints)):

            radius = int(5 - (i * 15) / 200)
            if radius < 1:  # check if radius is 0
                radius = 1
            cv2.circle(frame, traversedPoints[i], radius, [0, 255, 255], -1)

            cv2.circle(frame, (XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList),
                       detectionRadiusOfFarthestPointsFromCommonFarthestPoint, [0, 255, 0], 1)


def calculateCommonCenterPointOfPointlist(pointList) -> (int, int):
    """Calculates the Common centerpoint of a given list.
    Sums up all X coordinates and divides them by their number"""

    SumOfAllXCoordinates = 0
    SumOfAllYCoordinates = 0

    if pointList is not None:
        for i in range(len(pointList)):
            SumOfAllXCoordinates += pointList[i][0]
            SumOfAllYCoordinates += pointList[i][1]

        if len(pointList) > 0:
            return (SumOfAllXCoordinates // len(
                pointList), SumOfAllYCoordinates // len(pointList))


def isPointInRangeOfOfOtherPoint(givenX1, givenY1, givenX2, givenY2, givenRange):
    """Checks whether Point 1 is within the defined range of Point 2"""
    if np.abs(np.sqrt(np.square(givenX1 - givenX2) + np.square(
            givenY1 - givenY2))) <= givenRange:
        return True
    return False


def resetWhetherCameraShouldBeShownCountdownTimer():
    global countDownWhetherCameraShouldBeShown, shouldCameraBeShown
    countDownWhetherCameraShouldBeShown = 40
    shouldCameraBeShown = True


def evaluateFrame(frame, hand_hist):
    """Evaluates the given frame using the given histogram to find the Center of
    an area that matches with the histogram. Also finds the farthest point within the
    matching area, to make pointy things e.g. a pointing finger out.
    These special Areas are Marked with a colored dot"""
    global shouldCameraBeShown, countDownWhetherCameraShouldBeShown, lastCenterPointPositions, XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList, XCenterPointOfCenterPointList, YCenterPointOfCenterPointList
    maskedHistogramImage = maskFrameWithHistogram(frame, hand_hist)

    cv2.imshow("evaluateFrame_noisyImage", maskedHistogramImage)

    # reduce noise
    maskedHistogramImage = cv2.erode(maskedHistogramImage, None, iterations=2)
    maskedHistogramImage = cv2.dilate(maskedHistogramImage, None, iterations=2)

    cv2.imshow("evaluateFrame_noiseReducedImage", maskedHistogramImage)

    contourList = getContoursFromMaskedImage(maskedHistogramImage)

    # check whether the contourList is emtpy AKA no hand is seen in the frame AKA dots all around the frame
    if contourList:
        maxCont = max(contourList, key=cv2.contourArea)

        centerOfMaxCont = getCenterCoordinatesOfContour(maxCont)
        if centerOfMaxCont is not None:

            #build up centerPointList
            if len(lastCenterPointPositions) < 15:
                lastCenterPointPositions.append(centerOfMaxCont)
                XCenterPointOfCenterPointList, YCenterPointOfCenterPointList = calculateCommonCenterPointOfPointlist(lastCenterPointPositions)
                cv2.circle(frame, centerOfMaxCont, 5, [255, 0, 255], -1)
                resetWhetherCameraShouldBeShownCountdownTimer()

            #check whether new centerpoint is within a given tolerance range
            elif isPointInRangeOfOfOtherPoint(centerOfMaxCont[0], centerOfMaxCont[1], XCenterPointOfCenterPointList, YCenterPointOfCenterPointList, detectionRadiusOfNewCenterPointsFromCommonCenterPoint):
                lastCenterPointPositions.append(centerOfMaxCont)
                lastCenterPointPositions.pop(0)
                XCenterPointOfCenterPointList, YCenterPointOfCenterPointList = calculateCommonCenterPointOfPointlist(lastCenterPointPositions)
                cv2.circle(frame, centerOfMaxCont, 5, [255, 0, 255], -1)
                resetWhetherCameraShouldBeShownCountdownTimer()

            else:
                countDownWhetherCameraShouldBeShown -= 1
                if countDownWhetherCameraShouldBeShown <= 0:
                    shouldCameraBeShown = False

        if maxCont is not None:
            hull = cv2.convexHull(maxCont, returnPoints=False)
            defects = cv2.convexityDefects(maxCont, hull)
            farthestPoint = getFarthestPointFromContour(defects, maxCont, centerOfMaxCont)
            # print("Centroid : " + str(centerOfMaxCont) + ", farthest Point : " + str(farthestPoint))

            #Build up farthest point list
            if len(farthestPointList) < 25:
                farthestPointList.append(farthestPoint)
                cv2.circle(frame, farthestPoint, 5, [0, 0, 255], -1)
                XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList = calculateCommonCenterPointOfPointlist(farthestPointList)

            #check if new farthest point is within a given tolerance range
            elif isPointInRangeOfOfOtherPoint(farthestPoint[0], farthestPoint[1], XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList, detectionRadiusOfFarthestPointsFromCommonFarthestPoint):
                farthestPointList.pop(0)
                farthestPointList.append(farthestPoint)
                cv2.circle(frame, farthestPoint, 5, [0, 0, 255], -1)
                XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList = calculateCommonCenterPointOfPointlist(farthestPointList)


            drawCirclesOnTraversedPoints(frame, farthestPointList)


def main():
    global handHistogram, detectionRadiusOfFarthestPointsFromCommonFarthestPoint
    isHandHistogramCreated = False
    isImageFlipped = False
    # capture video
    capture = cv2.VideoCapture(returnCameraIndexes()[0])

    sct = mss()
    monitor = sct.monitors[1]
    mon = {'top': 0, 'left': 0, 'width': monitor["width"] / 2, 'height': monitor["height"] / 2, "mon": 0}

    while capture.isOpened():
        # Read Monitor
        screen = sct.grab(monitor)
        screen = cv2.resize(np.array(screen), (int(mon["width"]), int(mon["height"])), interpolation=cv2.INTER_AREA)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        # Read Camera
        cam = capture.read()[1]
        cam = cv2.resize(np.array(cam), (640, 360), interpolation=cv2.INTER_AREA)

        x_offset = 0
        y_offset = 0
        output = screen
        if shouldCameraBeShown:
            output[y_offset:y_offset + cam.shape[0], x_offset:x_offset + cam.shape[1]] = cam

        #cv2.imshow('main_screen_with_PIP_camera_w/_info', screen)

        # Fingerdetection
        pressedKey = cv2.waitKey(1)
        dontCare, frame = capture.read()

        # flip image if f is pressed
        if pressedKey & 0xFF == ord('f'):
            isImageFlipped = not isImageFlipped

        if isImageFlipped:
            frame = cv2.flip(frame, 1)

        # capture handhistogram if 'z' is pressed
        if pressedKey & 0xFF == ord('z'):
            handHistogram = createHistogramFromMeasuringRectangles(frame)
            isHandHistogramCreated = True

        # enlargen or shrink detection radius if + or - is pressed
        if pressedKey & 0xFF == ord('+'):
            detectionRadiusOfFarthestPointsFromCommonFarthestPoint += 10

        if pressedKey & 0xFF == ord('-'):
            detectionRadiusOfFarthestPointsFromCommonFarthestPoint -= 10

        # TODO ADD RESET FEATURE

        if isHandHistogramCreated:
            evaluateFrame(frame, handHistogram)

        # Draw rectangles for Handhistogram capture
        else:
            frame = drawMeasuringRectangles(frame)

        cv2.imshow("main_camera_with_info", rescaleFrame(frame))

        if pressedKey == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
