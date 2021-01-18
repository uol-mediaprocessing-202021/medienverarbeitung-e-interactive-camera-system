import copy
import operator
import tkinter as tk
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageTk
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
detectionRadiusOfNewCenterPointsFromCommonCenterPoint = int(detectionRadiusOfFarthestPointsFromCommonFarthestPoint*3/4)

shouldCameraBeShown = True
countDownWhetherCameraShouldBeShown = 40
zoomValue = 1.0
maxZoomValue = 3.0

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# create windows
app = tk.Tk()
app.title("Interactive Camerasystem V0.5 BETA")
app.geometry()
gui = tk.Frame(app)
gui.grid(row=0, column=0, pady=2)
imageViewer = tk.Frame(gui)
imageViewer.grid(row=1, column=0, sticky="N")

monitorDropDownValue = None
cameraDropDownValue = None
Monitors = []
Cameras = []

# read monitors
sct = mss()

MonitorIndex = 1
CameraIndex = 0

pressed_key = ""


class ImageShower(object):
    """Creates another TKInter Window and shows the given Image
    """

    def __init__(self, name="Window", window=None):
        """
        Initialize a new ImageShower, by creating another TKInter Window and set its Name
        :param name:
        """
        if window is None:
            self.window = tk.Toplevel(app)
            self.window.title(name)
        else:
            self.window = window

        self.panel = None
        self.frame = None

    def update(self, image):
        """
        Update the Image witch will be shown in this Window
        :param image: The Image as cv2 Image in BGR
        """
        self.frame = image

    def show(self, width=640, height=360):
        """
        Shows the Image, witch has been already set by the Update Method or is given by an Optional Parameter
        :param frame: The Optional cv2 Image in BGR
        :param width: The Optional scaled Width of the Image
        :param height: The Optional scaled Height of the Image
        :return: None if no Image is given
        """
        if self.frame is None:
            return
        try:
            # Resize and Convert cv2 Image to TKInter Image
            img = cv2.resize(np.array(self.frame), (width, height), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            # if the panel is not None, we need to initialize it
            if self.panel is None:
                self.panel = tk.Label(self.window, image=img)
                self.panel.image = img
                self.panel.pack(side=tk.TOP)

            # otherwise, simply update the panel
            else:
                self.panel.configure(image=img)
                self.panel.image = img
        except RuntimeError:
            print("[INFO] caught a RuntimeError")
        except cv2.error:
            print("[DEBUG] Bildfehler! (Format richtig?)")


# Create Optional Windows for Debugging and Additional Infos
histogramWindow = ImageShower("Histogram")
histogramThreshWindow = ImageShower("Histogram mit Threshhold")
mainCameraWithInfo = ImageShower("Hauptkamera mit Infos")


class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        """
        Sets the Starttime and Frame-Counts to 0
        """
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        """
        Starts the Timer by saving the Current Time
        :return: Optional: The Own Object to create, start the Timer and save the Object at the same Time
        """
        self._start_time = datetime.now()
        return self

    def increment(self):
        """
        Adds One to the Frame Counter
        """
        self._num_occurrences += 1

    def countsPerSec(self):
        """
        Calculates the Frames per Second
        :return: The Frames per Second or 1 if time was to low
        """
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        if elapsed_time != 0:
            return self._num_occurrences / elapsed_time
        else:
            return 1


class MonitorGrabber(object):
    """
    Reads the Current Screen in another Thread and Stores it for easy Access
    """

    def __init__(self, src=1, width=1280, height=720):
        """
        Initialize a new MonitorGrabber
        :param src: MonitorIndex from mss
        :param width: Scaled Output Image width
        :param height: Scaled Output Image hight
        """
        self.setSrc(src)
        self.width = width
        self.height = height

        # Grab Monitor Image, Resize, Convert and Store it
        img = sct.grab(self.src)
        # noinspection PyTypeChecker
        img = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.picture = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.stopped = False

    def start(self):
        """
        Starts another Thread for its own get-Method, to grab the Image out of Mainloop
        :return:  Optional: The Own Object to create, start the Thread and save the Object at the same Time
        """
        Thread(target=self.get, args=()).start()
        return self

    def setSrc(self, src):
        """
        Re-Sets the Monitor Input Source Index of mss
        :param src: The new Monitor Index
        """
        self.src = sct.monitors[src]

    def get(self):
        """
        Grabs the current Monitor Image, Resize, convert and stores it
        """
        while not self.stopped:
            img = sct.grab(self.src)
            img = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.picture = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def stop(self):
        """
        Stops the MonitorGrabber-Get-Thread started by the start-Method
        """
        self.stopped = True


class CameraGrabber(object):
    """
    Reads the Current Camera-feed in another Thread and Stores it for easy Access
    """

    def __init__(self, src, width=1280, height=720):
        """
        Initialize a new CameraGrabber
        :param src: CameraIndex from mss
        :param width: Scaled Output Image width
        :param height: Scaled Output Image hight
        """
        self.width = width
        self.height = height

        # Grab Camera Image, Resize, Convert and Store it
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, img) = self.stream.read()
        self.picture = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.stopped = False

    def start(self):
        """
        Starts another Thread for its own get-Method, to grab the Image out of Mainloop
        :return:  Optional: The Own Object to create, start the Thread and save the Object at the same Time
        """
        Thread(target=self.get, args=()).start()
        return self

    def setSrc(self, src):
        """
        Re-Sets the Camera Input Source Index of mss
        :param src: The new Camera Index
        """
        self.stream = cv2.VideoCapture(src)

    def get(self):
        """
        Grabs the current Camera Image, Resize and stores it
        """
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, img) = self.stream.read()
                self.picture = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)

    def stop(self):
        """
        Stops the CameraGrabber-Get-Thread started by the start-Method
        """
        self.stopped = True


def putIterationsPerSec(frame, iterations_per_sec, x=10, y=30):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} FPS".format(iterations_per_sec),
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


# noinspection PyUnresolvedReferences
def getCameraDropDownValue():
    """
    Reads the selected Camera in the Dropdown Menue and parses it
    :return: The Camera Index based on cv2.VideoCapture()
    """
    return int(cameraDropDownValue.get()[7:cameraDropDownValue.get().index(":")])


# noinspection PyUnresolvedReferences
def getMonitorDropDownValue():
    """
    Reads the selected Monitor in the Dropdown Menue and parses it
    :return: The Monitor Index based on mss
    """
    return int(monitorDropDownValue.get()[8:monitorDropDownValue.get().index(":")])


def createMonitorAndCameraDropDownMenu():
    """
    Creates the Dropdown Menu for the Gui by checking the Avaiable Displays and Camera Inputs
    :return: None
    """
    # Create Monitor Dropdown
    global monitorDropDownValue, cameraDropDownValue, Monitors, Cameras

    # Grabing a Frame from each Monitor to Display Screen Resolution for each Monitor in Menu
    for mon in range(1, len(sct.monitors)):
        monitor = sct.monitors[mon]
        Monitors.append("Monitor " + str(mon) + ": " + str(monitor["width"]) + "x" + str(monitor["height"]))

    monitorDropDownValue = tk.StringVar()
    monitorDropDownValue.set(Monitors[0])

    # Create Camera DropdownMenu
    # checks the first 3 Camera inputs and returns an array containing the available inputs.
    index = 0

    while index <= 3:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            _, img = cap.read()
            h, w, _c = img.shape
            Cameras.append("Kamera " + str(index) + ": " + str(w) + "x" + str(h))
            cap.release()
        index += 1

    cameraDropDownValue = tk.StringVar()
    # TODO EXCEPTION IF NO CAMERA IS PRESENT
    cameraDropDownValue.set(Cameras[0])


def createGUI():
    """
    Creates the Gui with TKInter Elements
    :return: None
    """
    # Gui erstellen
    dropdowns = tk.Frame(gui)
    dropdowns.grid(row=0, column=0)
    monitorDropDown = tk.Frame(dropdowns)
    tk.Label(monitorDropDown, text="Zu verwendender Monitor").pack(side=tk.LEFT)
    monitorDropDownMenu = tk.OptionMenu(monitorDropDown, monitorDropDownValue, *Monitors)
    monitorDropDownMenu.config(width=30)
    monitorDropDownMenu.pack(side=tk.LEFT)
    monitorDropDown.pack(side=tk.LEFT)
    cameraDropDown = tk.Frame(dropdowns)
    tk.Label(cameraDropDown, text="Zu verwendende Kamera").pack(side=tk.LEFT)
    cameraDropDownMenu = tk.OptionMenu(cameraDropDown, cameraDropDownValue, *Cameras)
    cameraDropDownMenu.config(width=30)
    cameraDropDownMenu.pack(side=tk.LEFT)
    cameraDropDown.pack(side=tk.BOTTOM)
    app.geometry('1280x720')


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
    histogramMaskBackProjection = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 5)
    histogramWindow.update(histogramMaskBackProjection)
    # cv2.imshow("histogramMaskedFrame_histogramBackProjection", histogramMaskBackProjection)

    maskingCircle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    closedBackProjection = cv2.morphologyEx(histogramMaskBackProjection, cv2.MORPH_CLOSE,
                                            maskingCircle, iterations=2)

    # cv2.imshow("histogramMaskedFrame_closed", closedBackProjection)

    openedBackProjection = cv2.morphologyEx(closedBackProjection, cv2.MORPH_OPEN,
                                            maskingCircle, iterations=2)

    # cv2.imshow("histogramMaskedFrame_opened", openedBackProjection)

    cv2.filter2D(histogramMaskBackProjection, -1, maskingCircle, histogramMaskBackProjection)

    ret, thresh = cv2.threshold(openedBackProjection, 1, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    histogramThreshWindow.update(thresh)
    # cv2.imshow("histogramMaskedFrame_thresh", thresh)

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

            cv2.circle(frame, (XCenterPointOfCenterPointList, YCenterPointOfCenterPointList), detectionRadiusOfNewCenterPointsFromCommonCenterPoint, [0,0,255], 1)

        return frame


def calculateCommonCenterPointOfPointlist(pointList) -> (int, int):
    """Calculates the Common centerpoint of a given list.
    Sums up all X coordinates and divides them by their number"""

    if pointList is not None:
        SumOfAllXCoordinates = 0
        SumOfAllYCoordinates = 0

        for i in range(len(pointList)):
            SumOfAllXCoordinates += pointList[i][0]
            SumOfAllYCoordinates += pointList[i][1]

        if len(pointList) > 0:
            return (SumOfAllXCoordinates // len(
                pointList), SumOfAllYCoordinates // len(pointList))


def isPointInRangeOfOfOtherPoint(givenX1, givenY1, givenX2, givenY2, givenRange):
    """Checks whether Point 1 is within the defined range of Point 2"""
    return np.abs(np.sqrt(np.square(givenX1 - givenX2) + np.square(
        givenY1 - givenY2))) <= givenRange


def resetWhetherCameraShouldBeShownCountdownTimer():
    """
    Resets the Counter, witch defines weather the Camera-Input should be shown
    :return: None
    """
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

    # cv2.imshow("evaluateFrame_noisyImage", maskedHistogramImage)

    # reduce noise
    maskedHistogramImage = cv2.erode(maskedHistogramImage, None, iterations=2)
    maskedHistogramImage = cv2.dilate(maskedHistogramImage, None, iterations=2)

    # cv2.imshow("evaluateFrame_noiseReducedImage", maskedHistogramImage)

    contourList = getContoursFromMaskedImage(maskedHistogramImage)

    # check whether the contourList is emtpy AKA no hand is seen in the frame AKA dots all around the frame
    if contourList:
        maxCont = max(contourList, key=cv2.contourArea)

        centerOfMaxCont = getCenterCoordinatesOfContour(maxCont)
        if centerOfMaxCont is not None:

            # build up centerPointList
            if len(lastCenterPointPositions) < 15:
                lastCenterPointPositions.append(centerOfMaxCont)
                XCenterPointOfCenterPointList, YCenterPointOfCenterPointList = calculateCommonCenterPointOfPointlist(
                    lastCenterPointPositions)
                cv2.circle(frame, centerOfMaxCont, 5, [255, 0, 255], -1)
                resetWhetherCameraShouldBeShownCountdownTimer()

            # check whether new centerpoint is within a given tolerance range
            elif isPointInRangeOfOfOtherPoint(centerOfMaxCont[0], centerOfMaxCont[1], XCenterPointOfCenterPointList,
                                              YCenterPointOfCenterPointList,
                                              detectionRadiusOfNewCenterPointsFromCommonCenterPoint):
                lastCenterPointPositions.append(centerOfMaxCont)
                lastCenterPointPositions.pop(0)
                XCenterPointOfCenterPointList, YCenterPointOfCenterPointList = calculateCommonCenterPointOfPointlist(
                    lastCenterPointPositions)
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
            if farthestPoint is not None:
                # Build up farthest point list
                if len(farthestPointList) < 25:
                    farthestPointList.append(farthestPoint)
                    cv2.circle(frame, farthestPoint, 5, [0, 0, 255], -1)
                    XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList = calculateCommonCenterPointOfPointlist(
                        farthestPointList)

                # check if new farthest point is within a given tolerance range
                elif isPointInRangeOfOfOtherPoint(farthestPoint[0], farthestPoint[1], XCenterPointOfFarthestPointList,
                                                  YCenterPointOfFarthestPointList,
                                                  detectionRadiusOfFarthestPointsFromCommonFarthestPoint):
                    farthestPointList.pop(0)
                    farthestPointList.append(farthestPoint)
                    cv2.circle(frame, farthestPoint, 5, [0, 0, 255], -1)
                    XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList = calculateCommonCenterPointOfPointlist(
                        farthestPointList)

            return drawCirclesOnTraversedPoints(frame, farthestPointList)


def zoomOntoPointedRegion(frame, zoomFactor):
    """Zooms into the given Frame at the tip of the shown finger. This is achieved by drawing a vector from the
    centerpoint of the centerpointlist to the centerpoint of the farthestpointlist"""
    global XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList, XCenterPointOfCenterPointList, YCenterPointOfCenterPointList

    if frame is None:
        return

    vectorToNewFrameCenter = round((XCenterPointOfFarthestPointList - XCenterPointOfCenterPointList) * 1.5), round(
        (YCenterPointOfFarthestPointList - YCenterPointOfCenterPointList) * 1.5)

    xCenterOfNewFrame, yCenterOfNewFrame = XCenterPointOfCenterPointList + vectorToNewFrameCenter[
        0], YCenterPointOfCenterPointList + vectorToNewFrameCenter[1]

    # determine whether the vector is still in frame
    # take centerpoint of farthestpointlist alternatively

    if frame.shape[1] < xCenterOfNewFrame or 0 > xCenterOfNewFrame or frame.shape[
        0] < yCenterOfNewFrame or 0 > yCenterOfNewFrame:
        xCenterOfNewFrame, yCenterOfNewFrame = XCenterPointOfFarthestPointList, YCenterPointOfFarthestPointList

    # determine shown rectangle
    leftX, rightX = int(xCenterOfNewFrame - frame.shape[1] // zoomFactor // 2), int(xCenterOfNewFrame + frame.shape[
        1] // zoomFactor // 2)
    bottomY, topY = int(yCenterOfNewFrame - frame.shape[0] // zoomFactor // 2), int(yCenterOfNewFrame + frame.shape[
        0] // zoomFactor // 2)

    # determine whether shown rectangle is in frame
    # translate it otherwise

    if 0 > leftX or 0 > rightX:
        translateAmount = -leftX
        leftX, rightX = leftX + translateAmount, rightX + translateAmount
    elif frame.shape[1] < leftX or frame.shape[1] < rightX:
        translateAmount = frame.shape[1] - rightX
        leftX, rightX = leftX + translateAmount, rightX + translateAmount

    if 0 > bottomY or 0 > topY:
        translateAmount = -bottomY
        bottomY, topY = bottomY + translateAmount, topY + translateAmount
    elif frame.shape[0] < bottomY or frame.shape[0] < topY:
        translateAmount = frame.shape[0] - topY
        bottomY, topY = bottomY + translateAmount, topY + translateAmount

    frame = frame[int(bottomY):int(topY), int(leftX):int(rightX)]
    return frame


def key_pressed(event):
    global pressed_key
    pressed_key = event.char


def getGesturePredictionFromTensorflow(frame, model):
    if frame is None or model is None or type(frame) != np.ndarray or type(model) != tf.keras.Sequential:
        return "OTHER"
    h1 = frame.shape[0]
    w1 = frame.shape[1]

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    dimension = (224, 224)
    image = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    # print(prediction)
    predictionDictionary = {
        "LEFT": prediction[0][0],
        "RIGHT": prediction[0][1],
        "OTHER": prediction[0][2]
    }
    return max(predictionDictionary.items(), key=operator.itemgetter(1))[0]


def main():
    createMonitorAndCameraDropDownMenu()
    createGUI()

    oldMonitorDropDownValue = getMonitorDropDownValue()
    oldCameraDropDownValue = getCameraDropDownValue()

    # Starte die Threads um
    monitor_stream = MonitorGrabber(oldMonitorDropDownValue, 1280, 720).start()
    camera_stream = CameraGrabber(oldCameraDropDownValue, 640, 360).start()
    monitor_stream_view = ImageShower(monitor_stream.picture, imageViewer)

    cps = CountsPerSec().start()

    global handHistogram, detectionRadiusOfFarthestPointsFromCommonFarthestPoint, pressed_key, detectionRadiusOfNewCenterPointsFromCommonCenterPoint, zoomValue, maxZoomValue
    isHandHistogramCreated = False
    isImageFlipped = False

    # Load Tensorflow Model
    model = tf.keras.models.load_model('keras_model.h5')

    # Bind Key-Press-Event to Window
    app.bind("<Key>", key_pressed)

    while True:
        # Check Monitor Dropdown Value
        value = getMonitorDropDownValue()
        if value != oldMonitorDropDownValue:
            oldMonitorDropDownValue = value
            MonitorIndex = value
            monitor_stream.setSrc(MonitorIndex)
            print("Ausgewählter Monitor: " + str(MonitorIndex))

        # Check Camera DropDown Value
        value = getCameraDropDownValue()
        if value != oldCameraDropDownValue:
            oldCameraDropDownValue = value
            CameraIndex = value
            camera_stream.setSrc(CameraIndex)
            print("Ausgewählte Kamera: " + str(CameraIndex))

        # Read Monitor
        screen = copy.deepcopy(monitor_stream.picture)
        # Read Camera
        frame = copy.deepcopy(camera_stream.picture)
        cameraOriginalFrame = copy.deepcopy(camera_stream.picture)

        # cv2.imshow('main_screen_with_PIP_camera_w/_info', screen)

        # flip image if f is pressed
        if pressed_key == 'f':
            isImageFlipped = not isImageFlipped

        if isImageFlipped:
            cameraOriginalFrame = cv2.flip(cameraOriginalFrame, 1)
            frame = cv2.flip(frame, 1)

        # capture handhistogram if 'z' is pressed
        if pressed_key == 'z' and not isHandHistogramCreated:
            handHistogram = createHistogramFromMeasuringRectangles(frame)
            isHandHistogramCreated = True

        # enlargen or shrink detection radius if + or - is pressed
        if pressed_key == '+':
            detectionRadiusOfFarthestPointsFromCommonFarthestPoint = int(detectionRadiusOfFarthestPointsFromCommonFarthestPoint * 1.1)
            detectionRadiusOfNewCenterPointsFromCommonCenterPoint = int(detectionRadiusOfFarthestPointsFromCommonFarthestPoint*3/4)

        if pressed_key == '-' and (
                detectionRadiusOfFarthestPointsFromCommonFarthestPoint > 10 and detectionRadiusOfNewCenterPointsFromCommonCenterPoint > 10):
            detectionRadiusOfFarthestPointsFromCommonFarthestPoint = int(detectionRadiusOfFarthestPointsFromCommonFarthestPoint * 0.9)
            detectionRadiusOfNewCenterPointsFromCommonCenterPoint = int(detectionRadiusOfFarthestPointsFromCommonFarthestPoint*3/4)

        if pressed_key == 'r' and isHandHistogramCreated:
            handHistogram = None
            isHandHistogramCreated = False

        if pressed_key == 'a' and isHandHistogramCreated and zoomValue < maxZoomValue:
            zoomValue = zoomValue + 0.1

        if pressed_key == 'y' and zoomValue > 1:
            zoomValue = zoomValue - 0.1

        # fix eventual problems
        if zoomValue > maxZoomValue:
            zoomValue = maxZoomValue
        elif zoomValue < 1:
            zoomValue = 1

        pressed_key = ""

        if isHandHistogramCreated and frame is not None and cameraOriginalFrame is not None:
            try:
                frame = evaluateFrame(frame, handHistogram)

                # Evaluate Gesture with Tensorflow
                print(getGesturePredictionFromTensorflow(histogramThreshWindow.frame, model))

                # zoom onto the pointed region
                cameraOriginalFrame = zoomOntoPointedRegion(cameraOriginalFrame, zoomValue)
                cameraOriginalFrame = cv2.resize(cameraOriginalFrame, (frame.shape[1], frame.shape[0]),
                                                 interpolation=cv2.INTER_AREA)


            except RuntimeError:
                print("[INFO] caught a RuntimeError")
            except AttributeError:
                pass

        # Draw rectangles for Handhistogram capture
        else:
            frame = drawMeasuringRectangles(frame)
            cameraOriginalFrame = copy.deepcopy(frame)

        # Update the Camera-Feed
        if frame is not None:
            mainCameraWithInfo.update(frame)
        if shouldCameraBeShown and frame is not None:
            x_offset = 0
            y_offset = 0
            screen[y_offset:y_offset + cameraOriginalFrame.shape[0],
            x_offset:x_offset + cameraOriginalFrame.shape[1]] = cameraOriginalFrame
        frame = screen
        frame = putIterationsPerSec(frame, cps.countsPerSec(), 10, 700)
        monitor_stream_view.update(copy.deepcopy(frame))

        if pressed_key == 27:
            break

        monitor_stream_view.show(1230, 670)

        # Update little Windows
        histogramWindow.show()
        histogramThreshWindow.show()
        mainCameraWithInfo.show()

        cps.increment()

        app.update_idletasks()
        app.update()


if __name__ == '__main__':
    main()
