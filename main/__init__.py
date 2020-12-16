import copy
import tkinter as tk
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk
from mss import mss

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


class imageShower(object):

    def __init__(self, name="Window"):
        self.window = tk.Toplevel(app)
        self.window.title(name)
        self.panel = None
        self.frame = None

    def update(self, image):
        self.frame = image

    def show(self, frame=None, width=640, height=360):
        if frame is None:
            if self.frame is not None:
                frame = self.frame
            else:
                return
        try:
            img = cv2.resize(np.array(frame), (width, height), interpolation=cv2.INTER_AREA)
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
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")


histogramWindow = imageShower("Histogram")
histogramThreshWindow = imageShower("Histogram mit Threshhold")
mainCameraWithInfo = imageShower("Hauptkamera mit Infos")


class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        if elapsed_time != 0:
            return self._num_occurrences / elapsed_time
        else:
            return 1


class MonitorGrabber(object):

    def __init__(self, src=1, width=1280, height=720):
        self.setSrc(src)
        self.width = width
        self.height = height
        img = sct.grab(self.src)
        img = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.picture = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def setSrc(self, src):
        self.src = sct.monitors[src]

    def get(self):
        while not self.stopped:
            img = sct.grab(self.src)
            img = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.picture = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def stop(self):
        self.stopped = True


class CameraGrabber(object):

    def __init__(self, src, width=1280, height=720):
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, img) = self.stream.read()
        self.picture = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def setSrc(self, src):
        self.stream = cv2.VideoCapture(src)

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, img) = self.stream.read()
                self.picture = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)

    def stop(self):
        self.stopped = True


class VideoShower:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None, width=1280, height=720):
        self.picture = frame
        self.stopped = False
        self.panel = None
        self.width = width
        self.height = height

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        try:
            img = cv2.resize(np.array(self.picture), (self.width, self.height), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            self.picture = ImageTk.PhotoImage(img)
            # if the panel is not None, we need to initialize it
            if self.panel is None:
                self.panel = tk.Label(imageViewer, image=self.picture)
                self.panel.image = self.picture
                self.panel.pack(side=tk.TOP)

            # otherwise, simply update the panel
            else:
                self.panel.configure(image=self.picture)
                self.panel.image = self.picture
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def stop(self):
        self.stopped = True


def putIterationsPerSec(frame, iterations_per_sec, x=10, y=50):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} FPS".format(iterations_per_sec),
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


def getCameraDropDownValue():
    return int(cameraDropDownValue.get()[7:cameraDropDownValue.get().index(":")])


def getMonitorDropDownValue():
    return int(monitorDropDownValue.get()[8:monitorDropDownValue.get().index(":")])


def createMonitorAndCameraDropDownMenu():
    # Monitor Dropdown erstellen
    global monitorDropDownValue, cameraDropDownValue, Monitors, Cameras

    for mon in range(1, len(sct.monitors)):
        monitor = sct.monitors[mon]
        Monitors.append("Monitor " + str(mon) + ": " + str(monitor["width"]) + "x" + str(monitor["height"]))

    monitorDropDownValue = tk.StringVar()
    monitorDropDownValue.set(Monitors[0])

    # Camera DropdownMenu erstellen
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
    # Gui erstellen
    dropdowns = tk.Frame(gui)
    dropdowns.grid(row=0, column=0)
    monitorDropDown = tk.Frame(dropdowns)
    monitorDropDownLabel = tk.Label(monitorDropDown, text="Zu verwendender Monitor").pack(side=tk.LEFT)
    monitorDropDownMenu = tk.OptionMenu(monitorDropDown, monitorDropDownValue, *Monitors)
    monitorDropDownMenu.config(width=30)
    monitorDropDownMenu.pack(side=tk.LEFT)
    monitorDropDown.pack(side=tk.LEFT)
    cameraDropDown = tk.Frame(dropdowns)
    cameraDropDownLabel = tk.Label(cameraDropDown, text="Zu verwendende Kamera").pack(side=tk.LEFT)
    cameraDropDownMenu = tk.OptionMenu(cameraDropDown, cameraDropDownValue, *Cameras)
    cameraDropDownMenu.config(width=30)
    cameraDropDownMenu.pack(side=tk.LEFT)
    cameraDropDown.pack(side=tk.BOTTOM)
    app.geometry('1280x720')


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


class FingerDetector(object):

    def __init__(self):
        self.handHistogram = None
        self.amountOfMeasuringRectangles = 9
        self.xCoordinatesOfMeasuringRectangles_topLeft = None
        self.yCoordinatesOfMeasuringRectangles_topLeft = None
        self.xCoordinatesOfMeasuringRectangles_bottomRight = None
        self.yCoordinatesOfMeasuringRectangles_bottomRight = None

        self.farthestPointList = []
        self.XCenterPointOfFarthestPointList = None
        self.YCenterPointOfFarthestPointList = None
        self.detectionRadiusOfFarthestPointsFromCommonFarthestPoint = 200

        self.lastCenterPointPositions = []
        self.XCenterPointOfCenterPointList = None
        self.YCenterPointOfCenterPointList = None
        self.detectionRadiusOfNewCenterPointsFromCommonCenterPoint = 75

        self.shouldCameraBeShown = True
        self.countDownWhetherCameraShouldBeShown = 20
        self.updated = False
        self.started = False
        self.frame = None
        self.output = None
        self.idle = False
        self.fps = CountsPerSec()

    def isHandHistogramCreated(self):
        return self.handHistogram is not None

    def resetHandHistogram(self):
        self.handHistogram = None

    def start(self):
        if not self.started:
            self.started = True
            self.fps.start()
            Thread(target=self.loop, args=()).start()
            return self

    def stop(self):
        self.started = False

    def loop(self):
        while self.started:
            if self.updated and not self.idle and self.frame is not None and self.handHistogram is not None:
                self.idle = True
                self.evaluateFrame()
                self.idle = False
                self.updated = False

    def update(self, inputFrame):
        self.frame = copy.deepcopy(inputFrame)
        self.updated = True

    def getOutput(self):
        return self.output

    def drawMeasuringRectangles(self, frame):
        """Draws 'amountOfMeasuringRectangles' Rectangles on the given frame and returns the modified image"""
        rows, cols, dontCare = frame.shape

        # position messure points of hand histogram
        self.xCoordinatesOfMeasuringRectangles_topLeft = np.array(
            [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
             12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

        self.yCoordinatesOfMeasuringRectangles_topLeft = np.array(
            [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20,
             9 * cols / 20,
             10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

        # define shape of drawn small rectangles | here 10x10
        self.xCoordinatesOfMeasuringRectangles_bottomRight = self.xCoordinatesOfMeasuringRectangles_topLeft + 10
        self.yCoordinatesOfMeasuringRectangles_bottomRight = self.yCoordinatesOfMeasuringRectangles_topLeft + 10

        # draw calculated rectangles
        for i in range(self.amountOfMeasuringRectangles):
            cv2.rectangle(frame,
                          (self.yCoordinatesOfMeasuringRectangles_topLeft[i],
                           self.xCoordinatesOfMeasuringRectangles_topLeft[i]),
                          (self.yCoordinatesOfMeasuringRectangles_bottomRight[i],
                           self.xCoordinatesOfMeasuringRectangles_bottomRight[i]),
                          (0, 255, 0), 1)
        self.output = frame

    def createHistogramFromMeasuringRectangles(self, frame):
        """Creates a Histogram from the given frame and measuring Rectangles through combining said rectangles to one Image"""

        # convert cv2 bgr colorspace to hsv colorspace for easier handling
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # create new blank Region Of Interest matrix/image
        roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

        # fill ROI with the sample rectangles
        for i in range(self.amountOfMeasuringRectangles):
            roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[self.xCoordinatesOfMeasuringRectangles_topLeft[i]:
                                                        self.xCoordinatesOfMeasuringRectangles_topLeft[i] + 10,
                                              self.yCoordinatesOfMeasuringRectangles_topLeft[i]:
                                              self.yCoordinatesOfMeasuringRectangles_topLeft[i] + 10]

        # create a Hand histogram and normalize it
        hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

        # remove noise and retun
        self.handHistogram = cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

    def maskFrameWithHistogram(self):
        """Returns the given frame masked by the given Histogram. The mask is created using Circles drawn over the matching Pixels"""
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        # mask area that matches with the histogram via back projection
        histogramMaskBackProjection = cv2.calcBackProject([hsv], [0, 1], self.handHistogram, [0, 180, 0, 256], 5)
        histogramWindow.update(histogramMaskBackProjection)

        maskingCircle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

        closedBackProjection = cv2.morphologyEx(histogramMaskBackProjection, cv2.MORPH_CLOSE,
                                                maskingCircle, iterations=2)

        openedBackProjection = cv2.morphologyEx(closedBackProjection, cv2.MORPH_OPEN,
                                                maskingCircle, iterations=2)

        cv2.filter2D(histogramMaskBackProjection, -1, maskingCircle, histogramMaskBackProjection)

        ret, thresh = cv2.threshold(openedBackProjection, 1, 255, cv2.THRESH_BINARY)

        # thresh = cv2.dilate(thresh, None, iterations=5)

        thresh = cv2.merge((thresh, thresh, thresh))

        histogramThreshWindow.update(thresh)

        return cv2.bitwise_and(self.frame, thresh)

    def drawCirclesOnTraversedPoints(self, traversedPoints):
        """Draws Circles on the given frame using coordinates contained in traversedPoints. The circles are ever decreasingly in size.
        Also draws a Circle around the common centerpoint of the traversedPoints."""
        if traversedPoints is not None:
            for i in range(len(traversedPoints)):

                radius = int(5 - (i * 15) / 200)
                if radius < 1:  # check if radius is 0
                    radius = 1
                cv2.circle(self.frame, traversedPoints[i], radius, [0, 255, 255], -1)

                cv2.circle(self.frame, (self.XCenterPointOfFarthestPointList, self.YCenterPointOfFarthestPointList),
                           self.detectionRadiusOfFarthestPointsFromCommonFarthestPoint, [0, 255, 0], 1)
            return self.frame

    def resetWhetherCameraShouldBeShownCountdownTimer(self):
        self.countDownWhetherCameraShouldBeShown = 40
        self.shouldCameraBeShown = True

    def evaluateFrame(self):
        """Evaluates the given frame using the given histogram to find the Center of
        an area that matches with the histogram. Also finds the farthest point within the
        matching area, to make pointy things e.g. a pointing finger out.
        These special Areas are Marked with a colored dot"""
        maskedHistogramImage = self.maskFrameWithHistogram()

        # reduce noise
        maskedHistogramImage = cv2.erode(maskedHistogramImage, None, iterations=2)
        maskedHistogramImage = cv2.dilate(maskedHistogramImage, None, iterations=2)

        contourList = getContoursFromMaskedImage(maskedHistogramImage)

        # check whether the contourList is emtpy AKA no hand is seen in the frame AKA dots all around the frame
        if contourList:
            maxCont = max(contourList, key=cv2.contourArea)

            centerOfMaxCont = getCenterCoordinatesOfContour(maxCont)
            if centerOfMaxCont is not None:

                # build up centerPointList
                if len(self.lastCenterPointPositions) < 15:
                    self.lastCenterPointPositions.append(centerOfMaxCont)
                    self.XCenterPointOfCenterPointList, self.YCenterPointOfCenterPointList = calculateCommonCenterPointOfPointlist(
                        self.lastCenterPointPositions)
                    cv2.circle(self.frame, centerOfMaxCont, 5, [255, 0, 255], -1)
                    self.resetWhetherCameraShouldBeShownCountdownTimer()

                # check whether new centerpoint is within a given tolerance range
                elif isPointInRangeOfOfOtherPoint(centerOfMaxCont[0], centerOfMaxCont[1],
                                                  self.XCenterPointOfCenterPointList,
                                                  self.YCenterPointOfCenterPointList,
                                                  self.detectionRadiusOfNewCenterPointsFromCommonCenterPoint):
                    self.lastCenterPointPositions.append(centerOfMaxCont)
                    self.lastCenterPointPositions.pop(0)
                    self.XCenterPointOfCenterPointList, self.YCenterPointOfCenterPointList = calculateCommonCenterPointOfPointlist(
                        self.lastCenterPointPositions)
                    cv2.circle(self.frame, centerOfMaxCont, 5, [255, 0, 255], -1)
                    self.resetWhetherCameraShouldBeShownCountdownTimer()

                else:
                    self.countDownWhetherCameraShouldBeShown -= 1
                    if self.countDownWhetherCameraShouldBeShown <= 0:
                        self.shouldCameraBeShown = False

            if maxCont is not None:
                hull = cv2.convexHull(maxCont, returnPoints=False)
                defects = cv2.convexityDefects(maxCont, hull)
                farthestPoint = getFarthestPointFromContour(defects, maxCont, centerOfMaxCont)
                # print("Centroid : " + str(centerOfMaxCont) + ", farthest Point : " + str(farthestPoint))

                # Build up farthest point list
                if len(self.farthestPointList) < 25:
                    self.farthestPointList.append(farthestPoint)
                    cv2.circle(self.frame, farthestPoint, 5, [0, 0, 255], -1)
                    self.XCenterPointOfFarthestPointList, self.YCenterPointOfFarthestPointList = calculateCommonCenterPointOfPointlist(
                        self.farthestPointList)

                # check if new farthest point is within a given tolerance range
                elif isPointInRangeOfOfOtherPoint(farthestPoint[0], farthestPoint[1],
                                                  self.XCenterPointOfFarthestPointList,
                                                  self.YCenterPointOfFarthestPointList,
                                                  self.detectionRadiusOfFarthestPointsFromCommonFarthestPoint):
                    self.farthestPointList.pop(0)
                    self.farthestPointList.append(farthestPoint)
                    cv2.circle(self.frame, farthestPoint, 5, [0, 0, 255], -1)
                    self.XCenterPointOfFarthestPointList, self.YCenterPointOfFarthestPointList = calculateCommonCenterPointOfPointlist(
                        self.farthestPointList)

                self.output = self.drawCirclesOnTraversedPoints(self.farthestPointList)


def key_pressed(event):
    global pressed_key
    pressed_key = event.char


def main():
    createMonitorAndCameraDropDownMenu()
    createGUI()

    oldMonitorDropDownValue = getMonitorDropDownValue()
    oldCameraDropDownValue = getCameraDropDownValue()

    # Starte die Threads um
    monitor_stream = MonitorGrabber(oldMonitorDropDownValue, 1280, 720).start()
    camera_stream = CameraGrabber(oldCameraDropDownValue, 640, 360).start()
    monitor_stream_view = VideoShower(monitor_stream.picture, 1230, 670)
    fingerDetector = FingerDetector().start()

    cps = CountsPerSec().start()

    global pressed_key
    isImageFlipped = False

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

        if monitor_stream.stopped or monitor_stream_view.stopped:
            monitor_stream_view.stop()
            monitor_stream.stop()
            break

        # Read Monitor
        screen = copy.deepcopy(monitor_stream.picture)
        # Read Camera
        frame = copy.deepcopy(camera_stream.picture)
        cameraOriginal = copy.deepcopy(camera_stream.picture)

        # flip image if f is pressed
        if pressed_key == 'f':
            isImageFlipped = not isImageFlipped

        if isImageFlipped:
            cameraOriginal = cv2.flip(cameraOriginal, 1)

        # capture handhistogram if 'z' is pressed
        if pressed_key == 'z' and not fingerDetector.isHandHistogramCreated():
            fingerDetector.createHistogramFromMeasuringRectangles(frame)

        # enlargen or shrink detection radius if + or - is pressed
        if pressed_key == '+':
            fingerDetector.detectionRadiusOfFarthestPointsFromCommonFarthestPoint += 10

        if pressed_key == '-':
            fingerDetector.detectionRadiusOfFarthestPointsFromCommonFarthestPoint -= 10

        if pressed_key == 'r' and fingerDetector.isHandHistogramCreated():
            fingerDetector.resetHandHistogram()

        if pressed_key == 'q':
            camera_stream.stop()
            monitor_stream.stop()
            fingerDetector.stop()
            exit()

        pressed_key = ""

        if fingerDetector.isHandHistogramCreated() and not fingerDetector.idle:
            fingerDetector.update(frame)

        # Draw rectangles for Handhistogram capture
        else:
            fingerDetector.drawMeasuringRectangles(frame)

        # Update the Camera-Feed
        if fingerDetector.getOutput() is not None:
            mainCameraWithInfo.update(fingerDetector.getOutput())
        if fingerDetector.shouldCameraBeShown and fingerDetector.getOutput() is not None:
            x_offset = 0
            y_offset = 0
            screen[y_offset:y_offset + cameraOriginal.shape[0],
            x_offset:x_offset + cameraOriginal.shape[1]] = cameraOriginal
        screen = putIterationsPerSec(screen, cps.countsPerSec(), 10, 700)
        monitor_stream_view.picture = screen

        monitor_stream_view.show()

        # Update little Windows
        histogramWindow.show()
        histogramThreshWindow.show()
        mainCameraWithInfo.show()

        cps.increment()

        app.update_idletasks()
        app.update()


if __name__ == '__main__':
    main()
