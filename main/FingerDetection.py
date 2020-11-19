import cv2
import numpy as np

handHistogram = None
traversePoint = []
amountOfMeasuringRectangles = 9
xCoordinatesOfMeasuringRectangles_topLeft = None
yCoordinatesOfMeasuringRectangles_topLeft = None

xCoordinatesOfMeasuringRectangles_bottomRight = None
yCoordinatesOfMeasuringRectangles_bottomRight = None


def returnCameraIndexes():
    """ checks the first 10 Camerainputs and returns an array containing the available inputs."""
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


def rescale_frame(frame, wpercent=130, hpercent=130):
    """Rescales the frame to a given percentage"""
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def getContoursfromMaskedImage(maskedHistogramImage):
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
    histogramMask = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    # cv2.imshow("hist_masking_dst", histogramMask)

    maskingCircle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(histogramMask, -1, maskingCircle, histogramMask)

    ret, thresh = cv2.threshold(histogramMask, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    # cv2.imshow("hist_masking_thresh", thresh)

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def getFarthestPointFromContour(defects, contour, centroid):
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
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def drawCircles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (i * 15) / 200), [0, 255, 255], -1)

def evaluateFrame(frame, hand_hist):
    maskedHistogramImage = maskFrameWithHistogram(frame, hand_hist)

    cv2.imshow("evaluateFrame_noisyImage", maskedHistogramImage)

    #reduce noise
    maskedHistogramImage = cv2.erode(maskedHistogramImage, None, iterations=2)
    maskedHistogramImage = cv2.dilate(maskedHistogramImage, None, iterations=2)

    cv2.imshow("evaluateFrame_noiseReducedImage", maskedHistogramImage)


    contour_list = getContoursfromMaskedImage(maskedHistogramImage)
    maxCont = max(contour_list, key=cv2.contourArea)

    centerOfMaxCont = centroid(maxCont)
    cv2.circle(frame, centerOfMaxCont, 5, [255, 0, 255], -1)

    if maxCont is not None:
        hull = cv2.convexHull(maxCont, returnPoints=False)
        defects = cv2.convexityDefects(maxCont, hull)
        farthestPoint = getFarthestPointFromContour(defects, maxCont, centerOfMaxCont)
        print("Centroid : " + str(centerOfMaxCont) + ", farthest Point : " + str(farthestPoint))
        cv2.circle(frame, farthestPoint, 5, [0, 0, 255], -1)
        if len(traversePoint) < 20: # DONT PUT THIS NUMBER TOO HIGH! LONG LISTS RISK CALC FAILURE DURING CIRCLE DRAWING
            traversePoint.append(farthestPoint)
        else:
            traversePoint.pop(0)
            traversePoint.append(farthestPoint)

        drawCircles(frame, traversePoint)


def main():
    global handHistogram
    is_hand_hist_created = False
    # capture video
    capture = cv2.VideoCapture(returnCameraIndexes()[0])

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        dontCare, frame = capture.read()
        frame = cv2.flip(frame, 1)

        # capture handhistogram if 'z' is pressed
        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            handHistogram = createHistogramFromMeasuringRectangles(frame)

        if is_hand_hist_created:
            evaluateFrame(frame, handHistogram)

        # Draw rectangles for Handhistogram capture
        else:
            frame = drawMeasuringRectangles(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
