import cv2
import numpy as np
from mss import mss


def returnCameraIndexes():
    # checks the first 3 Camerainputs and returns an array containing the available inputs.
    index = 0
    arr = []
    i = 3
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


# define a video capture object
vid = cv2.VideoCapture(returnCameraIndexes()[0])

np.set_printoptions(suppress=True)

# Ersten Monitor erkennen
sct = mss()
monitor = sct.monitors[1]
mon = {'top': 0, 'left': 0, 'width': monitor["width"] / 2, 'height': monitor["height"] / 2, "mon": 0}

while 1:
    # Monitor einlesen
    screen = sct.grab(monitor)
    screen = cv2.resize(np.array(screen), (int(mon["width"]), int(mon["height"])), interpolation=cv2.INTER_AREA)
    screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    # Kamera einlesen
    cam = vid.read()[1]
    cam = cv2.resize(np.array(cam), (640, 360), interpolation=cv2.INTER_AREA)

    x_offset = 50
    y_offset = 50
    output = screen
    output[y_offset:y_offset + cam.shape[0], x_offset:x_offset + cam.shape[1]] = cam

    cv2.imshow('test', screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
