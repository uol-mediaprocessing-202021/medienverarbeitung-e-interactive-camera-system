import tkinter as tk
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
from mss import mss


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
        return self._num_occurrences / elapsed_time


class MonitorGrabber(object):

    def __init__(self, src=1):
        self.src = sct.monitors[src]
        self.frame = sct.grab(self.src)
        self.frame = cv2.resize(np.array(self.frame), (1280, 720), interpolation=cv2.INTER_AREA)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            self.frame = sct.grab(self.src)
            self.frame = cv2.resize(np.array(self.frame), (1280, 720), interpolation=cv2.INTER_AREA)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)

    def stop(self):
        self.stopped = True


class CameraGrabber(object):

    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class VideoShower:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            # TODO NICHT CV2.imgshow, SONDERN
            # cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} FPS".format(iterations_per_sec),
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


# Show App Loading Screen
loading = CameraGrabber("loading.avi").start()

app = tk.Tk()
app.geometry()

# Ersten Monitor erkennen
sct = mss()

MonitorIndex = 1
CameraIndex = 0

# Monitor Dropdown erstellen
Monitors = []

for mon in range(1, len(sct.monitors)):
    monitor = sct.monitors[mon]
    Monitors.append("Monitor " + str(mon) + ": " + str(monitor["width"]) + "x" + str(monitor["height"]))


def getMonitorDropDownValue():
    return int(monitorDropDownValue.get()[8:monitorDropDownValue.get().index(":")])


monitorDropDownValue = tk.StringVar()
monitorDropDownValue.set(Monitors[0])

# Camera DropdownMenu erstellen

Cameras = []

# checks the first 3 Camera inputs and returns an array containing the available inputs.
index = 0
i = 3
# Ready with loading
loading.stop()

while i > 0:
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        _, img = cap.read()
        h, w, _c = img.shape
        Cameras.append("Kamera " + str(index) + ": " + str(w) + "x" + str(h))
        cap.release()
    index += 1
    i -= 1


def getCameraDropDownValue():
    return int(cameraDropDownValue.get()[7:cameraDropDownValue.get().index(":")])


cameraDropDownValue = tk.StringVar()
cameraDropDownValue.set(Cameras[0])

# Gui erstellen
dropdowns = tk.Frame(app)
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
dropdowns.pack(side=tk.TOP)

app.geometry('1280x720')

oldMonitorDropDownValue = getMonitorDropDownValue()
oldCameraDropDownValue = getCameraDropDownValue()

monitor_stream = MonitorGrabber(oldMonitorDropDownValue).start()
monitor_stream_view = VideoShower(monitor_stream.frame).start()
cps = CountsPerSec().start()

# Mainloop
while True:
    # Check Monitor Dropdown Value
    value = getMonitorDropDownValue()
    if value != oldMonitorDropDownValue:
        oldMonitorDropDownValue = value
        MonitorIndex = value
        print("Ausgewählter Monitor: " + str(MonitorIndex))

    # Check Camera DropDown Value
    value = getCameraDropDownValue()
    if value != oldCameraDropDownValue:
        oldCameraDropDownValue = value
        CameraIndex = value
        print("Ausgewählte Kamera: " + str(CameraIndex))

    if monitor_stream.stopped or monitor_stream_view.stopped:
        monitor_stream_view.stop()
        monitor_stream.stop()
        break

    frame = monitor_stream.frame
    # frame = putIterationsPerSec(frame, cps.countsPerSec())
    monitor_stream_view.frame = frame
    cps.increment()

    # TODO Normal Program Loop

    app.update_idletasks()
    app.update()
