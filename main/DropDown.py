import tkinter as tk
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk
from mss import mss

# Gui Erstellen
app = tk.Tk()
app.geometry()
gui = tk.Frame(app)
gui.grid(row=0, column=0, pady=2)
imageViewer = tk.Frame(gui)
imageViewer.grid(row=1, column=0, sticky="N")


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

    def __init__(self, src=1, width=1280, height=720):
        self.setSrc(src)
        self.width = width
        self.height = height
        img = sct.grab(self.src)
        img = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
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
            self.frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def stop(self):
        self.stopped = True


class CameraGrabber(object):

    def __init__(self, src, width=1280, height=720):
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, img) = self.stream.read()
        self.frame = cv2.resize(np.array(img), (self.width, self.height), interpolation=cv2.INTER_AREA)
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
                self.frame = cv2.resize(cv2.UMat(img), (self.width, self.height), interpolation=cv2.INTER_AREA)

    def stop(self):
        self.stopped = True


class VideoShower:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None, width=1280, height=720):
        self.frame = frame
        self.stopped = False
        self.panel = None
        self.width = width
        self.height = height

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        try:
            img = cv2.resize(np.array(self.frame), (self.width, self.height), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            self.frame = ImageTk.PhotoImage(img)
            # if the panel is not None, we need to initialize it
            if self.panel is None:
                self.panel = tk.Label(imageViewer, image=self.frame)
                self.panel.image = self.frame
                self.panel.pack(side=tk.TOP, padx=10, pady=10)

            # otherwise, simply update the panel
            else:
                self.panel.configure(image=self.frame)
                self.panel.image = self.frame
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def stop(self):
        self.stopped = True


def putIterationsPerSec(frame, iterations_per_sec, x, y):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} FPS".format(iterations_per_sec),
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


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

oldMonitorDropDownValue = getMonitorDropDownValue()
oldCameraDropDownValue = getCameraDropDownValue()

monitor_stream = MonitorGrabber(oldMonitorDropDownValue, 1280, 720).start()
monitor_stream_view = VideoShower(monitor_stream.frame, 1230, 670)

camera_stream = CameraGrabber(oldCameraDropDownValue).start()

cps = CountsPerSec().start()

# Mainloop
while True:
    # Check Monitor Dropdown Value
    value = getMonitorDropDownValue()
    if value != oldMonitorDropDownValue:
        oldMonitorDropDownValue = value
        MonitorIndex = value
        monitor_stream.setSrc(value)
        print("Ausgewählter Monitor: " + str(MonitorIndex))

    # Check Camera DropDown Value
    value = getCameraDropDownValue()
    if value != oldCameraDropDownValue:
        oldCameraDropDownValue = value
        CameraIndex = value
        camera_stream.setSrc(value)
        print("Ausgewählte Kamera: " + str(CameraIndex))

    if monitor_stream.stopped or monitor_stream_view.stopped:
        monitor_stream_view.stop()
        monitor_stream.stop()
        break

    frame = monitor_stream.frame

    #  TODO Mach wat mit dem frame

    frame = putIterationsPerSec(frame, cps.countsPerSec(), 10, 700)
    monitor_stream_view.frame = frame
    monitor_stream_view.show()
    cps.increment()

    app.update_idletasks()
    app.update()
