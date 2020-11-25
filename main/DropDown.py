import tkinter as tk

import cv2
from mss import mss

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
monitorDropDownMenu.config(width=10)
monitorDropDownMenu.pack(side=tk.LEFT)
monitorDropDown.pack(side=tk.LEFT)
cameraDropDown = tk.Frame(dropdowns)
cameraDropDownLabel = tk.Label(cameraDropDown, text="Zu verwendende Kamera").pack(side=tk.LEFT)
cameraDropDownMenu = tk.OptionMenu(cameraDropDown, cameraDropDownValue, *Cameras)
cameraDropDownMenu.config(width=10)
cameraDropDownMenu.pack(side=tk.LEFT)
cameraDropDown.pack(side=tk.BOTTOM)
dropdowns.pack(side=tk.TOP)

app.geometry('1280x720')

oldMonitorDropDownValue = getMonitorDropDownValue()
oldCameraDropDownValue = getCameraDropDownValue()

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

    # TODO Normal Program Loop

    app.update_idletasks()
    app.update()
