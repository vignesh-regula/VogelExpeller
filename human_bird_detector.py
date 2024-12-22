import cv2
import numpy as np
import time
import os
from tkinter import Tk, messagebox
import pygame

if not os.path.isfile('deploy.prototxt') or not os.path.isfile('mobilenet_iter_73000.caffemodel'):
    print("Required files not found. Make sure 'deploy.prototxt' and 'mobilenet_iter_73000.caffemodel' are in the same directory as this script.")
    exit()

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Class labels for the MobileNet SSD model
class_labels = {3: 'bird'}

cap = cv2.VideoCapture(0)

bird_detected = False
start_time = None
last_sound_time = 0 

alert_sound_path = 'alert_sound.mp3'

pygame.mixer.init()

try:
    alert_sound = pygame.mixer.Sound(alert_sound_path)
except pygame.error as e:
    print(f"Error loading sound: {e}")
    exit()

def show_alert():
    root = Tk()
    root.withdraw()
    messagebox.showinfo("Alert", "Bird has been present for more than 30 seconds!")
    root.destroy()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    bird_present = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx in class_labels and class_labels[idx] == 'bird':
                bird_present = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                label = f"{class_labels[idx]}: {confidence:.2f}"

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if bird_present:
        if not bird_detected:
            # Bird just detected, start the timer
            bird_detected = True
            start_time = time.time()

        current_time = time.time()
        if current_time - last_sound_time >= 1:
            alert_sound.play()
            last_sound_time = current_time
        elapsed_time = current_time - start_time
        print(f"Bird detected for {elapsed_time:.2f} seconds.")

        if elapsed_time > 30:
            show_alert() 
            bird_detected = False
            start_time = None
    else:
        if bird_detected:
            print("Bird left the frame.")
        bird_detected = False
        start_time = None
        last_sound_time = 0
        alert_sound.stop()

    cv2.imshow('Bird Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
