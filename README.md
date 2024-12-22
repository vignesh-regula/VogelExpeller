# VogelExpeller
  This project aims to create a bird detection system using a webcam, leveraging a pre-trained MobileNet SSD (Single Shot Multibox Detector) model for object detection. 
 
# Libraries Used:
   OpenCV (cv2): 
   A powerful computer vision library to process images and videos in real-time.
         
    pip install opencv-python opencv-python-headless

   NumPy:
   A fundamental package for scientific computing in Python, used here to manipulate arrays.
         
    pip install numpy

   pygame :
   A set of Python modules designed for writing video games, which includes handling sound and other media files.
         
    pip install pygame
        
   Tkinter :
   A standard Python interface to the Tk GUI toolkit, used here to display a message box when a bird is detected for over 30 seconds.
       
    sudo apt-get install python3-tk
# MobileNet SSD (Single Shot Multibox Detector)
1. deploy.prototxt:
What is it?
The deploy.prototxt file is a prototxt file that defines the architecture of the neural network. It is written in a human-readable format, typically used by the Caffe framework (a deep learning framework developed by Berkeley AI Research). The prototxt file outlines the structure and configuration of the neural network.

2. mobilenet_iter_73000.caffemodel:
What is it?
The mobilenet_iter_73000.caffemodel is the trained model file that contains the weights learned by the MobileNet SSD model. It is a binary file that holds the values for all the parameters (weights and biases) of the neural network, which were optimized during the training process. The mobilenet_iter_73000.caffemodel file is generated after training the model for a number of iterations (in this case, 73,000 iterations) on a dataset.

deploy.prototxt: Defines the architecture (layers, input/output) of the MobileNet SSD object detection model.

mobilenet_iter_73000.caffemodel: Contains the pre-trained weights and biases of the MobileNet SSD model. These two files are critical for performing object detection, and in your project, they are used together to detect birds in real-time video frames using the webcam.

# Working Flow
1. Check for Required Files
The script checks if the necessary files (deploy.prototxt and mobilenet_iter_73000.caffemodel) are present in the working directory. If either file is missing, the script exits with an error message.

2. Load Pre-trained Model
The script loads the pre-trained MobileNet SSD model using OpenCV's cv2.dnn.readNetFromCaffe() method. This method loads both the model architecture (deploy.prototxt) and the trained weights (mobilenet_iter_73000.caffemodel).

3. Setup Video Capture
Initializes the webcam using cv2.VideoCapture(0) to capture video frames in real-time.

4. Initialize Sound System
The Pygame mixer is initialized to handle sound alerts. The script attempts to load the alert_sound.mp3 file. If an error occurs (e.g., the file is missing), the script exits.

5. Start Real-time Video Processing
The script enters an infinite loop to process each frame from the webcam:
Each frame is resized and converted into a "blob" using cv2.dnn.blobFromImage().
The blob is passed to the MobileNet SSD model for object detection using net.forward().

6. Object Detection
For each detection in the model’s output:
The script checks if the detected object is a "bird" (class label 3).
If a bird is detected with a confidence score greater than 50%, the script draws a bounding box around the bird and labels it with the confidence score.

7. Start Timer on First Bird Detection
If the bird is detected for the first time, the script starts a timer to track how long the bird remains in the frame.

8. Play Alert Sound and Track Time
If a bird is continuously detected:
The alert sound is played every second (using Pygame’s alert_sound.play()).
The script keeps track of how long the bird has been in the frame.

10. Trigger Alert After 30 Seconds
If the bird has been present for more than 30 seconds:
A pop-up alert is shown using Tkinter, notifying the user that the bird has been present for over 30 seconds.
The detection state is reset, and the timer stops.

11. Handle Bird Leaving Frame
If no bird is detected in the current frame:
The script resets the detection state, stops playing the sound, and clears the timer.

12. Display Frame
The processed frame, with any bounding boxes and labels, is displayed using cv2.imshow().

13. Exit on Keypress
The loop continues until the user presses the 'q' key, at which point the program exits.

14. Clean Up
After exiting the loop, the script releases the webcam and closes all OpenCV windows using cap.release() and cv2.destroyAllWindows().
# Conclusion:
This project uses a pre-trained MobileNet SSD model to detect birds in real-time from a webcam feed. When a bird is detected for over 30 seconds, an alert is triggered both through an on-screen message and an audible sound. This can be extended to detect other objects by modifying the class labels and adapting the code further

