# Person-Protective-Equipment-Detection
This project endeavours to leverage state-of-the-art computer vision technology to detect the presence of protective equipment worn by individuals in construction site environments. By employing the YOLOv8 model, a powerful object detection algorithm, in conjunction with the Ultralytics framework, the project aims to enable real-time monitoring of safety equipment through live CCTV footage or video streams. To achieve this, the project integrates the YOLOv8 model with the OpenCV library for video processing and the CVZone library for overlaying informative annotations onto the video feed. Additionally, the Math library is employed for performing complex calculations related to pixel manipulation and geometric transformations.

## Work Flow of the Project
This Repo is divided into parts of Custom Training and Videos. Apart form those you have a "PPE_detection.py" python executable file which is the main file. The Custom Training Directory is used to store the files required for custom training of the YOLO V8 model and produce the "best.pt" file with the best encountered weights after training. 

###Recommended: Upload the whole Custom Training folder to your Drive and then run "Model_Custom_training.ipynb" file on Google Colab. 

The Videos directory contains some Demo Construction site Videos on which you can try your model out before connecting to the Webcam/Live CCTV  footage.

The "PPE_detection.py" file is having the lines of code which connect the Program to the Webcam / live CCTV footage. You just need to un-comment it and comment the Video one. Also remember to change the camera no. inside the "VideoCapture()" function. There '0' is ussually mapped to Webcam of the device. Just find the camera no. of your connected cctv and replace it with '0'.

###Refer: "https://www.scaler.com/topics/cv2-videocapture/" for help.
