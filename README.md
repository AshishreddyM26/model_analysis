# model_analysis

Here I am attempting a novel approach to Computer Vision's Multi-Object Tracking capabilities, that is, seperating the detection from detection and tracking in order to save time and memory, if you are analysing any computer vision project for object detection and tracking, we have to run the detection model and tracker, multiple times. This consumes lot of time and memory as well. To overcome this issue, we can pass the required detections to the tracker and perform the analysis faster than the existing method. This process needs to perform the object detection only once on our source video.

Example: 
Lets say, I am going to run 9 different region of interest in the frame. For this, we need to run multi object tracking (object detection model and tracker) 9 times, which consumes lot of memery and time as well. But here, we run the object detection model once and then trimming those results according to our region of interest of the video. Then pass these trimmed detections frame by frame to the tracker. This process is faster and efficient in saving time and memory.
