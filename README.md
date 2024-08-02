# model_analysis

Here I am attempting a novel approach to Computer Vision's Multi-Object Tracking capabilities, the traditional method is to detect by tracking but here, seperating the detection from detection by tracking in order to save time and memory (given that you are analysing a video but with multiple region of interests), if you are analysing any computer vision project for object detection and tracking, we have to run the detection model and tracker, multiple times. This consumes lot of time and memory as well. To overcome this issue, we can pass the required detections to the tracker and perform the analysis faster than the existing approach. This process needs to perform the object detection only once on our source video, then passing the required region of interest objects to the tracker for the further analysis.

Example: 

Lets say, I am going to run 9 different region of interest in the frame. For this, I need to run detection model and tracker for 9 times, which consumes lot of memery and time as well. But here, we run the object detection model once and then trimming those results according to our region of interest of the video. Then pass these trimmed detections frame by frame to the tracker. This process is faster and efficient in saving time and memory.
