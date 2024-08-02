# Model Analysis

This project explores a novel approach to enhancing Multi-Object Tracking capabilities in Computer Vision by separating detection from tracking. This method aims to save time and memory, particularly when analyzing videos with multiple regions of interest.

## Key Concept
Traditionally, detection by tracking is performed, which involves running the detection model and tracker multiple times for each region of interest. This process is time-consuming and memory-intensive.

### Proposed Approach
1. Single Detection Pass: Perform object detection only once on the source video.
2. Trim Results: Trim the detection results according to the regions of interest in the video.
3. Efficient Tracking: Pass the trimmed detections to the tracker for further analysis.

### Example
If you need to analyze 9 different regions in a frame, traditionally, you would run the detection model and tracker for each region. With this approach, you run the detection model once, trim the results for each region, and then pass these trimmed detections to the tracker. This method is faster and more efficient, saving both time and memory.
