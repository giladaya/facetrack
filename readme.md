# Face tracking demo
This is a demo for detecting and tracking multiple faces in real time.  
It uses the [jsfeat](https://inspirit.github.io/jsfeat/) CV library for the detection and the tracking.


[Live Demo](https://giladaya.github.io/facetrack/)


## How it works
- We start by running face detection algorithm on the current video frame.
- For each detected face, tracking points are randomly selected within the detected face region.
- We then track these points in the video using an optical flow algorithm, until a specified time interval has passed.  
- We then run the detection algorithm again and match the freshly detected faces to the existing ones using a minimum distance approach.  
- If a face was not re-detected, we prune it after several detection cycles.

## Compatibility
This demo uses getUserMedia to access the camera feed.  
It was tested on Chrome 52 (desktop and mobile) and Firefox 48 (the latest as of this writing)