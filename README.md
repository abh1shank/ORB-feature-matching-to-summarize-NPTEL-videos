# ORB-feature-matching-to-summarize-NPTEL-videos

ORB Similarity matching to extract relevant frames from a video.
Algorithm :

1) Convert the video into an array of frames, where each frame is a numpy array.

2) Compare ith and (i-1)th frame, and match their similarity using ORB Similarity matching. If there is decrement in similarity beyond a particular threshold it means there has been frame a change. Store the (i-1)th frame.

3) Compare every frame with every other frame to see if there are any any duplicates, this will also be done by ORB
similarly matching, if the similarity index is very high it would mean the frames are duplicate.

4) Use of YOLO to run inference on the remaining slides to see if there are any frames with the instructor present in them, those frames are also eliminated 

5) Rest are written into a pdf file.

Algorithm was run on a video clip of 10 min length, consisting of over 111,000 frames, It was able to extract 10 relevant frames from it.