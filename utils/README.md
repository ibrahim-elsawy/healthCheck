# Quality 
- [getQuality](https://github.com/ibrahim-elsawy/heart_rate_mesure/blob/main/new/quality.py) takes the directory of image as argument and return true if image has acceptable quality. comparing it with threshold constant 8 which is chosen by testing many images.

# Heart Rate
-[face_utilities](https://github.com/ibrahim-elsawy/healthCheck/blob/main/utils/face_utilities.py) this file responsible for loading the landmark model and handle all face detection processing for calculating heart rate.
-[signal_processing](https://github.com/ibrahim-elsawy/healthCheck/blob/main/utils/signal_processing.py) this file contains the fast fourier transform used to get the fundemntal frequency of the variance of face color to calculate heart rate.
-[MainHR](https://github.com/ibrahim-elsawy/healthCheck/blob/main/utils/MainHR.py) reads the video frame by frame and use the utilities to get the heart rate.
-[tongueSeg](https://github.com/ibrahim-elsawy/healthCheck/blob/main/utils/tongueSeg.py) load the tongue segmentation model and get input of image of user tongue and use the model to segment the tongue then apply filter to get the six regions of the tongue then look for variation in tongue to get cracks. 
