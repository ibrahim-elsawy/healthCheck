The following are the two main functions for calculating the heart rate and checking the quality of image
# heart_rate_mesure

- [calcHR](https://github.com/ibrahim-elsawy/heart_rate_mesure/blob/main/new/main_HR.py) function process vedio and calcualtes the hr by passing vedio argument True by default and false for vedio streaming and takes directory of the vedio.
- it much better for the accuaracy to get vedio from user of length > 8sec.


# Quality 
- [getQuality](https://github.com/ibrahim-elsawy/heart_rate_mesure/blob/main/new/quality.py) takes the directory of image as argument and return true if image has acceptable quality. comparing it with threshold constant 8 which is chosen by testing many images.
