import imquality.brisque as brisque
import PIL.Image


def getQuality(img):
    '''
    check the quality of image return true if good quality
    Args:
        img (PIL.image): the image needed to check quality 
    Outputs:
        quality (bool): true if quality of tongue image is good 
    '''

    qualityScore = brisque.score(img)
    return True if qualityScore < 9.5 else False


