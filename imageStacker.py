"""
====================================
Filename:         imageStacker.py 
Author:              Joseph Farah 
Description:       Stacks images from a folder, aligns them, and takes their mean/median
====================================
Notes
     
"""

#------------- imports -------------#
import sys
import copy
import glob 
import numpy as np 
from PIL import Image
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

#------------- global variables -------------#
valve = None

#------------- classes -------------#
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def warning(message):
        print bcolors.WARNING + "[" + message + "]" + bcolors.ENDC

    @staticmethod
    def success(message):
        print bcolors.OKGREEN + "[" + message + "]" + bcolors.ENDC

    @staticmethod
    def failure(message):
        print bcolors.FAIL + "[" + message + "]" + bcolors.ENDC


#------------- functions -------------#
def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def main():
    """
        Main function flow execution. 
    
        Args:
            none (none): none
    
        Returns:
            none (none): none
    
    """
    
    bcolors.warning("Commencing program")

    #------------- get command line arguments -------------#
    _folder_path = sys.argv[1]
    stackFunction = sys.argv[2]

    #------------- load all the images -------------#
    images = []
    for _file in glob.glob(_folder_path + "/*.JPG"):
        images.append(plt.imread(_file))

    #------------- get center from shape -------------#
    baseImage = images[0]
    trueCenter = (int(baseImage.shape[1]/2.), int(baseImage.shape[0]/2.))

    print "[trueCenter: {0}]".format(trueCenter)

    #------------- align -------------#
    alignedImages = []
    for image in images:

        ## get image center ##
        belowThreshold = np.where(image < 0.85*np.max(image))
        imageCopy = copy.copy(image)
        imageCopy[belowThreshold] = 0
        imageCenter = map(int, ndimage.measurements.center_of_mass(imageCopy))
        imageCenter = (imageCenter[1], imageCenter[0])
        print "[imageCenter: {0}]".format(imageCenter)

        ## UNCOMMENT THIS TO SEE HOW THE CENTERING IS WORKING ##
        # plt.imshow(image)
        # plt.scatter(*imageCenter, c='red', label='ic')
        # plt.scatter(*trueCenter, label='true')
        # plt.legend()
        # plt.show()

        centerDiff = (-(imageCenter[1] - trueCenter[1]), -(imageCenter[0] - trueCenter[0]))
        print "[centerDiff: {0}]".format(centerDiff)

        alignedImage = np.roll(image, centerDiff, axis=(0, 1))

        alignedImages.append(alignedImage)

    avgim = np.median(alignedImages, axis=0)
    avgim = Image.fromarray(avgim.astype('uint8'))
    plt.imshow(avgim)
    plt.show()
    bcolors.success("Program terminated with no errors") 
    

## switchboard ##
if __name__ == '__main__':
    main()