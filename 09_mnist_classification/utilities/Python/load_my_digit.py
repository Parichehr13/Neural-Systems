import numpy as np
from PIL import Image


def rgb2gray(rgb):
    ''' Function that returns a gray-scale image from an input RGB image

     Arguments
     ---------
     rgb: ndarray
           Numpy array containing the RGB image (3-D).
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def load_my_digit(fpath):
    ''' Load and pre-process a handwritten digit contained in a
     picture. The digit is returned as a 28x28 uint8 ndarray.

     Arguments
     ---------
     fpath: string
           Filepath of the target image.

     Author
     ------
     Davide Borra, 2022
    '''
    my_digit = Image.open(fpath)
    my_digit = np.array(my_digit)
    my_digit = rgb2gray(my_digit)

    my_digit = 255-my_digit # inverting dynamic (higher values: foreground, low values: background)
    my_digit[my_digit<128]=0 # roughly bringing to 0 all background

    my_digit = np.array(Image.fromarray(my_digit).resize((28,28)))
    my_digit[my_digit<0] = 0 # saturating to 0 and 255 interpolated values exceeding 0-255 range
    my_digit[my_digit>255] = 255
    return my_digit.astype(np.uint8)
