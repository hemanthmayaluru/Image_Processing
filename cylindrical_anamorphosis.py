#imports
import numpy as np
import scipy.misc as msc
from skimage import io
import scipy.ndimage as img
from cv2 import *
from PIL import Image

def image_anamorphosis():
    input_img = io.imread('bauckhage.jpg', flatten=True).astype('float')
    rows,cols = input_img.shape
    #center is 0 for cylindrical. change the center for torus to different values
    torus_center = 0
    torus_ext = torus_center + rows
    #Setting size of Output Image
    y = 2*torus_ext
    x = 2*torus_ext
    output_img = np.zeros((y,x))         
    
    #Calculate polar coordinates about the center of the destination image
    Y,X = np.meshgrid(range(y),range(x))
    R = np.sqrt(np.square(Y-y/2)  + np.square(X-x/2))
    theta = np.arctan2(X - x/2,Y - y/2)
    
    #The corresponding coordinates in the input image
    Y_input = rows - (R - torus_center)
    X_input = ((np.pi+theta)*(cols-1)/(2.*np.pi))
    output_img = img.map_coordinates(input_img,np.array([Y_input,X_input]),order=3)
    #resize the image
    output_img = resize(output_img,input_img.shape)
    Image.fromarray(output_img).show()
    msc.imsave('clock_anamorphosis_image.jpg', output_img)
    
if __name__ == '__main__':
    image_anamorphosis()
