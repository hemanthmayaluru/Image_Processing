#imports
import math, os
from PIL import Image
#Get directory of the image
path_to_image = os.getcwd()
image1 = 'clock.jpg'
image2 = ''
#image reading using PIL
im = Image.open(os.path.join(path_to_image, image1))
#im.show()
new_image = im.copy()
#getting the dimensions of the image - Width & Height
width, height = im.size
w2=width/2
h2=height/2
#Input the minimum and maximum thresholds
rmin = int(input('Enter rmin: '))
rmax = int(input('Enter rmax: '))
#Computing the new image function as defined in the project
for x in range(0, width):
    for y in range(0, height):
        distance=math.sqrt((x - w2)**2 + (y - h2)**2)
        pix = new_image.load()
        if rmin <= distance <= rmax: 
            pix[x, y] = 0
        else:
            pix[x, y] = pix[x, y]
im.show()
new_image.show()

#almost center: 2,5
#ideal: 40,80
#fills the clock: 5,105
#inner circle: 55,61
#outer ring : 60, 119





