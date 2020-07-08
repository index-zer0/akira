#!/usr/bin/python

import os, sys
from PIL import Image
import numpy as np

imagesDir = sys.argv[1]

if not os.path.exists(imagesDir):
    print(imagesDir + " is not a directory")
    exit(1)

for filename in os.listdir(imagesDir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_name = imagesDir + "/" + filename
        image = Image.open(image_name, 'r')
        pixel_values = np.asarray(image)
        np.savetxt(image_name[:-3] + "txt", pixel_values.ravel(), fmt='%d', delimiter=',')