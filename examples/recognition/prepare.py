import json
import codecs
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd 
from io import BytesIO
from PIL import Image
import os
from pathlib import Path

data = []
images = []
fd_json = os.path.dirname(os.path.realpath(__file__)) + "/face_detection.json"
with codecs.open(fd_json, 'rU', 'utf-8') as js:
    for line in js:
        data.append(json.loads(line))

print(len(data), " images")
directory = os.path.dirname(os.path.realpath(__file__)) + "/images"

if not os.path.exists(directory):
    os.makedirs(directory)

f = open("images_info.txt", "w")
i = 0
for content in tqdm(data):
    response = requests.get(content['content'])
    img = Image.open(BytesIO(response.content)).convert('RGB')
    images.append([img, content['annotation']])
    img.save(directory + "/img" + str(i).zfill(4) + ".png")

    pixel_values = np.asarray(img)
    np.savetxt(directory + "/img" + str(i).zfill(4) + ".txt", pixel_values.ravel(), fmt='%d', delimiter=',')

    f.write('%s' % str(i).zfill(4))
    for face in content['annotation']:
        for point in face['points']:
            f.write(',{},{}'.format(point['x'], point['y']))
    f.write('\n')
    i = i + 1
f.close()