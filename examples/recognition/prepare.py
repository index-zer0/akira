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

i = 0
for content in tqdm(data):
    response = requests.get(content['content'])
    img = Image.open(BytesIO(response.content))
    images.append([img, content['annotation']])
    img.save(directory + "/img" + str(i).zfill(4) + ".png")
    i = i + 1