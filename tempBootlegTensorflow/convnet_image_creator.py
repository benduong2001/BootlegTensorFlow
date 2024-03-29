#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math as mt
import os

import numpy as np
# import pandas as pd
import matplotlib.pyplot as mpl
from PIL import Image

# In[3]:
from BTF_global_folderpaths import *

def bellcurve(x):
    return (-1)*((255)*(mt.e**(-(x**2)))) + 255
def impaled(x):
    return [x]
def discrete(x):
    return int(x)


# In[4]:


def make_image_1(l1, l2, u1, u2):
    j = float
    # l1, l2 = 3, 2
    # u1, u2 = 0.2, 0.2 #desmos domain for bellcurve. closer to 0 on y axis/farther from center = lighter
    variation_1 = [2, 1, 2] # drawn at center
    variation_2 = [1, 1, 3] # drawn between left and center
    variation_3 = [0, 1, 4] # drawn at left
    variation_4 = variation_2[::-1]
    variation_5 = variation_3[::-1]
    variations = [variation_1, variation_2, variation_3, variation_4, variation_5]
    vi = np.random.choice(list(range(5)), 1)
    variation = variations[int(vi)]
    image_array = []
    for _ in range(5):
        image_layer = []
        for _ in range(3):
            rowx1 = list(np.random.uniform(-l1,-l2, variation[0]).astype(j))
            rowx2 = list(np.random.uniform( -u1, u2, variation[1]).astype(j))
            rowx3 = list(np.random.uniform(l2, l1, variation[2]).astype(j))
            rowx = rowx1 + rowx2 + rowx3
            rowy = list(map(bellcurve,rowx))
            rowy = list(map(discrete, rowy))
            rowy = list(map(impaled, rowy))
            image_layer.append(np.array(rowy))
        image_layer = np.hstack(tuple(image_layer))
        image_array.append(image_layer)
    new_image_array = np.array(image_array).astype("uint8")
    return new_image_array


# In[5]:


def make_image_0(l1, l2, u1, u2):
    j = float
    image_array = []
    image_layer1 = []
    for _ in range(3):
        rowx1 = list(np.random.uniform(-u1, u2,5).astype(j))
        rowy = list(map(bellcurve,rowx1))
        rowy = list(map(discrete, rowy))
        rowy = list(map(impaled, rowy))
        image_layer1.append(np.array(rowy))
    image_layer1 = np.hstack(tuple(image_layer1))
    image_array.append(image_layer1)
    for _ in range(3):
        image_layer2 = []
        for _ in range(3):
            rowx1 = list(np.random.uniform(-u1,u2,1).astype(j))
            rowx2 = list(np.random.uniform( -l1, -l2, 3).astype(j))
            rowx3 = list(np.random.uniform(-u1, u2, 1).astype(j))
            rowx = rowx1 + rowx2 + rowx3
            rowy = list(map(bellcurve,rowx))
            rowy = list(map(discrete, rowy))
            rowy = list(map(impaled, rowy))
            image_layer2.append(np.array(rowy))
        image_layer2 = np.hstack(tuple(image_layer2))
        image_array.append(image_layer2)
    image_layer3 = []
    for _ in range(3):
        rowx1 = list(np.random.uniform(-u1, u2,5).astype(j))
        rowy = list(map(bellcurve,rowx1))
        rowy = list(map(discrete, rowy))
        rowy = list(map(impaled, rowy))
        image_layer3.append(np.array(rowy))
    image_layer3 = np.hstack(tuple(image_layer3))
    image_array.append(image_layer3)
    new_image_array = np.array(image_array).astype("uint8")
    return new_image_array


# In[7]:


def create_stockphotos0(l1, l2, u1, u2):
    for i in range(1, 101):
        new_image_array = make_image_0(l1, l2, u1, u2)
        new_image = Image.fromarray(new_image_array)
        file_name = IMAGE_DIRECTORY + "\conv0 ({0}).jpg".format(i)
        new_image.save(file_name,  'JPEG')


# In[9]:


def create_stockphotos1(l1, l2, u1, u2):
    for i in range(1, 101):
        new_image_array = make_image_1(l1, l2, u1, u2)
        new_image = Image.fromarray(new_image_array)
        file_name = IMAGE_DIRECTORY + "\conv1 ({0}).jpg".format(i)
        new_image.save(file_name,  'JPEG')
def create_stockphotos1_underfit(l1, l2, u1, u2):
    for i in range(1, 21):
        new_image_array = make_image_0(l1, l2, u1, u2)
        new_image = Image.fromarray(new_image_array)
        file_name = IMAGE_DIRECTORY + "\conv1 ({0}).jpg".format(i)
        new_image.save(file_name,  'JPEG')

# In[8]:

def proper_categories():
    create_stockphotos0(3, 1.5, .4, .4)
    create_stockphotos1(3, 1.5, .4, .4)
def red_herrings():
    create_stockphotos0(3, 3, 3, 3)
    create_stockphotos1(3, 3, 3, 3)
def underfit():
    create_stockphotos0(3, 1.5, .4, .4)
    create_stockphotos1_underfit(3, 1.5, .4, .4)
    

# Path 
# mode = 0o666
  
# Path
image_folder_exists = False

for di in os.listdir(FOLDERPATH):
    if os.path.isdir(os.path.join(FOLDERPATH, di)):
        if di == IMAGE_DIRECTORY:
            image_folder_exists = True
            print("convnet image folder already exists")
            break

if not image_folder_exists:
    path = os.path.join(FOLDERPATH, IMAGE_DIRECTORY) 
    try:  
        os.mkdir(path)  
    except OSError as error:  
        print(error)

    proper_categories()
    print("convnet image folder now created")
