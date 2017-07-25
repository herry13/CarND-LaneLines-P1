#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2

def process(fpath):
    img = cv2.imread(fpath)
    '''# White mask
    lower, upper = [200, 200, 200], [255, 255, 255]
    lower = np.array(lower, dtype='uint8')
    upper = np.array(upper, dtype='uint8')
    white_mask = cv2.inRange(img, lower, upper)
    # Yellow mask
    lower, upper = [200, 200, 0], [255, 255, 0]
    lower = np.array(lower, dtype='uint8')
    upper = np.array(upper, dtype='uint8')
    yellow_mask = cv2.inRange(img, lower, upper)
    # Combine white and yellow masks
    mask = white_mask + yellow_mask
    output = cv2.bitwise_and(img, img, mask=mask)'''
    output = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return np.hstack([img, output])

input_dir = 'test_images/'
output_dir = 'test_images_output/'
for fname in os.listdir(input_dir):
    if fname[0] == '.':
        continue
    fpath = input_dir + fname
    print fpath
    img = process(fpath)
    cv2.imwrite(output_dir + fname, img)
