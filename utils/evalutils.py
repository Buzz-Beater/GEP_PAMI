"""
Created on 5/18/19

@author: Baoxiong Jia

Description:

"""
import numpy as np

def upsample(prediction, freq=10, length=None):
    upsampled_prediction = [i for i in prediction for _ in range(freq)]
    if length:
        if len(upsampled_prediction) > length:
            upsampled_prediction = upsampled_prediction[:length]
    return upsampled_prediction