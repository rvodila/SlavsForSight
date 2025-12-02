# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:19:41 2024

@author: Lozano







Translated into python (chatGPT) and tested agains the original MATLAB
script. The mapping will be useful for instances and chanels, but needs
to be adapted regarding arrays and areas when using with Mr Nilson, since this one uses Lick or Ashton
maps.



"""


import matplotlib.pyplot as plt


def electrode_mapping(instanceInd, channelInd):
    """
    Assigns channels to the correct array and location on array, depending on
    the instance number and their order in the raw data file.
    
    Parameters:
    instanceInd (int): Instance number.
    channelInd (int): Channel index.
    
    Returns:
    tuple: (channelNum, arrayNum, area) where:
          - channelNum is the channel number after reordering.
          - arrayNum is the array number the channel belongs to.
          - area is the cortical area (V1 or V4) the array is implanted in.
    """
    if instanceInd < 5:
        channelOrder = list(range(97, 129)) + list(range(1, 97))  # DABC
        if channelInd <= 32 or channelInd > 96:
            arrayOrder = 1
        else:
            arrayOrder = 2
    else:
        channelOrder = list(range(65, 97)) + list(range(33, 65)) + list(range(1, 33)) + list(range(97, 129))  # CBAD
        if channelInd <= 32 or channelInd > 96:
            arrayOrder = 2
        else:
            arrayOrder = 1

    channelNum = channelOrder[channelInd - 1]
    if channelNum > 64:
        channelNum -= 64

    arrayNum = (instanceInd - 1) * 2 + arrayOrder  # determine to which of the 16 arrays the channel belongs

    V4Arrays = [2, 3]  # arrays implanted in V4
    V1Arrays = [1] + list(range(4, 17))  # arrays implanted in V1
    if arrayNum in V1Arrays:
        area = 'V1'
    else:
        area = 'V4'
    
    return channelNum, arrayNum, area



# Test examples
examples = [
    (1, 1),  # Instance 1, Channel 1
    (4, 128),  # Instance 4, Channel 128
    (5, 1),  # Instance 5, Channel 1
    (6, 64),  # Instance 6, Channel 64
    (3, 97)  # Instance 3, Channel 97
]

# Execute tests and print results
test_results = [electrode_mapping(*example) for example in examples]
test_results
