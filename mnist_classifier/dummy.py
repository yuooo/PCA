# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 14:11:25 2017

@author: jhh2677
"""

import numpy as np

A = np.array([[1,2], [1,4], [2,2], [1,2]])

B = np.array([1,2])

#print((A == B).all(1))

print(np.sum((A-B)**2, axis=1))

python pca.py


