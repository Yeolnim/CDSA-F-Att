# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:38:13 2020

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

# CDSA-F-att
data = [[0.97, 0.9486, 0.9486, 0.95, 0.9419],
        [0.9546, 0.97, 0.9613, 0.9603, 0.965],
        [0.9543, 0.9535, 0.97, 0.9601, 0.9586],
        [0.9611, 0.9651, 0.966, 0.97, 0.9671],
        [0.9653, 0.971, 0.9671, 0.968, 0.97]]

# CDSA-F
'''
data = [[0.97,0.9455,0.947,0.9403,0.9418],
        [ 0.9496,0.97,0.9568,0.9581,0.9596],
        [ 0.9546,0.9508,0.97,0.96 ,0.9498],
        [0.952 ,0.9628,0.9641,0.97,0.9628],
        [0.961,0.9665 ,0.9623,0.9691,0.97]]

'''

# Average
'''
data = [[0.9, 0.8664875,0.8267625,0.8347625,0.8611125],
        [ 0.8795625 ,0.9, 0.846175,0.8455625,0.8897375],
        [ 0.84455,0.8390625,0.9,0.8975375, 0.8400125],
        [ 0.8520875,0.858375,0.9003125,0.9, 0.8451125],
        [ 0.88575,0.8968375,0.8494125,0.8579375,	0.9]]
'''

# LSTM
'''
data = [[0.9	,0.9511	,0.9306,	0.9411,	0.9355],
        [0.9366,	0.9,	0.9163,	0.9441,	0.934],
        [0.9265,	0.9528,	0.9,	0.9401,	0.9176],
        [0.9401,	0.9576,	0.9315,	0.9,	0.9356],
        [0.9416,	0.9401,	0.9168,	0.9456,	0.9]]

'''

d = np.array(data)
d = d.transpose()
col = ['Book', 'DVD', 'Electronic', 'Kitchen', 'Video']
index = ['Book', 'DVD', 'Electronic', 'Kitchen', 'Video']
df = pd.DataFrame(d, columns=col, index=index)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
fig.colorbar(cax)

tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fontdict = {'rotation': 90}
ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
ax.set_yticklabels([''] + list(df.index))

plt.show()
