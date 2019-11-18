#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:10:39 2019

@author: logancross
"""

import matplotlib.pyplot as plt
import pandas as pd

teach_file_name = '/Users/logancross/Documents/DeepRL/AsymmPlay/project/torch/rl-starter-files/storage/test_teach/log.csv'
noteach_file_name = '/Users/logancross/Documents/DeepRL/AsymmPlay/project/torch/rl-starter-files/storage/test_no_teach/log.csv'

data_teach = pd.read_csv(teach_file_name)
data_noteach = pd.read_csv(noteach_file_name)

ax = plt.gca()
data_teach.plot(kind='scatter',x='update',y='rreturn_mean',color='red',ax=ax, label='teaching')
data_noteach.plot(kind='scatter',x='update',y='rreturn_mean',color='blue',ax=ax, label='no teaching')
plt.legend()
plt.show()

ax = plt.gca()
line.set_label('Label via method')
ax.legend()
plt.show()