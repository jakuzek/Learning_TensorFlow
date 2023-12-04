import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# Linear Regression
# Line of best fit refers to a line through a scatter plot of data points that best expresses the relationship between those points
import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
#plt.plot(x, y, 'ro')
#plt.axis([0, 6, 0, 20])
#plt.show()

plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()

# 3 dimentions
# If we have 2 values like x and y we can always predict the third one 'z'
# x,y -> z
# z,y -> x
# x,z -> y

