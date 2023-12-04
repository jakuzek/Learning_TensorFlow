import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

t = tf.zeros([5,5,5,5])
t = tf.reshape(t, [125, -1])
print(t)