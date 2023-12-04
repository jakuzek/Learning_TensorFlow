import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Creating tensors
string = tf.Variable("This is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)


# Rank / Degree of Tensors
# Another word for rank is degree, these terms simply mean the number of dimensions involved in the tensor.
# What we created above is a tensor of rank 0, also known as a scalar.
rank1_tensor = tf.Variable(["Test", "Ok", "Tim"], tf.string)
rank2_tensor = tf.Variable([ ["test", "ok", "Tim"], ["test", "yes", "Tim"] ], tf.string)

# To determine the rank of a tensor we can call the following method.
print(tf.rank(number))


# Shape of Tensors
# What a shape simply tells us is how many items we have in each dimension.
print(rank2_tensor.shape)



# Changing Shape
tensor1 = tf.ones([1,2,3])              # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
                                        # this will reshape the tensor to [3,2]

# The number of elements in the reshaped tensor MUST match the number in the original

print(tensor1)
print(tensor2)
print(tensor3)



# Types of Tensors
# - Variable
# - Constant
# - Placeholder
# - SparseTensor

# With the execption of Variable all of these tensors are immuttable, meaning their value may not change during execution.



# Evaluating Tensors
# There will be some times that we need to evaluate a tensor. In other words, get its value.
# Since tensors represent a partially complete computation we will sometimes need to run what's called a session to evaluate the tensor.
#with tf.Session() as sess:
#    tensor.eval()