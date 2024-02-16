import numpy as np
import tensorflow as tf
from tensorflow.python.ops.variables import trainable_variables

#set w initialize to 0 and the type is float32
w = tf.Variable(0,dtype=tf.float32) #the parameter you want to optimize
#use Adam optimization algorithm to optimize and set the learning rate to 0.1
optimizer = tf.keras.optimizers.Adam(0.1)

#define a function to loop
def train_step():
  #implement forward prop to compute cost function and tensorflow will automatically compute backward prop using gradient tape
  with tf.GradientTape() as tape:
    #define the cost function
    cost = w**2 -10*w +25 #a fixed function
  #define the trainable variables to do iteration training
  trainable_variables = [w]
  #compute the gradients
  grads = tape.gradient(cost,trainable_variables)
  #use optimizer to apply gradients
  optimizer.apply_gradients(zip(grads,trainable_variables))

#make sure w is 0
print(w)

#train one time
train_step()
print(w)

#train 1000 times
for i in range(1000):
  train_step()
print(w)

#if cost function is not a fixed function
w = tf.Variable(0,dtype=tf.float32)
#define x as a list of numbers as array and the type is float32(play the role of coefficients of cost function)
x = np.array([1.0,-10.0,25.0],dtype=np.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

def cost_fn():
  #same cost function with coefficients x
  return x[0]*w**2 +x[1]*w +x[2]

#check w is 0
print(w)

#take 1 step of the optimal algorithm
optimizer.minimize(cost_fn,[w])
print(w)

#define a function to loop
def training(x,y,optimizer):
  def cost_fn():
    return x[0]*w**2 +x[1]*w +x[2] #allowing tensorflow to construct a computation graph
  for i in range(1000):
    optimizer.minimize(cost_fn,[w])
  return w

#tensorflow will figure out backward propagation automatically to compute w
w = training(x,w,optimizer)
print(w)