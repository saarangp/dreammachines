import numpy as np
import helmholtz as hm #helmholtz_machine
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.animation as animation

# Load in MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# X = (np.concatenate((x_train, x_test), axis = 0).reshape((len(x_train)+len(x_test), 28*28)) > .5).astype(float)
X = ((np.concatenate((x_train, x_test), axis = 0))).astype(float) 
y = np.concatenate((y_train, y_test), axis = 0)
print(X.shape, y.shape)

# Isolate Ones
digit = X[np.where(y == 1)]
digit = (digit.reshape((len(digit), 28*28))).astype(float)
print(digit.shape)
plt.imshow(digit.reshape((len(digit),28,28))[0]/255, cmap='gray')
plt.show()

h = hm.helmholtz([784,784], 'beta', .1)
for image in tqdm(digit[:100]):
    h.train(image)

dreams = h.dreams
print(len(dreams)), len(dreams[0])

plt.figure()
plt.imshow(dreams[-1].reshape(28,28), cmap='gray')
plt.show()

frames = [] # for storing the generated images
fig = plt.figure()
for dream in tqdm(dreams[:1000:10]):
    dream = dream.reshape(28,28)
    frames.append([plt.imshow(dream,animated=True, cmap='gray')])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
ani.save('dreaming.gif')
print('Animation Creation Finished')

plt.show()