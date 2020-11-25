import numpy as np
import helmholtz_machine as hm
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.animation as animation

# Load in MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# X = (np.concatenate((x_train, x_test), axis = 0).reshape((len(x_train)+len(x_test), 28*28)) > .5).astype(float)
X = ((np.concatenate((x_train, x_test), axis = 0)) > .5).astype(float) 
y = np.concatenate((y_train, y_test), axis = 0)
print(X.shape, y.shape)

# Isolate Ones
ones = X[np.where(y == 1)]
ones = (ones.reshape((len(ones), 28*28)) > .5).astype(float)
print(ones.shape)
plt.imshow(ones.reshape((len(ones),28,28))[0])
plt.show()

h = hm.helmholtz(.1, 784)
for image in tqdm(ones[:100]):
    h.train(image)

dreams = h.dreams
print(len(dreams)), len(dreams[0])

plt.figure()
plt.imshow(dreams[-1].reshape(28,28))
plt.show()

frames = [] # for storing the generated images
fig = plt.figure()
for dream in tqdm(h.dreams[::10]):
    dream = dream.reshape(28,28)
    frames.append([plt.imshow(dream,animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
# ani.save('movie.gif')
plt.show()