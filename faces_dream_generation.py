import numpy as np
import helmholtz_machine as hm
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.animation as animation
from sklearn.datasets import fetch_olivetti_faces
import math

faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True)
n_samples, n_features = faces.shape
image_shape = (64, 64)
print(faces.shape)

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

temp = np.concatenate((faces_centered, faces_centered))
# print(max(temp[0]), min(temp[0]))
# temp = (temp > 0).astype(float)
print("Example Face")
plt.imshow(temp[0].reshape(image_shape), cmap = 'gray')

h = hm.helmholtz(.1, 4096)
for image in tqdm(temp[:100]):
    h.train(image)

dreams = h.dreams
print(len(dreams)), len(dreams[0])

plt.figure()
plt.imshow(dreams[-1].reshape(image_shape), cmap = 'gray')
plt.show()

frames = [] # for storing the generated images
fig = plt.figure()
for dream in tqdm(dreams[::100]):
    dream = dream.reshape(image_shape)
    frames.append([plt.imshow(dream,cmap = 'gray',animated=True)])

print('Animation Creation Started')
ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True)
ani.save('faces_dreaming.gif')
print('Animation Creation Finished')

plt.show()