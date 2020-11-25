# import tensorflow_datasets as tfds

# mnist_train, mnist_test = tfds.load('mnist', split=['train', 'test'])
# mnist_train = tfds.as_numpy(mnist_train)
# mnist_test = tfds.as_numpy(mnist_test)
# print(mnist_train.shape)
# # images = []
# # label = []
# # for ex in mnist:
# #   images.append(ex['image'].reshape(784))
# #   label.append(ex['label'])

# # images = np.array(images)
# # label = np.array(label)

# import numpy as np
# import helmholtz_machine as hm

image = np.random.random(100)
h = hm.helmholtz(.1, 100)
h.train(image)
print(h.V_R)