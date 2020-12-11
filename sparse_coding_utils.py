import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_bar_images(n_samples, x_dim, y_dim):
    X = []
    for i in range(n_samples):
        img = np.zeros((x_dim, y_dim))
        x_bars = np.random.choice(np.arange(x_dim), np.random.randint(x_dim // 1.5), replace=False)
        y_bars = np.random.choice(np.arange(y_dim), np.random.randint(y_dim // 1.5), replace=False)
        y_bars = y_bars[:np.random.randint(y_dim)]
        for x in x_bars:
            img[x, :] += np.ones(y_dim)
        for y in y_bars:
            img[:, y] += np.ones(x_dim)
        img = img.reshape((x_dim * y_dim))
        X.append(img)

    return np.array(X)

# Gets MNIST images for a given number. (ie. if number=1, all images of 1 are returned)
def get_mnist_images(number=1):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    X = ((np.concatenate((x_train, x_test), axis = 0))).astype(float) 
    y = np.concatenate((y_train, y_test), axis = 0)

    if number == -1:
        digit = X
        digit = (digit.reshape((len(digit), 28*28))).astype(float)
        return digit
    # Isolate Ones
    digit = X[np.where(y == number)]
    digit = (digit.reshape((len(digit), 28*28))).astype(float)
    return digit


def get_natural_images(img_mat, size):
    dim = int(512 // size)
    images = []
    for n in range(10):
        for i in range(dim):
            for j in range(dim):
                y_dim = i * size
                x_dim = j * size
                next_img = img_mat[x_dim:x_dim+size, y_dim:y_dim+size, n]
                images.append(next_img.reshape(size * size))
    return np.array(images)

def get_flower_images(num_images = 400):
    n = num_images - 1
    ones_place = n % 10
    tens_place = (n // 10) % 10
    hund_place = (n // 100) % 10
    images = []
    for i in range(hund_place + 1):
        for j in range(tens_place + 1):
            for k in range(ones_place + 1):
                next_image = plt.imread("archive/natural_images/flower/flower_0%s%s%s.jpg" % (i, j, k))
                next_image = rgb2gray(next_image[:16, :16])
                next_image = next_image.reshape(16 * 16)
                images.append(next_image)
    return np.array(images)

def create_patchwork(Phi, x_dim, y_dim, shape):
    patchwork = np.zeros((x_dim * shape[0], y_dim * shape[1]))
    count = 0
    for i in range(x_dim):
        for j in range(y_dim):
            a = i*shape[0]
            b = j*shape[1]
            patchwork[a:a+shape[0], b:b+shape[1]] = Phi[:, count].reshape(shape[0], shape[1])
            count += 1
    return patchwork