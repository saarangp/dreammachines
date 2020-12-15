import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import ndimage
from skimage.transform import resize

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

def contrast_images(images):
    return images**2

def undo_contrast_images(images):
    return np.sqrt(images)

def blur_images(images, scale=3):
    return np.array([ndimage.gaussian_filter(img, scale) for img in images])
    
def sharpen_images(images, scale=5):
    return np.array([img + scale * (img - ndimage.gaussian_filter(img, 1)) for img in images])

def resize_images(images, scale):
    assert len(images.shape) > 2, "Must have at least dimension 3"
    if len(images.shape) == 4:
        return ndimage.zoom(images, (1, scale, scale, 1))
    else:
        return ndimage.zoom(images, (1, scale, scale))

def image_MSE(a, b):
    return (1 / (a.shape[0] * a.shape[0])) * (np.sum((a - b)**2))

def image_PSNR(a, b):
    return 20 * np.log10(255) - 10 * np.log10(image_MSE(255*a, 255*b))