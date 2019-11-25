import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

def gaussien_noise(images, f=0.1):
    '''
    Add gaussien noise to image
    prob: Probability of the noise
    '''
    output = images + f * np.random.normal(loc=0.0, scale=1.0, size=images.shape) 
    output /= np.max(output)
    return output
    
def predict_plot(autoencoder, imgs, n=10):
    encoded_imgs = imgs[:n]
    decoded_imgs = autoencoder.predict(encoded_imgs)
    plt.figure(figsize=(2 * n, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(encoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)