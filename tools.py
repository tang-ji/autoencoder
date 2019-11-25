import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import os
import glob

def generate_video(decoder, l, outputname):
    if outputname is None:
        outputname = 'transition{:3f}-{:3f}'.format(np.min(l), np.max(l))
    try:
        os.mkdir('imgs')
    except:
        pass
    pictures = decoder.predict(l)
    i = 0
    for p in pictures:
        plt.ioff()
        plt.figure(figsize=(28,28), dpi=28)
        plt.imshow(p.reshape(28, 28))
        plt.gray()
        plt.axis('off')
        plt.savefig('imgs/{:05d}.jpg'.format(i))
        plt.close()
        i += 1
        
    img_array = []
    for filename in glob.glob('imgs/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
    try:
        os.mkdir('videos')
    except:
        pass
    out = cv2.VideoWriter('videos/' + outputname + '.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

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