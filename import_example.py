import pickle
import numpy as np
import tensorflow as tf
import PIL.Image as Image
from scipy.misc import imresize
import os
import dataset
import config
import matplotlib.pyplot as pyplot

# Initialize TensorFlow session.
tf.InteractiveSession()



with open('weight file here', 'rb') as file:
    G, D, Gs = pickle.load(file)

num = 1
filepath="your input filepath here"
files = os.listdir(filepath)
for file in files:
    image = Image.open(filepath + file)

    image = image.convert('RGB')

    image = np.array(image)

    image = image[150-128:150+128,150-128:150+128]

    image = Image.fromarray(image)

    image = imresize(image, (256, 256))

    image = np.array(image)


    image = np.transpose(image,(2,0,1))


    for i in range(0,1):

        latents = image

        latents = np.expand_dims(latents, axis = 0)

        # Generate dummy labels (not used by the official networks).
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

        # Run the generator to produce a set of images.
        images = Gs.run(latents, labels)

        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images[:,0:3] + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

        # Save images as PNG.
        for idx in range(images.shape[0]):
            Image.fromarray(images[idx], 'RGB').save('./results/img%d.png' % num)

    num += 1
