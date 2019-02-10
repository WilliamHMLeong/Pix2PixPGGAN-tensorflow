import pickle
import numpy as np
import tensorflow as tf
import PIL.Image as Image
from scipy.misc import imresize
import dataset
import config
import matplotlib.pyplot as pyplot

# Initialize TensorFlow session.
tf.InteractiveSession()

#face_set = dataset.load_dataset(data_dir=config.face_dir, verbose=True, **config.faceset)

with open('E:/dp/data/result/287-pgan-celeba-face-feature-preset-v2-1gpu-fp32/network-snapshot-004220.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

num = 1

indix = [0 for i in range(512)]
for i in range(0, 512):
    indix[i] = int(i / 128)



print(indix )


def funclin(image):
    latents = image

    latents = tf.convert_to_tensor(latents)
    latents = tf.cast(latents, tf.float32)
    latents = tf.transpose(latents, (2, 0, 1))
    tmplatents = tf.reduce_mean(latents, 0)
    tmplatents = tf.reduce_mean(tmplatents, 0)
    tmplatents = tf.divide(tmplatents,25)
    tmplatents = tf.add(tmplatents,-1)
    #tmplatents = tf.add(tmplatents, -0.5)
    # tmplatents = tf.divide(tmplatents, 2)
    # tmplatents = tf.divide(tmplatents[:], 4)

    # tmplatents = tf.tile(tmplatents,[2])

    tmp = tf.gather(tmplatents, indix, axis=0)

  #  print("tmplatents")
  #  print(tmplatents)
    return tmp
    """
    latents1 = np.zeros((256,256))
    for i in range(0,3):
        latents1 = latents1 + latents[i]
    latents2 = np.zeros(256)
    for i in range(0,255):
        latents2 = latents2 + latents1[i]
    tmplatents = np.zeros(2*latents2.size)

    tmplatents = np.zeros(2 * latents.size)
    for i in range(0,latents.size):
        tmplatents[i] = latents[i]
        tmplatents[latents.size + i] = latents[i]

    return tmplatents
    """


# image.reshape(512)
# Generate latent vectors.

def npnormalize(x):
    offset = 0
    scale = 0.1
    variepslon = 0.0001
    xmean = np.mean(x, axis=0)
    xvira = np.var(x, axis=0)
    xmean = np.multiply(xmean, -1)
    tmpnorm = np.add(x, xmean)
    tmpvar = np.add(xvira, variepslon)
    tmpvar = np.sqrt(tmpvar)
    tmpnorm = np.divide(tmpnorm, tmpvar)
    xnorm = np.add(np.multiply(tmpnorm, scale), offset)
    return xnorm




# for i in range(512):
#    if tmplatents[i] != 255*255*3:

#       print(tmplatents[i])
#      #tmplatents1 = np.multiply(np.log(tmplatents/255/255/3),-1)
#     tmplatents = -np.log(tmplatents/255/255/3)
#    print(tmplatents[i])

# tmplatents.reshape(*Gs.input_shapes[0][1:])

def pixel_norm(x, epsilon=1e-8):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=0, keepdims=True) + epsilon)


sess = tf.Session()

for i in range (1,100):
    print(num)
    image = Image.open("E:/dp/img_highres_merge_resize/%06d.png"%num)

    #image = imresize(image,(4,4))
    image = imresize(image, (256, 256))


    image = np.array(image)
    image = np.transpose(image,[2,0,1])

    #image = (image[:, 0::2, 0::2] + image[:, 0::2, 1::2] + image[:, 1::2, 0::2] + image[:, 1::2, 1::2]) * 0.25
    #image = np.rint(image).clip(0, 255).astype(np.uint8)
    #Image.fromarray(image, 'RGB').save(
     #   'C:/Users/akila/Desktop\gan\progressive_growing_of_gans-master-retest/tmp/%06d.png' % num)
    #r, g, b = image.split()  # rgb通道分离
                # 注意：下面一定要reshpae(1024)使其变为一维数组，否则拼接的数据会出现错误，导致无法恢复图片
    #r_arr = np.array(r).reshape(256)
    #g_arr = np.array(g).reshape(256)
    #b_arr = np.array(b).reshape(256)
                # 行拼接，类似于接火车；最终结果：共n行，一行3072列，为一张图片的rgb值
    #image_arr = np.concatenate((r_arr, g_arr, b_arr))
    #result = np.concatenate((result, image_arr))
    # Import official CelebA-HQ networks.




   # print(can.eval(session=sess))
   # print("----")
    # tmplatents = tf.cast(tmplatents,tf.float32)
    # tmplatents = tf.multiply(tmplatents, 0.00382156)
    # tmplatents = tf.cast(tmplatents,tf.uint8)
    # tmplatents = tf.log(tmplatents)
    # tmplatents = tf.negative(tmplatents)
    # tmplatents = np.expand_dims(tmplatents, axis = 0)




    #latents = 0.8*np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) + 0.2*tmplatents/2550 # 1000 random latents
    for i in range(0,1):
        #tmplatents = np.expand_dims(tmplatents, 0)
        #tmplatents[0] = 10


        latents = image
        #latents = tf.convert_to_tensor(latents)


        #latents = pixel_norm(latents)

#        latents = latents.eval(session = sess)
        #a = 0.00025645134897566131548646513248979461674651798
  #      print(latents)
        print("------")
        #latents = tmplatents
        #latents = npnormalize((latents))
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
            Image.fromarray(images[idx], 'RGB').save('C:/Users/akila/Desktop\gan\progressive_growing_of_gans-master-retest/result/img%d.png' % num)

    num += 1
