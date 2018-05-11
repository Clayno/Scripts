import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.io
from scipy import ndimage

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import os
from PIL import Image

# input image dimensions
img_rows, img_cols, img_chns = 50, 200, 3
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 100
latent_dim = 256
intermediate_dim = 1024
epsilon_std = 1.0
epochs = 20


if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)


x = Input(batch_shape=(batch_size,) + original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 100 * 25, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 25, 100)
else:
    output_shape = (batch_size, 25, 100, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters, num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 50, 200)
else:
    output_shape = (batch_size, 50, 200, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.mean_squared_error(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


y = CustomVariationalLayer()([x, x_decoded_mean_squash])
vae = Model(x, y)
vae.compile(optimizer=Adam(lr=0.0005), loss=None)
vae.summary()

path = "samples"

folder = "samples"
min_num_images = 1000
image_files = os.listdir(folder)
dataset = np.ndarray(shape=(len(image_files), img_rows, img_cols, img_chns),
                         dtype=np.float32)
print(folder)
num_images = 0
for image in image_files:
  image_file = os.path.join(folder, image)
  try:
    im = Image.open(image_file).convert('RGB')
    image_data = np.array(im)
    dataset[num_images, :, :, :] = image_data
    num_images = num_images + 1
  except IOError as e:
    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
dataset = dataset[0:num_images, :, :, :]
if num_images < min_num_images:
  raise Exception('Many fewer images than expected: %d < %d' %
                  (num_images, min_num_images))
    
print('Full dataset tensor:', dataset.shape)
print('Mean:', np.mean(dataset))
print('Standard deviation:', np.std(dataset))

x_train = dataset[:1000]
x_test = dataset[1000:]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


print('x_train.shape:', x_train.shape)
plot_model(vae, to_file="model_captcha_deconv.png", show_shapes=True)

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)


decoded_imgs = generator.predict(x_test_encoded)

plt.imshow(x_test[200])
plt.imshow(decoded_imgs[200])

print("[+] Generation performance")
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
number = np.random.randint(len(x_test) - n, high=None, size=None)
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i + number].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i + number].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# interpolation part
print("[+] Interpolation")
while True:
  r = raw_input("Press enter to generate image")
  if r != "" :
    break
  number = np.random.randint(len(x_test_encoded[number]), high=None, size=None)
  a = x_test_encoded[number]
  number = np.random.randint(len(x_test_encoded[number]), high=None, size=None)
  b = x_test_encoded[number]
  digit_size = 32
  n = 10  # how many digits we will display
  plt.figure(figsize=(10, 10))
  for i in range(n):  
      c = 0.1*i*a.reshape(latent_dim) + (1-(i*0.1))*b.reshape(latent_dim)
      transposed = generator.predict(c.reshape(1, latent_dim))
      # display reconstruction
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(transposed.reshape(32, 32, 3))
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()

print("[+] Image generation")
while True:
  r = raw_input("Press enter to generate image")
  if r != "" :
    exit()      
  digit_size = 32
  # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
  # to produce values of the latent variables z, since the prior of the latent space is Gaussian
  z_sample = np.random.normal(loc=0.0, scale=1.0, size=(1,latent_dim))
  x_decoded = generator.predict(z_sample)
  digit = x_decoded[0].reshape(digit_size, digit_size, 3)
  
  plt.figure(figsize=(10, 10))
  plt.imshow(digit)
  plt.show()