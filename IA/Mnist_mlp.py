from __future__ import print_function

import sys
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from keras.layers.normalization import BatchNormalization

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha=34, sigma=4, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    
    if np.random.rand()*10 > 5:
        return image
    
    image = image.reshape(28,28)

    shape = image.shape
    if random_state is None:
        random_state = np.random.RandomState(None)

    

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(28,28,1)


filename = "D:\\Documents\\Cours\\INF6953H - Deep Learning\\TP1\\Result.csv"
filemodel = "D:\\Documents\\Cours\\INF6953H - Deep Learning\\TP1\\model.png"
batch_size = 100
num_classes = 10
epochs = 200
augmentation = False
rotation = False
elastic = False
shift = False
skip = False
modelNumber = 0
activation = "relu"

if len(sys.argv) > 1:
    modelNumber = int(sys.argv[1])
    rotation = "-r" in sys.argv
    if rotation :
        augmentation = True
        print("[+] Data augmentation : random rotation")
    elastic = "-e" in sys.argv
    if elastic :
        augmentation = True
        print("[+] Data augmentation : elastic transformation")
    shift = "-s" in sys.argv
    if shift :
        augmentation = True
        print("[+] Data augmentation : random shift")
    skip = "-S" in sys.argv
    if skip:
        activation = "sigmoid"
        modelNumber += 3
    if "--sigmoid" in sys.argv :
        activation = "sigmoid"
    print("[+] Model number " + str(modelNumber) + " used")
else:
    print("[-] Model number not specified")
    print("Usage : python Mnist_mlp.py ModelNumber [options]")

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Création du set de validation
x_val = x_train[int(60000*0.90):].reshape(-1,28, 28, 1)
y_val = y_train[int(60000*0.90):].reshape(-1,10)
x_train = x_train[:int(60000*0.90)].reshape(-1, 28, 28, 1)
y_train = y_train[:int(60000*0.90)].reshape(-1,10)

# Initialisation des callbacks
csv_logger = CSVLogger(filename)
early_stopping = EarlyStopping(monitor='val_acc', patience=3)

# Création des modèles
# Modèles sans skip connections
if modelNumber == 2:
    model = Sequential()
    model.add( Reshape((784,),input_shape=(28,28,1)))
    for i in range(1,3):
        model.add(Dense(2000, activation=activation))
        model.add(BatchNormalization())
    for i in range(1,3):
        model.add(Dense(1500, activation=activation))
        model.add(BatchNormalization())
    for i in range(1,3):
        model.add(Dense(1000, activation=activation))
        model.add(BatchNormalization())
    for i in range(1,3):
        model.add(Dense(500, activation=activation))
        model.add(BatchNormalization())
    for i in range(1,3):
        model.add(Dense(250, activation=activation))
        model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
elif modelNumber == 1:
    model = Sequential()
    model.add( Reshape((784,),input_shape=(28,28,1)))
    model.add(Dense(2000, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(1500, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(1000, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(500, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(250, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
elif modelNumber == 3:
    model = Sequential()
    model.add( Reshape((784,),input_shape=(28,28,1)))
    for i in range(1,13):
        model.add(Dense(512, activation=activation))
        model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
# Modèles avec skip connections
elif modelNumber == 5:
    inputs = Input(shape=(28,28,1))
    x = Reshape((784,),input_shape=(28,28,1))(inputs)
    for i in range(0,4):
        a = Dense((2000 - i*500), activation=activation)(x)
        x = BatchNormalization()(a)
        x = Dense((2000 - i*500), activation=activation)(x)
        x = BatchNormalization()(x)
        x = keras.layers.concatenate([a, x])
    a = Dense(250, activation=activation)(x)
    x = BatchNormalization()(a)
    x = Dense(250, activation=activation)(x)
    x = BatchNormalization()(x)
    x = keras.layers.concatenate([a, x])
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    
elif modelNumber == 4:
    inputs = Input(shape=(28,28,1))
    x = Reshape((784,),input_shape=(28,28,1))(inputs)
    a = Dense(2000, activation=activation)(x)
    x = BatchNormalization()(a)
    x = Dense(1500, activation=activation)(x)
    x = keras.layers.concatenate([a, x])
    a = Dense(1000, activation=activation)(x)
    x = BatchNormalization()(a)
    x = Dense(500, activation=activation)(x)
    x = keras.layers.concatenate([a, x])
    x = Dense(250, activation=activation)(x)
    x = BatchNormalization()(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    
elif modelNumber == 6:
    inputs = Input(shape=(28,28,1))
    x = Reshape((784,),input_shape=(28,28,1))(inputs)
    for i in range(1,6):
        a = Dense(1000, activation=activation)(x)
        x = BatchNormalization()(a)
        x = Dense(1000, activation=activation)(x)
        x = BatchNormalization()(x)
        x = keras.layers.concatenate([a, x])
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

elif modelNumber == 7:
    model = Sequential()
    model.add( Reshape((784,),input_shape=(28,28,1)))
    model.add(Dense(1500, activation=activation))
    model.add(Dense(1000, activation=activation))
    model.add(Dense(500, activation=activation))
    model.add(Dense(10, activation='softmax'))

model.summary()

plot_model(model, to_file=filemodel, show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1, momentum=0.5, nesterov=True),
              metrics=['accuracy'])

# Gestion de l'augmentation des données 
if augmentation :
    if rotation:
        # Data augmentation
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images
        
        datagen.fit(x_train)
        
        for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
            	# create a grid of 3x3 images
            	for i in range(0, 9):
                    pyplot.subplot(330 + 1 + i)
                    pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            	# show the plot
            	pyplot.show()
            	break
    elif elastic:
        datagen = ImageDataGenerator(featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=False,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=elastic_transform)
        
        datagen.fit(x_train)
        
        for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
            	# create a grid of 3x3 images
            	for i in range(0, 9):
                    pyplot.subplot(330 + 1 + i)
                    pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            	# show the plot
            	pyplot.show()
            	break
    elif shift:
        # Data augmentation
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images
        
        datagen.fit(x_train)
        
        for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
            	# create a grid of 3x3 images
            	for i in range(0, 9):
                    pyplot.subplot(330 + 1 + i)
                    pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            	# show the plot
            	pyplot.show()
            	break
    else :
        print("Error, no data augmentation type specified but data augmentation is ON, weird")
    
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
                                  steps_per_epoch=(2*len(x_train) / batch_size),
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=(x_val, y_val),
                                  callbacks=[csv_logger, early_stopping])
else :

    print("Size of training set : " + str(x_train.shape[0]))
    print("Size of testing set : " + str(x_test.shape[0]))
    
    # Validation sur 2% du training set
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[csv_logger, early_stopping])
    

# Évaluation du modèle sur le testing set
score = model.evaluate(x_test, y_test, verbose=0)

# Écriture dans le fichier des résultats du test
fout = open(filename, 'a')
fout.write("test loss,test accuracy\n\n")

fout.write(str(score[0]) + "," + str(score[1]) + "\n")
fout.close()

print('Test loss:', score[0])
print('Test accuracy:', score[1])