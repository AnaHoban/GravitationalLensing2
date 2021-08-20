import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_tuner import RandomSearch

#### LOSS FUNCTIONS ####
def custom_loss_cfis(y_true, y_pred):
    return keras.losses.MSE(y_true*np.sqrt(weights_cfis), y_pred*np.sqrt(weights_cfis))

def custom_loss_ps1(y_true, y_pred):
    return keras.losses.MSE(y_true*np.sqrt(weights_ps1), y_pred*np.sqrt(weights_ps1))

def custom_loss_all(y_true, y_pred):
    return keras.losses.MSE(y_true*np.sqrt(weights_all), y_pred*np.sqrt(weights_all))


#### MODELS ####

def create_autoencoder1(shape):
    '''Autoencoder with pooling layers'''
    input_img = keras.Input(shape=shape)
    
    x = keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128)(x)
    encoded = keras.layers.Dense(1024*4)(x)
    
    x = keras.layers.Reshape((32,32,4))(encoded)
    x = keras.layers.UpSampling2D((2,2))(x)
    x = keras.layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2,2))(x)
    x = keras.layers.Conv2DTranspose(16, kernel_size=3, activation='relu', padding='same')(x)
    decoded = keras.layers.Conv2D(shape[2], (3,3), activation='linear', padding='same')(x)
    
    return keras.Model(input_img, decoded)


def create_autoencoder2(shape):
    '''Autoencoder with convolutional layers'''
    input_img = keras.Input(shape=shape)
    x = keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same')(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    y = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(input_img)
    y = keras.layers.BatchNormalization()(y)
    encoded = keras.layers.Add()([x,y])
    
    x = keras.layers.Conv2DTranspose(32, kernel_size=4, activation='relu', padding='same')(encoded)
    x = keras.layers.Conv2DTranspose(16, kernel_size=4, activation='relu', padding='same')(x)
    decoded = keras.layers.Conv2D(shape[2], kernel_size=3, activation='linear', padding='same')(x)
    
    return keras.Model(input_img, decoded)



num_cands = 232
num_rand = 34*num_cands
initial_bias = np.log10([num_rand/num_cands])
cutout_size = 64

bands = 2
def MSE_with_uncertainty(y_true, y_pred): 
    weights = y_pred[...,bands:] 
    y_pred_image = y_pred[...,:bands]
    
    return K.square(tf.math.multiply((y_true - y_pred_image), weights) )

model_path = "../../Models/job15.h5"
autoencoder_cfis = keras.models.load_model(model_path, custom_objects={'MSE_with_uncertainty': MSE_with_uncertainty})
encoder = keras.Model(autoencoder_cfis.input, autoencoder_cfis.layers[7].output)

for i in range(len(encoder.layers)):
    encoder.layers[i].trainable = False

def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr*tf.math.exp(-0.5)
    
def step_decay(epoch, lr):
    drop = 0.5
    epochs_drop = 20.0
    lr = lr * tf.math.pow(drop,  
           tf.math.floor((1+epoch)/epochs_drop))
    return lr

def tune_classifier(hp, output_bias = initial_bias):
    '''Binary classifier'''
    model = keras.Sequential(encoder)
    model.add(keras.layers.Flatten())
    
    for i in range(hp.Int("num_layers", 1, 10)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )

    if output_bias is not None:
        model.add(keras.layers.Dense(1, activation="sigmoid",bias_initializer=keras.initializers.Constant(output_bias))) 
    else:
        model.add(keras.layers.Dense(1, activation="sigmoid")) 
    
    optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]))
    #callback = keras.callbacks.LearningRateScheduler(step_decay)
    callback = keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=('Recall', "Precision"))

    return model

def create_classifier( output_bias = initial_bias):
    '''Binary classifier'''
    model = keras.Sequential(encoder)
    model.add(keras.layers.Flatten())
    
    #model.add(keras.layers.Dense(1000,kernel_initializer=tf.keras.initializers.RandomNormal(seed=42)))
    keras.layers.Dense(64)
    keras.layers.Dense(32)
    keras.layers.Dense(16)
                    
    
    if output_bias is not None:
        model.add(keras.layers.Dense(1, activation="sigmoid",bias_initializer=keras.initializers.Constant(output_bias))) 
    else:
        model.add(keras.layers.Dense(1, activation="sigmoid")) 
    

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    #callback = keras.callbacks.LearningRateScheduler(step_decay)
    callback = keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=('Recall', "Precision"))

    return model
