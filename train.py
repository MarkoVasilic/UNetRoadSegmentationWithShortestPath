import keras
import numpy as np
import cv2
import os
import sys
import tensorflow as tf

IMAGE_SIZE = 256
ORGINAL_SIZE = 1500
BASE_NUM_FEATURES = 24
LOAD_IMAGE_COUNT = 32
TRAINING_INPUT_PATH = "./mass_roads/train/sat/"
TRAINING_OUTPUT_PATH = "./mass_roads/train/map/"
LIST_OF_PICS = os.listdir(TRAINING_OUTPUT_PATH)
EPOCHS = 4

def load_images(ID):
    return_output_images = []
    return_input_images = []
    for i in range(LOAD_IMAGE_COUNT):
        file_name = LIST_OF_PICS[(i + ID * LOAD_IMAGE_COUNT) % 800]
        output_image = cv2.imread(TRAINING_OUTPUT_PATH + file_name, cv2.IMREAD_GRAYSCALE)
        input_image = cv2.imread(TRAINING_INPUT_PATH + file_name + "f") 
        gray_scale_image = cv2.imread(TRAINING_INPUT_PATH + file_name + "f", cv2.IMREAD_GRAYSCALE)
        output_image, input_image = split_image(output_image, input_image, gray_scale_image)
        return_output_images = return_output_images + output_image
        return_input_images = return_input_images + input_image
    return return_output_images, return_input_images

def split_image(output_image, input_image, gray_scale_image):
    return_output_images = []
    return_input_images = []
    for i in range (0, 5):
        for j in range (0, 5):
            temp = gray_scale_image[i*IMAGE_SIZE:(i+1)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE]
            if np.sum(temp == 255) < 6000:
                return_input_images.append(input_image[i*IMAGE_SIZE:(i+1)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE]/255)
                return_output_images.append(output_image[i*IMAGE_SIZE:(i+1)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE]/255)
    return return_output_images, return_input_images

def model_create():
    inputs = keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    conv1 = keras.layers.Conv2D(BASE_NUM_FEATURES, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(BASE_NUM_FEATURES, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(BASE_NUM_FEATURES * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(BASE_NUM_FEATURES * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(BASE_NUM_FEATURES * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(BASE_NUM_FEATURES * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = keras.layers.Conv2D(BASE_NUM_FEATURES * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(BASE_NUM_FEATURES * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = keras.layers.Conv2D(BASE_NUM_FEATURES * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = keras.layers.Conv2D(BASE_NUM_FEATURES * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = keras.layers.Dropout(0.5)(conv5)

    up6 = keras.layers.Conv2DTranspose(BASE_NUM_FEATURES * 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=(2,2))((drop5))
    merge6 = keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = keras.layers.Conv2D(BASE_NUM_FEATURES * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = keras.layers.Conv2D(BASE_NUM_FEATURES * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = keras.layers.Conv2DTranspose(BASE_NUM_FEATURES * 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=(2,2))((conv6))
    merge7 = keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(BASE_NUM_FEATURES * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv2D(BASE_NUM_FEATURES * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = keras.layers.Conv2DTranspose(BASE_NUM_FEATURES * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=(2,2))((conv7))
    merge8 = keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(BASE_NUM_FEATURES * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv2D(BASE_NUM_FEATURES * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = keras.layers.Conv2DTranspose(BASE_NUM_FEATURES, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=(2,2))((conv8))
    merge9 = keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = keras.layers.Conv2D(BASE_NUM_FEATURES, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv2D(BASE_NUM_FEATURES, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = keras.layers.Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    return  keras.Model(inputs, conv10)

def main():
    ID = int(sys.argv[1])
    model = None
    if ID == 0:
        model = model_create()
    else:
        model = keras.models.load_model('auto.h5')
    training_output, training_input = load_images(ID)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss="binary_crossentropy", metrics = ['accuracy'], run_eagerly=True)
    model.fit(x = np.array(training_input), y=np.array(training_output), batch_size=2 ,epochs=EPOCHS, shuffle=True)
    model.save("auto.h5")

main()