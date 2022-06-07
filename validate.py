from tabnanny import verbose
import keras
import numpy as np
import cv2
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

IMAGE_SIZE = 256
TRAINING_INPUT_PATH = "./mass_roads/valid/sat/"
TRAINING_OUTPUT_PATH = "./mass_roads/valid/map/"
LIST_OF_PICS = os.listdir(TRAINING_OUTPUT_PATH)

def load_images():
    return_output_images = []
    return_input_images = []
    for file_name in LIST_OF_PICS:
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

def main():
    model = keras.models.load_model('auto.h5')
    validate_output, validate_input = load_images()
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss="binary_crossentropy", metrics = ['accuracy'], run_eagerly=True)
    eval = model.evaluate(np.array(validate_input), np.array(validate_output), verbose=0, batch_size=2)
    print(",".join(str(e) for e in eval))

main()