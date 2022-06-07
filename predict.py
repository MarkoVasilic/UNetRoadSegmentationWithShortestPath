import keras
import numpy as np
import cv2
import tensorflow as tf

IMAGE_SIZE = 256
IMAGE_SKIP = 128
IMAGE_ORIGINAL_SIZE = 1500
IMAGE_PATH = "23429080_15"

def split_image(input_image, gray_scale_image):
    return_input_images = []
    for i in range (0, 5):
        for j in range (0, 5):
            temp = gray_scale_image[i*IMAGE_SIZE:(i+1)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE]
            if np.sum(temp == 255) < 6000:
                return_input_images.append(input_image[i*IMAGE_SIZE:(i+1)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE]/255)
    return return_input_images

def split_image_opt(input_image):
    return_input_images = []
    for i in range (0, 10):
        for j in range (0, 10):
            return_input_images.append(input_image[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE]/255)
    return return_input_images


def test(model):
    img_path = "./mass_roads/test/sat/" + IMAGE_PATH + ".tiff"
    img = split_image(cv2.imread(img_path) ,cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    out = np.zeros((256*5,256*5,1))
    print(out.shape)
    for i in range (0, 5):
        for j in range (0, 5):
            y = model(np.array([img[i*5+j]]))
            out[i*256:(i+1)*256, j*256:(j+1)*256] = y[0]
    cv2.imwrite("./old/" + IMAGE_PATH + ".png", out * 255)

def test_opt(model):
    img_path = "./mass_roads/test/sat/" + IMAGE_PATH + ".tiff"
    img = split_image_opt(cv2.imread(img_path))
    print(len(img))
    out = np.zeros((256*5,256*5,1))
    mask = np.zeros((256*5,256*5,1))
    ones =np.ones((256,256,1))
    print(out.shape)
    for i in range (0, 10):
        for j in range (0, 10):
            y = model(np.array([img[i*10+j]]))
            out[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE] = np.add(out[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE],y[0])
            mask[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE] = np.add(mask[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE], ones) 
    cv2.imwrite("./old/" + IMAGE_PATH + "_opt_22_24param_bit.png", np.where(out/mask > 0.3, 255, 0))

def main():
    model = keras.models.load_model('models/model_94.h5')
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss="binary_crossentropy", metrics = ['accuracy'], run_eagerly=True)
    test_opt(model)

main()