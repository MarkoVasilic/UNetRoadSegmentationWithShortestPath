from typing import Deque
import cv2
import keras
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog as fd

tk.Tk().withdraw() 

TRAINING_INPUT_PATH = fd.askopenfile().name

PROCESSED_SIZE = 1408
IMAGE_SKIP = 128
IMAGE_SIZE = 256

up_move_offset = 0
number_of_dots = 0
dots = [(0, 0), (0, 0)]
img = cv2.imread(TRAINING_INPUT_PATH)[0 : PROCESSED_SIZE, 0 : PROCESSED_SIZE]
active_image = img.copy()
road_map_image = None

def show_active_image():
    cv2.imshow('image', active_image[100 * up_move_offset: 100 * up_move_offset + 772,:])

def split_image_opt():
    global img
    return_input_images = []
    for i in range (0, 10):
        for j in range (0, 10):
            return_input_images.append(img[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE]/255)
    return return_input_images

def predict_opt(model):
    global road_map_image
    out = np.zeros((PROCESSED_SIZE,PROCESSED_SIZE,1))
    mask = np.zeros((PROCESSED_SIZE,PROCESSED_SIZE,1))
    ones =np.ones((IMAGE_SIZE,IMAGE_SIZE,1))
    return_input_images = split_image_opt()
    for i in range (0, 10):
        for j in range (0, 10):
            y = model(np.array([return_input_images[i*10+j]]))
            out[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE] = np.add(out[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE],y[0])
            mask[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE] = np.add(mask[i*IMAGE_SKIP:i*IMAGE_SKIP+IMAGE_SIZE, j*IMAGE_SKIP:j*IMAGE_SKIP+IMAGE_SIZE], ones) 
    # temp = np.where(out/mask > 0.3, 255, 0)
    temp = out/mask * 255
    cv2.imwrite("./temp.png", temp)
    road_map_image = cv2.imread("./temp.png", cv2.IMREAD_GRAYSCALE)

def load_model():
    model = keras.models.load_model('models_24/model_22.h5')
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss="binary_crossentropy", metrics = ['accuracy'], run_eagerly=True)
    return model

def get_neighbors_pixels(pos):
    neighbors = []
    if pos[0] > 0:
        neighbors.append((pos[0]-1, pos[1]))
    if pos[1] > 0:
        neighbors.append((pos[0], pos[1]-1))
    if pos[0] < PROCESSED_SIZE-1:
        neighbors.append((pos[0]+1, pos[1]))
    if pos[1] < PROCESSED_SIZE-1:
        neighbors.append((pos[0], pos[1]+1))
    return neighbors

def clossest_road():
    global dots, road_map_image, active_image
    new_dots = []
    temp = None
    for dot in dots:
        queue = Deque()
        dist_map = -np.ones((PROCESSED_SIZE,PROCESSED_SIZE))
        dist_map[dot[1],dot[0]] = 0
        queue.append(dot)
        while len(queue) > 0:
            temp = queue.popleft()
            if road_map_image[temp[1],temp[0]] > 80:
                break
            for n in get_neighbors_pixels(temp):
                if dist_map[n[1],n[0]] == -1:
                    dist_map[n[1],n[0]] = dist_map[temp[1],temp[0]] + 1
                    queue.append(n)
        new_dots.append(temp)
        queue.clear()
        queue.append(temp)
        while dist_map[temp[1],temp[0]] != 0:
            for n in get_neighbors_pixels(temp):
                if dist_map[temp[1],temp[0]] == dist_map[n[1],n[0]] + 1:
                    temp = n
                    active_image = cv2.circle(active_image, (temp[0],temp[1]), radius=2, color=(0, 255, 255), thickness=-1)
                    break
    dots = new_dots

def find_path():
    global dots, road_map_image, active_image
    start = dots[0]
    end = dots[1]
    queue = Deque()
    dist_map = -np.ones((PROCESSED_SIZE,PROCESSED_SIZE))
    dist_map[start[1],start[0]] = 0
    queue.append(start)
    while len(queue) > 0:
        temp = queue.popleft()
        if temp[1] == end[1] and end[0] == temp[0]:
            break
        for n in get_neighbors_pixels(temp):
            if dist_map[n[1],n[0]] == -1 and road_map_image[n[1],n[0]] > 50:
                dist_map[n[1],n[0]] = dist_map[temp[1],temp[0]] + 1
                queue.append(n)
    queue.clear()
    if temp[1] == end[1] and end[0] == temp[0]:
        queue.append(temp)
        while dist_map[temp[1],temp[0]] != 0:
            for n in get_neighbors_pixels(temp):
                if dist_map[temp[1],temp[0]] == dist_map[n[1],n[0]] + 1:
                    temp = n
                    active_image = cv2.circle(active_image, (temp[0],temp[1]), radius=2, color=(255, 0, 0), thickness=-1)
                    break
    else:
        print("Path not found!")
        tk.messagebox.showerror("Not found", "Path between two selected points not found!")

def mouse_click(event, x, y, flags, param):
    global active_image, up_move_offset, number_of_dots
    if event == cv2.EVENT_LBUTTONDOWN:
        if number_of_dots < 2:
            active_image = cv2.circle(active_image, (x,y + up_move_offset * 100), radius=5, color=(0, 0, 255), thickness=-1)
            dots[number_of_dots] = (x,y + up_move_offset * 100)
            number_of_dots = number_of_dots + 1
            show_active_image()
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags < 0 and up_move_offset < 7:
            up_move_offset = up_move_offset + 1
            show_active_image()
        
        if flags > 0 and up_move_offset > 0:
            up_move_offset = up_move_offset - 1
            show_active_image()


model = load_model()
predict_opt(model)
show_active_image()
cv2.setMouseCallback('image', mouse_click)

while True:
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 27:
        break
    #print("The key code is:" + str(full_key_code))
    if full_key_code == 2621440 and up_move_offset < 7:
        up_move_offset = up_move_offset + 1
        show_active_image()
        
    if full_key_code == 2490368 and up_move_offset > 0:
        up_move_offset = up_move_offset - 1
        show_active_image()
    
    #reset = 114
    if full_key_code == 114:
        active_image = img.copy()
        number_of_dots = 0
        show_active_image()

    #find path enter = 13
    if full_key_code == 13 and number_of_dots == 2:
        clossest_road()
        find_path()
        show_active_image()

    if full_key_code == 49:
        active_image = img.copy()
        show_active_image()

    if full_key_code == 50:
        active_image = cv2.cvtColor(road_map_image.copy(), cv2.COLOR_GRAY2BGR)
        show_active_image()

    if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
        break

    
cv2.destroyAllWindows()

