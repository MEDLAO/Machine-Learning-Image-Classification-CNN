# import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt


# Avoid OOM errors by setting GPU memory consumption growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# 1 - Remove dodgy images :
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

img = cv2.imread(os.path.join(data_dir, 'happy_people', '7-Habits-of-Happy-People-image.jpeg'))
# print(img.shape)  # (894, 1280, 3) height, width and number of color channels (RGB)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

for image_class in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, image_class)
    # ensure that the item is a directory
    if os.path.isdir(class_dir):
        for image in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))




