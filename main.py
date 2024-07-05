import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np


# Avoid OOM errors by setting GPU memory consumption growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# 1 - Building a Data Pipeline
# Remove dodgy images :
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

# Load Data :
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
# get another batch from the iterator
batch = data_iterator.next()
# images represented as numpy arrays
print(batch[0].shape)  # (32, 256, 256, 3)
# Class 0 = happy people / Class 1 = sad people
print(batch[1])  # [0 0 1 1 0 1 0 0 0 0 1 0 0 1 1 1 0 1 0 0 0 0 1 1 1 1 0 1 0 1 0 0]

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

# plt.tight_layout()
# plt.show()

# 2 - Preprocessing Data
# Shuffle the dataset
data = data.shuffle(buffer_size=len(data), reshuffle_each_iteration=False)

# Scaling Images
scaled = batch[0] / 255
# print(scaled.max())  # 1.0

data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

print(batch[0].max())
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
# plt.show()

# Partitioning the Dataset
print(len(data))  # len(data) = 9
train_size = int(len(data)*.7)
# print(train_size)  # 6
val_size = int(len(data)*.2) + 1
# print(val_size)  # 2
test_size = int(len(data)*.1) + 1
# print(test_size) # 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# 3 - Building the Deep Neural Network
model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
print(model.summary())

# Train
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
print(hist)

# Epoch 1/20
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 7s 769ms/step - accuracy: 0.6260 - loss: 0.8251 - val_accuracy: 0.6562 - val_loss: 0.5786
# .
# .
# .
# Epoch 20/20
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 5s 798ms/step - accuracy: 1.0000 - loss: 0.0025 - val_accuracy: 1.0000 - val_loss: 0.0019

# Plot performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
# plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
# plt.show()

# 4 - Evaluating performance
# Evaluate
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

print(len(test))

for batch in test.as_numpy_iterator():
    X, y = batch
    y_hat = model.predict(X)
    pre.update_state(y, y_hat)
    re.update_state(y, y_hat)
    acc.update_state(y, y_hat)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')
# Precision: 1.0, Recall: 1.0, Accuracy: 1.0

img_test = cv2.imread('data/test/happytest.jpg')
plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
plt.show()

resize = tf.image.resize(img_test, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

y_hat_test = model.predict(np.expand_dims(resize/255, 0))
print(f'y_hat_test : {y_hat_test}')

if y_hat_test > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')
