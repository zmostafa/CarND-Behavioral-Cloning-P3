import cv2
import csv 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Read training images info
lines = []
with open('./recorded_data/driving_log.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    for line in csv_reader:
        lines.append(line)

# Read images and Steering measurement
images = []
steering = []
for line in lines:
    source_path = line[0]
    file_name = source_path.split('\\')[-1] # Note the path delimiter for windows 
    current_path = './recorded_data/IMG/' + file_name
    image = cv2.imread(current_path)
    # resize and transform to grayscale for LeNet 
    # image = cv2.resize(image,32,32)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image)
    measurement = line[3]
    steering.append(float(measurement)) 

X_train = np.array(images)
y_train = np.array(steering)

for i in range(10):
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.figure(figsize=(10,10))

# LeNet arch with input (160,320,3)
# model = Sequential([
#     Conv2D(6,(5,5), activation='relu',input_shape=(160,320,3)),
#     MaxPooling2D(pool_size=(2,2),strides=2),
#     Conv2D(16,(5,5), activation='relu'),
#     MaxPooling2D(pool_size=(2,2),strides=2),
#     Flatten(),
#     Dense(120),
#     Dense(84),
#     Dense(1)
# ])


# # Compile the model 
# model.compile(loss='mse', optimizer='adam')

# # Model Training 
# model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True, verbose=2)

# # Saving model
# model.save('model_lenet.h5')